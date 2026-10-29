"""Research-only GaussianModel variant with a tunable FastGS densification gate.

The pinned FastGS model hardcodes ``importance_score > 5`` as the densification
gate and never caps the model size. For a *short* continuation from ReSplat's
initialization almost no Gaussian is flagged by more than five sampled views, so
FastGS mostly prunes (455k -> 210k on DL3DV) instead of growing detail. This
subclass keeps the upstream algorithm and exposes the two numbers we need to
trade quality against step time:

* ``densify_count_thresh`` - how many sampled views must flag a Gaussian
* ``max_gaussians``        - hard cap, lowest (opacity * mean scale) pruned first

Nothing upstream is modified; we only subclass it.
"""
import torch

from scene.gaussian_model import GaussianModel  # FastGS package, read-only


class AggressiveGaussianModel(GaussianModel):
    """FastGS gaussian model with a configurable densification gate and cap."""

    densify_count_thresh = 2
    max_gaussians = 1_500_000

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii,
                                 args, importance_score=None, pruning_score=None):
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh,
                                      True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh,
                                          True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        metric_mask = importance_score > self.densify_count_thresh

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs),
                                          big_points_ws)

        scores = 1 - pruning_score
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget,
                                                replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)

        # Size cap: drop the least opaque / largest Gaussians once over budget.
        count = self.get_xyz.shape[0]
        if self.max_gaussians and count > self.max_gaussians:
            keep = int(self.max_gaussians)
            quality = (self.get_opacity.squeeze(-1)
                       * (1.0 / (self.get_scaling.mean(dim=1) + 1e-9)))
            threshold = torch.topk(quality, keep).values[-1]
            self.prune_points(quality < threshold)

        opacities_new = self.inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self.tmp_radii = None
        torch.cuda.empty_cache()
