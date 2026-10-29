"""Translation-only camera refinement for module B (optional, off by default).

The refinement is intentionally small and is *not* a pose solver: it optimizes
one 3-vector per training view (world-space camera translation) for the last M
steps of the FastGS continuation, while rotation and intrinsics stay frozen.
The full training entry point lives in ``research.depth_reg`` and enables this
module only when ``--pose-lr > 0``:

    python -m research.depth_reg ... --pose-lr 5e-5 --pose-start-frac 0.7

This file is also a thin wrapper that turns the feature on (unless
``--pose-lr`` is already given) so the pose variant can be launched directly:

    python -m research.pose_refine ... --init_ply ... --iterations 12000 \
        --depth-npz .../resplat_depth.npz --pose-lr 1e-4

Camera convention
-----------------
FastGS stores ``world_view_transform`` W (the row-vector view matrix) and the
rasterizer computes ``x_cam = W^T [p, 1]``.  With ``W[:3, :3] = R_w2c`` and
``W[3, :3] = T = -R_w2c C``, moving the camera centre ``C -> C + delta`` gives

    T' = T - R_w2c delta
    W'[3, :3] = T'
    full_proj' = W' P
    camera_center' = C + delta

Only ``world_view_transform``, ``full_proj_transform`` and ``camera_center``
are consumed by ``render_fastgs``; ``R``/``T`` on the camera are left at their
COLMAP values.  Poses are never written back to disk, so the held-out
evaluation (which loads test cameras from the scene) is unaffected except
through the Gaussians that were optimized.

``--self-check`` validates the transform math, the optimiser update and the
argument plumbing with tiny CPU tensors when torch is available.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import torch
except ImportError:  # CPU self-check still works without torch
    torch = None

DEFAULT_POSE_LR = 5e-5


def _require_torch():
    if torch is None:
        raise RuntimeError("camera refinement needs torch (run on the fastgs GPU env)")


def camera_forward(cam):
    """Unit camera forward (+Z) direction in world space.

    ``world_view_transform[:3, :3]`` is the world->camera rotation, so its third
    row is the camera-to-world forward axis.  Translation-only refinement never
    changes it, but it is cheap to recompute.
    """
    axis = cam.world_view_transform[2, :3]
    return axis / axis.norm().clamp_min(1e-8)


def apply_translation(base_world_view, base_projection, base_center, delta):
    """Return (W', full_proj', C') for a world-space translation ``delta``."""
    w2c = base_world_view[:3, :3]
    shift = w2c @ delta
    world_view = base_world_view.clone()
    world_view[3, :3] = base_world_view[3, :3] - shift
    full_proj = world_view @ base_projection
    center = base_center + delta
    return world_view, full_proj, center


class PoseRefiner:
    """Shared 3-DoF translation table over one or more camera resolutions."""

    def __init__(self, cameras, lr, max_shift=None, weight_decay=0.0, device=None):
        _require_torch()
        cameras = list(cameras)
        if not cameras:
            raise ValueError("PoseRefiner needs at least one camera")
        if lr <= 0:
            raise ValueError("pose lr must be positive")
        if max_shift is not None and max_shift < 0:
            raise ValueError("pose max shift must be non-negative")

        names = sorted({str(cam.image_name) for cam in cameras})
        self.names = names
        self._index = {name: i for i, name in enumerate(names)}
        self._base = {}
        self._cameras = {}
        first = cameras[0]
        device = device or first.world_view_transform.device
        for cam in cameras:
            cam_id = id(cam)
            if cam_id in self._base:
                continue
            name = str(cam.image_name)
            index = self._index[name]
            base_world_view = cam.world_view_transform.detach().clone()
            base_projection = cam.projection_matrix.detach().clone()
            base_center = cam.camera_center.detach().clone()
            self._base[cam_id] = dict(name=name, index=index,
                                      world_view=base_world_view,
                                      projection=base_projection,
                                      center=base_center)
            self._cameras[cam_id] = cam

        self.lr = float(lr)
        self.max_shift = None if max_shift is None else float(max_shift)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.delta = torch.nn.Parameter(
            torch.zeros((len(names), 3), dtype=torch.float32, device=device))
        self.optimizer = torch.optim.Adam(
            [self.delta], lr=self.lr, betas=(0.9, 0.999),
            weight_decay=self.weight_decay)
        self._applied = False
        self._frozen = False

    @property
    def num_views(self):
        return len(self.names)

    def _entry(self, cam):
        entry = self._base.get(id(cam))
        if entry is None:
            raise KeyError(
                f"camera {getattr(cam, 'image_name', '?')!r} was not registered "
                "with this PoseRefiner")
        return entry

    def apply(self, cam):
        """Write differentiable translated pose tensors onto ``cam``."""
        entry = self._entry(cam)
        delta = self.delta[entry["index"]]
        world_view, full_proj, center = apply_translation(
            entry["world_view"], entry["projection"], entry["center"], delta)
        cam.world_view_transform = world_view
        cam.full_proj_transform = full_proj
        cam.camera_center = center
        self._applied = True
        return cam

    def restore(self, cam):
        """Undo ``apply`` (original COLMAP pose); used by tests and fallbacks."""
        entry = self._entry(cam)
        world_view = entry["world_view"].clone()
        cam.world_view_transform = world_view
        cam.full_proj_transform = world_view @ entry["projection"]
        cam.camera_center = entry["center"].clone()
        return cam

    def freeze(self):
        """Keep the current refinement but detach it from the autograd graph."""
        if not self._applied:
            return
        with torch.no_grad():
            for cam_id, entry in self._base.items():
                cam = self._cameras[cam_id]
                delta = self.delta.detach()[entry["index"]]
                world_view, full_proj, center = apply_translation(
                    entry["world_view"], entry["projection"], entry["center"], delta)
                cam.world_view_transform = world_view
                cam.full_proj_transform = full_proj
                cam.camera_center = center
        self._frozen = True

    def step(self):
        if self.delta.grad is None:
            return
        self.optimizer.step()
        if self.max_shift is not None:
            with torch.no_grad():
                norm = self.delta.norm(dim=1, keepdim=True)
                scale = (self.max_shift / norm.clamp_min(1e-12)).clamp(max=1.0)
                self.delta.mul_(scale)
        self.optimizer.zero_grad(set_to_none=True)

    def delta_norms(self):
        with torch.no_grad():
            norm = self.delta.detach().norm(dim=1)
            return dict(mean=float(norm.mean().item()),
                        max=float(norm.max().item()))

    def summary(self):
        norms = self.delta_norms()
        return dict(n_views=self.num_views, lr=self.lr, max_shift=self.max_shift,
                    weight_decay=self.weight_decay,
                    mean_shift=norms["mean"], max_observed_shift=norms["max"],
                    frozen=self._frozen)


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _make_camera(center, forward, image_name="view0"):
    """Synthetic FastGS-like camera: W, P, camera_center, image_name."""
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(helper @ forward)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    right = np.cross(helper, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    c2w = np.stack([right, up, forward], axis=1)
    center = np.asarray(center, dtype=np.float64)
    w2c = c2w.T
    translation = -w2c @ center
    world_view = np.eye(4, dtype=np.float32)
    world_view[:3, :3] = w2c
    world_view[3, :3] = translation
    projection = np.eye(4, dtype=np.float32)
    projection[0, 0] = 1.2
    projection[1, 1] = 1.1

    class Camera:
        pass

    cam = Camera()
    cam.image_name = image_name
    cam.world_view_transform = torch.tensor(world_view)
    cam.projection_matrix = torch.tensor(projection)
    cam.camera_center = torch.tensor(center.astype(np.float32))
    return cam


def _expect_error(call, label, checks):
    try:
        call()
    except (ValueError, KeyError):
        return checks + 1
    raise AssertionError(f"self-check: expected an error for {label}")


def self_check():
    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(f"self-check: {message}")
        checks += 1

    # ---- argument plumbing (works without torch) --------------------------
    from research.depth_reg import build_parser, validate_args

    parser = build_parser()
    base_argv = ["--source", "/tmp/work", "--model_path", "/tmp/model",
                 "--init_ply", "/tmp/init.ply", "--iterations", "1000"]
    args = validate_args(parser.parse_args(base_argv))
    check(args.pose_lr == 0.0, "pose refinement is off by default")
    check(args.pose_start_frac >= 0.0 and args.pose_steps >= -1,
          "pose window defaults parse")
    args = validate_args(parser.parse_args(base_argv + ["--pose-lr", "5e-05"]))
    check(args.pose_lr == 5e-05, "--pose-lr enables the feature")

    injected = inject_pose_defaults(base_argv)
    check("--pose-lr" in injected, "wrapper injects --pose-lr")
    injected_args = validate_args(parser.parse_args(injected))
    check(injected_args.pose_lr == DEFAULT_POSE_LR, "wrapper default lr")
    check("--pose-lr" not in inject_pose_defaults(base_argv + ["--pose-lr", "0"]),
          "explicit --pose-lr 0 is preserved")
    bad_parser = build_parser()
    try:
        validate_args(bad_parser.parse_args(
            base_argv + ["--pose-lr", "1e-4", "--pose-start-frac", "0.9",
                         "--pose-steps", "10"]))
        # start = 900, steps = 10 -> end 909 < iterations, valid
        checks += 1
    except ValueError:
        raise AssertionError("self-check: valid pose window rejected")

    if torch is None:
        print(f"self-check OK: {checks} checks (argument plumbing only; "
              "torch unavailable, transform math skipped)")
        return checks

    # ---- translation transform -------------------------------------------
    center = np.array([1.0, -2.0, 0.5])
    forward = np.array([0.2, 0.9, 0.3])
    cam = _make_camera(center, forward)
    entry_w = cam.world_view_transform.clone()
    entry_p = cam.projection_matrix.clone()
    entry_c = cam.camera_center.clone()
    delta = torch.tensor([0.05, -0.02, 0.03])

    world_view, full_proj, new_center = apply_translation(entry_w, entry_p, entry_c, delta)
    check(torch.allclose(world_view[:3, :3], entry_w[:3, :3]),
          "translation leaves rotation block intact")
    check(torch.allclose(world_view[3, 3], entry_w[3, 3]), "W[3,3] unchanged")
    expected_shift = entry_w[:3, :3] @ delta
    check(torch.allclose(world_view[3, :3], entry_w[3, :3] - expected_shift),
          "W bottom row is T - R_w2c delta")
    check(torch.allclose(full_proj, world_view @ entry_p), "full_proj = W' P")
    check(torch.allclose(new_center, entry_c + delta), "centre moves by delta")

    point = torch.tensor([2.0, -1.0, 3.0])
    forward_vec = camera_forward(cam)
    base_z = forward_vec @ (point - entry_c)
    new_z = forward_vec @ (point - new_center)
    check(torch.allclose(new_z, base_z - forward_vec @ delta, atol=1e-6),
          "camera-space z decreases by f dot delta")

    # ---- PoseRefiner identity / apply / restore ---------------------------
    refiner = PoseRefiner([cam], lr=1e-2, max_shift=0.5)
    check(refiner.num_views == 1, "one view registered")
    refiner.apply(cam)
    check(torch.allclose(cam.world_view_transform, entry_w, atol=1e-6),
          "zero delta is identity")
    check(torch.allclose(cam.full_proj_transform, entry_w @ entry_p, atol=1e-6),
          "zero delta full_proj identity")
    check(torch.allclose(cam.camera_center, entry_c, atol=1e-6),
          "zero delta centre identity")

    with torch.no_grad():
        refiner.delta[0] = delta
    refiner.apply(cam)
    check(torch.allclose(cam.camera_center, entry_c + delta, atol=1e-6),
          "apply writes translated centre")
    refiner.restore(cam)
    check(torch.allclose(cam.camera_center, entry_c, atol=1e-6),
          "restore recovers the base pose")

    # ---- gradient descent on a depth-like loss ---------------------------
    target = 0.0
    initial = None
    seen_grad = False
    for _ in range(40):
        refiner.apply(cam)
        z = camera_forward(cam) @ (point - cam.camera_center)
        loss = (z - target) ** 2
        if initial is None:
            initial = float(loss.detach())
        refiner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if refiner.delta.grad is not None and float(refiner.delta.grad.norm()) > 0:
            seen_grad = True
        refiner.step()
    refiner.apply(cam)
    final = float(((camera_forward(cam) @ (point - cam.camera_center) - target) ** 2).detach())
    check(seen_grad, "loss backpropagates into the translation table")
    check(final < initial, "Adam reduces the depth-like loss")
    norms = refiner.delta_norms()
    check(norms["max"] <= 0.5 + 1e-5, "delta is clamped to --pose-max-shift")

    # clamp is enforced even for a huge step
    refiner2 = PoseRefiner([_make_camera(center, forward, "clamp")], lr=10.0,
                           max_shift=0.01)
    refiner2.delta.grad = torch.ones_like(refiner2.delta)
    refiner2.step()
    check(refiner2.delta_norms()["max"] <= 0.01 + 1e-6, "projection clamp")

    # ---- registration errors and sharing ---------------------------------
    stranger = _make_camera([0, 0, 0], [0, 0, 1], "stranger")
    checks = _expect_error(lambda: refiner.apply(stranger), "unregistered camera", checks)

    cam_a = _make_camera(center, forward, "shared")
    cam_b = _make_camera(center + 0.1, forward, "shared")
    base_a = cam_a.camera_center.clone()
    base_b = cam_b.camera_center.clone()
    shared = PoseRefiner([cam_a, cam_b], lr=1e-3)
    check(shared.num_views == 1, "same image_name shares one translation")
    with torch.no_grad():
        shared.delta[0] = torch.tensor([0.01, 0.0, 0.0])
    shared.apply(cam_a)
    shared.apply(cam_b)
    check(torch.allclose(cam_a.camera_center - base_a,
                         cam_b.camera_center - base_b, atol=1e-6),
          "shared translation applied to every resolution")

    print(f"self-check OK: {checks} checks "
          "(pose args, W/P/C update, gradient step, clamp, sharing)")
    return checks


def inject_pose_defaults(argv):
    argv = list(argv)
    has_pose = any(a == "--pose-lr" or a.startswith("--pose-lr=") for a in argv)
    if not has_pose:
        argv = ["--pose-lr", str(DEFAULT_POSE_LR)] + argv
    return argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-check" in argv:
        self_check()
        return 0
    from research.depth_reg import main as train_main
    return train_main(inject_pose_defaults(argv))


if __name__ == "__main__":
    raise SystemExit(main())
