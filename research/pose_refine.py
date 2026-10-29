"""Translation-only camera refinement for module B (optional, off by default).

The refinement is intentionally small: one world-space translation per training
view (3 params, no rotation, no intrinsics), optimized jointly with the
Gaussians for the last M steps.  The entry point is ``research.depth_reg`` and
the feature is enabled only when ``--pose-lr > 0``:

    python -m research.depth_reg ... --pose-lr 5e-5 --pose-start-frac 0.7

This file is also a thin wrapper that turns the feature on (unless
``--pose-lr`` is already given) so the pose variant can be launched directly:

    python -m research.pose_refine ... --init_ply ... --iterations 12000 \
        --depth-npz .../resplat_depth.npz --pose-lr 1e-4

Why the Gaussians move instead of the camera
--------------------------------------------
The pinned FastGS/Diff-Gaussian rasterizer receives ``viewmatrix``,
``projmatrix`` and ``campos`` inside ``GaussianRasterizationSettings``, which is
not a tensor argument of the autograd Function, so gradients do not flow to
camera tensors.  A pure camera translation is nevertheless exactly equivalent
to translating the *world* by ``-delta`` and keeping the camera fixed:

    camera-space point under (C + delta)   ==   (R (X - delta - C))
    view direction from (C + delta) to X   ==   (X - delta) - C
    covariance / scale / rotation are translation invariant

So ``wrap()`` returns a read-only Gaussian view with ``get_xyz = xyz - delta``.
The rasterizer backpropagates through ``means3D``, hence through ``delta``, and
the photometric + depth losses supervise the translation exactly as if the
camera had moved.  Nothing on the camera object is mutated, so held-out cameras
and later rendering are unaffected.

Densification note: FastGS's multi-view densification helper renders sampled
cameras with the *unshifted* model; the translation is clamped to a small
fraction of the scene extent, so this second-order inconsistency is accepted
(and documented in research/DEPTH.md).

``--self-check`` validates the camera/translation equivalence, the wrapper, the
optimizer update and the argument plumbing with tiny CPU tensors when torch is
available.
"""
from __future__ import annotations

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

    FastGS stores ``world_view_transform`` in the row-vector layout
    ``W = [[R_c2w, 0], [T^T, 1]]`` (the rasterizer computes ``W^T [p, 1]``), so
    the third column of the rotation block is the camera +Z axis in world space.
    The base camera is never mutated by this module, but the helper is shared
    with ``depth_reg``.
    """
    axis = cam.world_view_transform[:3, 2]
    return axis / axis.norm().clamp_min(1e-8)


def apply_translation(base_world_view, base_projection, base_center, delta):
    """Return (W', full_proj', C') for a world-space camera translation.

    Kept for the equivalence test and documentation; training uses
    :class:`PoseRefiner.wrap` instead because the rasterizer cannot differentiate
    the view matrix.
    """
    # W[:3,:3] is camera-to-world; its transpose is world-to-camera.
    w2c = base_world_view[:3, :3].transpose(0, 1)
    shift = w2c @ delta
    world_view = base_world_view.clone()
    world_view[3, :3] = base_world_view[3, :3] - shift
    full_proj = world_view @ base_projection
    center = base_center + delta
    return world_view, full_proj, center


class TranslatedGaussians:
    """Read-only Gaussian view translated by ``-delta`` (exact camera motion)."""

    def __init__(self, gaussians, delta):
        self._gaussians = gaussians
        self._delta = delta
        self._xyz = gaussians.get_xyz - delta

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._gaussians.get_opacity

    @property
    def get_scaling(self):
        return self._gaussians.get_scaling

    @property
    def get_rotation(self):
        return self._gaussians.get_rotation

    @property
    def get_features(self):
        return self._gaussians.get_features

    @property
    def get_features_dc(self):
        return self._gaussians.get_features_dc

    @property
    def get_features_rest(self):
        return self._gaussians.get_features_rest

    def get_covariance(self, scaling_modifier=1.0):
        return self._gaussians.get_covariance(scaling_modifier)

    def __getattr__(self, name):
        return getattr(self._gaussians, name)


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
        self.lr = float(lr)
        self.max_shift = None if max_shift is None else float(max_shift)
        self.weight_decay = float(weight_decay)
        device = device or cameras[0].world_view_transform.device
        self.device = device
        self.delta = torch.nn.Parameter(
            torch.zeros((len(names), 3), dtype=torch.float32, device=device))
        self.optimizer = torch.optim.Adam(
            [self.delta], lr=self.lr, betas=(0.9, 0.999),
            weight_decay=self.weight_decay)

    @property
    def num_views(self):
        return len(self.names)

    def wrap(self, cam, gaussians):
        """Return the translated Gaussian view for ``cam``'s current iteration."""
        name = str(cam.image_name)
        if name not in self._index:
            raise KeyError(f"camera {name!r} is not registered with this PoseRefiner")
        return TranslatedGaussians(gaussians, self.delta[self._index[name]])

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
                    mean_shift=norms["mean"], max_observed_shift=norms["max"])


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
    # FastGS row-vector view matrix: top-left is camera-to-world, bottom row is T^T.
    world_view[:3, :3] = c2w
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


def _homogeneous(point):
    return np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])


def _cam_coords(world_view, point):
    """Rows-vector convention used by diff-gaussian-rasterization."""
    if hasattr(world_view, "detach"):
        world_view = world_view.detach().cpu().numpy()
    if hasattr(point, "detach"):
        point = point.detach().cpu().numpy()
    w = np.asarray(world_view, dtype=np.float64)
    return (w.T @ _homogeneous(point))[:3]


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
    preserved = inject_pose_defaults(base_argv + ["--pose-lr", "0"])
    check(preserved.count("--pose-lr") == 1
          and preserved[preserved.index("--pose-lr") + 1] == "0",
          "explicit --pose-lr 0 is preserved")

    if torch is None:
        print(f"self-check OK: {checks} checks (argument plumbing only; "
              "torch unavailable, transform math skipped)")
        return checks

    # ---- camera translation == world translation --------------------------
    center = np.array([1.0, -2.0, 0.5])
    forward = np.array([0.2, 0.9, 0.3])
    cam = _make_camera(center, forward)
    base_w = cam.world_view_transform.clone()
    base_p = cam.projection_matrix.clone()
    base_c = cam.camera_center.clone()
    delta = torch.tensor([0.05, -0.02, 0.03])
    point = np.array([2.0, -1.0, 3.0])

    moved_w, moved_full, moved_c = apply_translation(base_w, base_p, base_c, delta)
    check(torch.allclose(moved_w[:3, :3], base_w[:3, :3]),
          "translation leaves the rotation block intact")
    check(torch.allclose(moved_c, base_c + delta), "centre moves by delta")
    check(torch.allclose(moved_full, moved_w @ base_p), "full_proj = W' P")

    moved_coords = _cam_coords(moved_w, point)
    shifted_coords = _cam_coords(base_w, point - delta.numpy())
    check(np.allclose(moved_coords, shifted_coords, atol=1e-6),
          "moved camera on X == fixed camera on X - delta")
    check(abs(moved_coords[2] - shifted_coords[2]) < 1e-6,
          "camera-space depth is preserved by the equivalence")
    forward_vec = camera_forward(cam)
    expected_forward = torch.inverse(base_w)[2, :3]
    check(torch.allclose(forward_vec, expected_forward, atol=1e-6),
          "camera forward matches inverse(W)'s third row")
    check(torch.allclose(forward_vec @ (base_c - moved_c),
                         -(forward_vec @ delta), atol=1e-6),
          "depth change equals -f dot delta")

    # ---- wrapper exposes the translated model only ------------------------
    xyz = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0],
                                           [-1.0, 0.5, 2.0]]))
    fake = type("FakeGaussians", (), {
        "get_xyz": property(lambda self: xyz),
        "get_opacity": property(lambda self: torch.ones(2, 1)),
        "get_scaling": property(lambda self: torch.ones(2, 3)),
        "get_rotation": property(lambda self: torch.ones(2, 4)),
        "get_features_dc": property(lambda self: torch.zeros(2, 1, 3)),
        "get_features_rest": property(lambda self: torch.zeros(2, 15, 3)),
        "get_features": property(lambda self: torch.zeros(2, 16, 3)),
        "active_sh_degree": 3,
    })()
    refiner = PoseRefiner([cam], lr=1e-2, max_shift=0.5)
    check(refiner.num_views == 1, "one view registered")
    wrapped = refiner.wrap(cam, fake)
    check(torch.allclose(wrapped.get_xyz, xyz), "zero delta is identity")
    check(wrapped.active_sh_degree == 3, "non-xyz attributes are forwarded")
    with torch.no_grad():
        refiner.delta[0] = delta
    wrapped = refiner.wrap(cam, fake)
    check(torch.allclose(wrapped.get_xyz, xyz - delta), "wrap shifts xyz by -delta")
    check(torch.allclose(cam.camera_center, base_c), "camera object is not mutated")
    check(torch.allclose(cam.world_view_transform, base_w),
          "view matrix is not mutated")

    # ---- gradient descent on a depth-like loss ---------------------------
    target = 0.0
    initial = None
    seen_grad = False
    for _ in range(40):
        wrapped = refiner.wrap(cam, fake)
        z = camera_forward(cam) @ (wrapped.get_xyz[0] - cam.camera_center)
        loss = (z - target) ** 2
        if initial is None:
            initial = float(loss.detach())
        refiner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if refiner.delta.grad is not None and float(refiner.delta.grad.norm()) > 0:
            seen_grad = True
        refiner.step()
    wrapped = refiner.wrap(cam, fake)
    final = float(((camera_forward(cam) @ (wrapped.get_xyz[0] - cam.camera_center)
                    - target) ** 2).detach())
    check(seen_grad, "loss backpropagates into the translation table")
    check(final < initial, "Adam reduces the depth-like loss")
    check(refiner.delta_norms()["max"] <= 0.5 + 1e-5,
          "delta is clamped to --pose-max-shift")

    # clamp is enforced even for a huge step
    refiner2 = PoseRefiner([_make_camera(center, forward, "clamp")], lr=10.0,
                           max_shift=0.01)
    refiner2.delta.grad = torch.ones_like(refiner2.delta)
    refiner2.step()
    check(refiner2.delta_norms()["max"] <= 0.01 + 1e-6, "projection clamp")

    # ---- registration errors and sharing ---------------------------------
    stranger = _make_camera([0, 0, 0], [0, 0, 1], "stranger")
    checks = _expect_error(lambda: refiner.wrap(stranger, fake),
                           "unregistered camera", checks)

    cam_a = _make_camera(center, forward, "shared")
    cam_b = _make_camera(center + 0.1, forward, "shared")
    shared = PoseRefiner([cam_a, cam_b], lr=1e-3)
    check(shared.num_views == 1, "same image_name shares one translation")
    with torch.no_grad():
        shared.delta[0] = torch.tensor([0.01, 0.0, 0.0])
    check(torch.allclose(shared.wrap(cam_a, fake).get_xyz, xyz - shared.delta[0]) and
          torch.allclose(shared.wrap(cam_b, fake).get_xyz, xyz - shared.delta[0]),
          "shared translation applied to every resolution")

    print(f"self-check OK: {checks} checks "
          "(pose args, camera/world equivalence, wrapper, gradient step, clamp, sharing)")
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
