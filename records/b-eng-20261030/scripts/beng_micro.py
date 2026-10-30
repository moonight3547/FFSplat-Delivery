import argparse, json, os, random, sys, time, gc
from pathlib import Path
import numpy as np
import torch

REPO = Path("/root/projects/FFSplat-Hackathon")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "competition/upstream/baseline/fastgs"))
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from research.image_cache import CachedImages
from utils.camera_utils import loadCam
from utils.general_utils import PILtoTorch

OUT = Path("/root/gpufree-data/outputs/b-eng-20261030/micro"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
device = "cuda:2"
torch.cuda.set_device(torch.device(device))
white = False

def build_args(work, model_path, sh_degree=3):
    p = argparse.ArgumentParser(); lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p)
    argv = ["--source_path", str(work), "--model_path", model_path, "--sh_degree", str(sh_degree), "--eval", "-r", "1"]
    a = p.parse_args(argv)
    return lp.extract(a), op.extract(a), pp.extract(a)

def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0  # ms

def bench_scene(case):
    e = manifest[case]; work = REPO / e["work"]; out = {}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    ds, opt, pipe = build_args(work, "/tmp/beng_scene_model")
    os.makedirs(ds.model_path, exist_ok=True)
    g = GaussianModel(3, "default")
    t0 = time.perf_counter()
    scene = Scene(ds, g, resolution_scales=[1.0])
    torch.cuda.synchronize()
    scene_s = time.perf_counter() - t0
    t_cams = time.perf_counter()
    cams = scene.getTrainCameras()
    torch.cuda.synchronize()
    cams_s = time.perf_counter() - t_cams
    mem_scene = torch.cuda.memory_allocated() - m0
    H, W = cams[0].image_height, cams[0].image_width
    n = len(cams)
    bytes_f32 = n * 3 * H * W * 4
    bytes_u8 = n * 3 * H * W
    # camera tensors (view/proj/full_proj/center + inverses) fp32 per camera
    cam_bytes = n * 4 * 16 * 4
    # image conversion alone (mimic loadCam): cached uint8 -> PIL -> float32 cuda
    cache = CachedImages(str(work / "image_cache"))
    names = list(cache.index.keys())
    t0 = time.perf_counter()
    for name in names:
        arr = cache.get(name)
        from PIL import Image as PILImage
        img = PILImage.fromarray(arr)
        torch_img = PILtoTorch(img, (W, H))
        _ = torch_img[:3, ...].clamp(0.0, 1.0).to("cuda")
    torch.cuda.synchronize()
    conv_s = time.perf_counter() - t0
    # PLY parse + upload
    torch.cuda.empty_cache(); m1 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    g2 = GaussianModel(3, "default"); g2.load_ply(str(REPO / e["init"]))
    torch.cuda.synchronize()
    ply_s = time.perf_counter() - t0
    mem_ply = torch.cuda.memory_allocated() - m1
    n_gauss = int(g2.get_xyz.shape[0])
    del g2; torch.cuda.empty_cache()
    out.update(case=case, n_train_views=n, resolution=[H, W],
               scene_load_seconds=round(scene_s, 3), camera_list_seconds=round(cams_s, 4),
               image_conversion_all_views_seconds=round(conv_s, 3),
               scene_gpu_bytes=int(mem_scene), images_f32_bytes=int(bytes_f32),
               images_u8_bytes=int(bytes_u8), camera_tensors_bytes=int(cam_bytes),
               ply_load_seconds=round(ply_s, 3), ply_gpu_bytes=int(mem_ply),
               n_gaussians_init=n_gauss)
    return out, scene, g, pipe, ds

results = {}
# ---- per-scene composition for bicycle + one DL3DV bad case
scene_objs = {}
for case in ["mipnerf360/bicycle",
             "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"]:
    out, scene, g, pipe, ds = bench_scene(case)
    results[case] = out
    scene_objs[case] = (scene, g, pipe, ds)
    print(json.dumps(out), flush=True)

# ---- per-step no-op / overhead microbenches on bicycle
case = "mipnerf360/bicycle"
scene, g, pipe, ds = scene_objs[case]
g.load_ply(str(REPO / manifest[case]["init"]))
cam = scene.getTrainCameras()[0]
bench = {}
bench["gt_image_device"] = str(cam.original_image.device)
bench["gt_image_dtype"] = str(cam.original_image.dtype)
bench["gt_image_cuda_noop_ms"] = timeit(lambda: cam.original_image.cuda(), n=500)
cams = scene.getTrainCameras()
bench["list_copy_169_ms"] = timeit(lambda: cams.copy(), n=2000)
bench["random_pop_from_copy_ms"] = timeit(lambda: (lambda p: p.pop(random.randint(0, len(p) - 1)))(cams.copy()), n=2000)
bench["realloc_at_parse_ms"] = timeit(lambda: [int(x) for x in str("1250").split(",") if str(x).strip()], n=2000)
H, W = cam.image_height, cam.image_width
bench["metric_map_zeros_ms"] = timeit(lambda: torch.zeros(H * W, dtype=torch.int, device="cuda"), n=2000)
# render allocation: full render forward with default metric_map vs shared reusable buffer
background = torch.tensor([0., 0., 0.], device="cuda")
opt = OptimizationParams(argparse.ArgumentParser()).extract(argparse.Namespace(
    iterations=5000, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
    position_lr_delay_mult=0.01, position_lr_max_steps=5000, feature_lr=0.0025,
    shfeature_lr=0.005, opacity_lr=0.025, scaling_lr=0.01, rotation_lr=0.001,
    percent_dense=0.001, lambda_dssim=0.6, densification_interval=100,
    opacity_reset_interval=5001, densify_from_iter=500, densify_until_iter=1250,
    densify_grad_threshold=2e-4, loss_thresh=0.1, grad_abs_thresh=0.0012,
    highfeature_lr=0.005, lowfeature_lr=0.0025, grad_thresh=2e-4, dense=0.001,
    mult=0.5, random_background=False, optimizer_type="default"))
shared = torch.zeros(H * W, dtype=torch.int, device="cuda")
def render_once(mm=None):
    return render_fastgs(cam, g, pipe, background, 0.5, 1.0, metric_map=mm)
r1 = render_once(None)["render"]; torch.cuda.synchronize()
r2 = render_once(shared)["render"]; torch.cuda.synchronize()
r3 = render_once(None)["render"]; torch.cuda.synchronize()
bench["render_reuse_buffer_maxdiff"] = float((r2 - r1).abs().max())
bench["render_repeat_maxdiff"] = float((r3 - r1).abs().max())
# redundant per-camera ops in the current Camera path (upstream cameras.py:39-46)
Hc, Wc = cam.image_height, cam.image_width
_tmp_img = torch.rand(3, Hc, Wc, device="cuda")
def redundant_camera_ops():
    x = _tmp_img.clamp(0.0, 1.0)
    ones = torch.ones((1, Hc, Wc), device="cuda")
    x = x * ones
    return x
bench["camera_redundant_ops_per_view_ms"] = timeit(redundant_camera_ops, n=200)
bench["camera_redundant_ops_all_views_s"] = round(
    bench["camera_redundant_ops_per_view_ms"] * len(cams) / 1000.0, 4)
bench["render_forward_ms"] = timeit(lambda: render_once(None), n=50)
bench["render_forward_reuse_ms"] = timeit(lambda: render_once(shared), n=50)
results["microbench"] = bench
print(json.dumps({"microbench": bench}), flush=True)
json.dump(results, open(OUT / "microbench.json", "w"), indent=2)
print("BENG_MICRO_DONE", OUT / "microbench.json")
