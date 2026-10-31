import json, sys, time
from pathlib import Path
import numpy as np
import torch
REPO = Path("/root/projects/FFSplat-Hackathon")
sys.path.insert(0, str(REPO))
from lpips import LPIPS
from research.gt_feature_cache import GTCachedLPIPS

OUT = Path("/root/gpufree-data/outputs/framework-opt-20261030"); OUT.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(torch.device("cuda:2"))
base = LPIPS(net="vgg", verbose=False).cuda().eval()
for p in base.parameters(): p.requires_grad_(False)
cached = GTCachedLPIPS(base).cuda().eval()
for p in cached.parameters(): p.requires_grad_(False)

bundle = Path("/root/gpufree-data/outputs/b-eng-20261030/scene_cache/bicycle")
images = np.load(bundle / "images_u8.npy", mmap_mode="r")
res = {}
views = [0, 40, 120]
torch.manual_seed(0)
rows = []
peak0 = torch.cuda.max_memory_allocated()
for i, vi in enumerate(views):
    gt = torch.from_numpy(np.ascontiguousarray(images[vi]).astype(np.float32) / 255.0).cuda()
    render = (gt + torch.randn_like(gt) * 0.02).clamp(0, 1)
    in0, in1 = render.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
    live = base(in0, in1)
    miss = cached(in0, in1, key=vi)
    hit = cached(in0, in1, key=vi)
    rows.append(dict(view=int(vi), live=float(live.mean()), miss=float(miss.mean()), hit=float(hit.mean()),
                     maxdiff_miss=float((live - miss).abs().max()),
                     maxdiff_hit=float((live - hit).abs().max()),
                     reldiff=float(((live - miss).abs() / live.abs().clamp_min(1e-12)).max())))
peak1 = torch.cuda.max_memory_allocated()
res["identity"] = rows
res["cache_views"] = cached.cache_views()
res["cache_bytes"] = cached.cache_bytes()
res["bytes_per_view"] = cached.cache_bytes() / max(1, cached.cache_views())
res["build_seconds_total"] = cached.stats["build_seconds"]
res["peak_vram_mb"] = round(peak1 / 2**20, 1)
res["base_vram_mb"] = round(peak0 / 2**20, 1)
# timing: live forward vs cached miss/hit (CUDA events, 10 reps)
def bench(fn, n=10):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    e0.record()
    for _ in range(n): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n
gt = torch.from_numpy(np.ascontiguousarray(images[0]).astype(np.float32) / 255.0).cuda()
render = (gt + torch.randn_like(gt) * 0.02).clamp(0, 1)
in0, in1 = render.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
cache_key = 999999  # pre-warm so timing measures hits only when we want
_ = cached(in0, in1, key=cache_key)
res["timing_ms"] = dict(
    live_forward=bench(lambda: base(in0, in1)),
    cached_hit=bench(lambda: cached(in0, in1, key=cache_key)),
    cached_miss=bench(lambda: cached(in0, in1, key=-1)),
)
res["vgg16_sha256"] = None
import inspect, hashlib
lp = Path(inspect.getfile(LPIPS))
wp = lp.parent / "weights/v0.1/vgg.pth"
if wp.exists():
    res["vgg16_sha256"] = hashlib.sha256(wp.read_bytes()).hexdigest()
res["lpips_pkg_sha256"] = hashlib.sha256(lp.read_bytes()).hexdigest()
res["lpips_source"] = str(lp)
json.dump(res, open(OUT / "gtcache_identity.json", "w"), indent=2)
print(json.dumps(res, indent=1)[:2600])
