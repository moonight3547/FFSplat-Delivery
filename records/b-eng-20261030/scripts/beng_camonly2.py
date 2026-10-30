import argparse, gc, json, os, sys, time
from pathlib import Path
import torch
REPO = Path("/root/projects/FFSplat-Hackathon")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "competition/upstream/baseline/fastgs"))
from scene import Scene, GaussianModel
import scene.dataset_readers as readers
from arguments import ModelParams, OptimizationParams, PipelineParams
from research.image_cache import CachedImages
OUT = Path("/root/gpufree-data/outputs/b-eng-20261030/camonly"); OUT.mkdir(parents=True, exist_ok=True)
manifest = {e["case"]: e for e in json.load(open(REPO / "records/evalnode-manifest-13.json"))}
torch.cuda.set_device(torch.device("cuda:2"))
os.makedirs("/tmp/beng_camonly_model", exist_ok=True)
saved_callbacks = dict(readers.sceneLoadTypeCallbacks)

def build_args(work):
    p = argparse.ArgumentParser(); lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p)
    a = p.parse_args(["--source_path", str(work), "--model_path", "/tmp/beng_camonly_model",
                      "--sh_degree", "3", "--eval", "-r", "1"])
    return lp.extract(a), op.extract(a), pp.extract(a)

def install_image_cache(work):
    cache = CachedImages(str(work / "image_cache"))
    from PIL import Image as PILImage
    orig_open = readers.Image.open
    def cached_open(path, *a, **k):
        name = os.path.basename(str(path))
        if name in cache.index:
            return PILImage.fromarray(cache.get(name))
        return orig_open(path, *a, **k)
    readers.Image.open = cached_open

def set_train_only(flag):
    readers.sceneLoadTypeCallbacks.clear()
    readers.sceneLoadTypeCallbacks.update(saved_callbacks)
    if flag:
        def make(orig):
            def wrap(*a, **k):
                return orig(*a, **k)._replace(test_cameras=[])
            return wrap
        for k in ("Colmap", "Blender"):
            readers.sceneLoadTypeCallbacks[k] = make(saved_callbacks[k])

def load_scene(work, train_only):
    set_train_only(train_only)
    ds, opt, pipe = build_args(work)
    g = GaussianModel(3, "default")
    t0 = time.perf_counter()
    scene = Scene(ds, g, resolution_scales=[1.0])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = dict(seconds=dt, n_test=len(scene.getTestCameras()))
    del scene, g
    gc.collect(); torch.cuda.empty_cache()
    return out

results = {}
for case in ["mipnerf360/bicycle",
             "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce"]:
    work = REPO / manifest[case]["work"]
    install_image_cache(work)
    tag = "bicycle" if "bicycle" in case else "0b5635"
    reps = []
    for i in range(4):
        # alternate order to cancel page-cache / allocator ordering bias;
        # also one same-code drift control (full then full)
        if i % 2 == 0:
            full_a = load_scene(work, False); to = load_scene(work, True); full_b = load_scene(work, False)
        else:
            to = load_scene(work, True); full_a = load_scene(work, False); full_b = load_scene(work, False)
        reps.append(dict(full_a=full_a["seconds"], train_only=to["seconds"],
                         full_b=full_b["seconds"], n_test=to["n_test"]))
    savings = [r["full_a"] - r["train_only"] for r in reps]
    drift = [r["full_b"] - r["full_a"] for r in reps]
    results[tag] = dict(reps=reps, savings=savings,
                        mean_saving=round(sum(savings)/len(savings), 3),
                        drift=drift, mean_drift=round(sum(drift)/len(drift), 3))
    print(json.dumps({tag: results[tag]}), flush=True)
json.dump(results, open(OUT / "camonly_isolated2.json", "w"), indent=2)
print("CAMONLY2_DONE")
