"""Run in each named Conda environment; fail instead of silently dropping SSIM."""
import argparse
import importlib
import json
import platform
import inspect
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["resplat", "fastgs", "mcmc"], required=True)
    args = parser.parse_args()
    modules = {
        "resplat": ["torch", "gsplat", "pointops", "hydra", "lpips", "skimage", "e3nn"],
        "fastgs": ["torch", "diff_gaussian_rasterization_fastgs", "diff_gaussian_rasterization", "simple_knn._C", "fused_ssim"],
        "mcmc": ["torch", "gsplat", "plyfile", "scipy", "PIL", "competition.train_utils"],
    }
    result = dict(runtime=args.runtime, platform=platform.platform(), modules={})
    ok = True
    for name in modules[args.runtime]:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = dict(path=getattr(module, "__file__", None), version=getattr(module, "__version__", None))
        except Exception as error:
            ok = False
            result["modules"][name] = dict(error=str(error))
    try:
        import torch
        result["cuda"] = torch.cuda.is_available()
        ok = ok and result["cuda"]
        if result["cuda"]:
            result["gpu"] = torch.cuda.get_device_name()
            result["gpu_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        ok = False
    if args.runtime in ("resplat", "mcmc") and result["modules"].get("gsplat", {}).get("version") != "1.5.3":
        ok = False
    if args.runtime == "resplat":
        result["colmap"] = shutil.which("colmap")
        ok = ok and result["colmap"] is not None
    if args.runtime == "fastgs" and "error" not in result["modules"]["diff_gaussian_rasterization_fastgs"]:
        from .environment import SETTINGS_FIELDS
        raster = importlib.import_module("diff_gaussian_rasterization_fastgs")
        result["fastgs_interface"] = SETTINGS_FIELDS <= set(getattr(raster.GaussianRasterizationSettings, "_fields", ())) and {"means2D", "dc", "shs"} <= set(inspect.signature(raster.GaussianRasterizer.forward).parameters)
        ok = ok and result["fastgs_interface"]
    result["acceptance_scope"] = "Import/API/CUDA availability only; render/backward and metric acceptance still required"
    result["ready"] = ok
    print(json.dumps(result, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
