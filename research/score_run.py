"""Score a completed organizer run directory with the proxy /100 formula.

Reads ``<work>/results/<case>/{resplat,module_b,reference_fastgs}`` and
``<work>/results/<case>/module_b/metrics.json`` and prints per-case and mean
proxy scores. TIME uses module-B seconds over the case's own measured
random-init FastGS 30K reference, as fixed in ProblemDesc.md 4.

  python -m research.score_run --work outputs/O0-20261030
"""
import argparse
import json
import statistics
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "benchmarks/organizer-examples3-v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points(psnr, ssim, lpips):
    return dict(psnr=20.0 * max(0.0, min(1.0, (psnr - 15.0) / 25.0)),
                ssim=20.0 * max(0.0, min(1.0, (ssim - 0.70) / 0.25)),
                lpips=40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20)))


def time_points(b_seconds, budget):
    if b_seconds >= budget:
        return 0.0
    return 20.0 * min(1.0, (1.0 - b_seconds / budget) / 0.9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--json", default=None, help="write the table to this JSON path")
    args = parser.parse_args(argv)
    spec = read(args.spec)
    root = Path(args.work)
    rows = []
    for case in spec["cases"]:
        folder = root / "results" / case["id"]
        metrics = read(folder / "module_b/metrics.json")["mean"]
        b = read(folder / "module_b/module_b.json")
        ref = read(folder / "reference_fastgs/module_b.json")
        resplat = read(folder / "resplat/resplat.json")
        module_a = read(root / "work" / case["id"] / "module_a.json")
        pts = points(metrics["psnr"], metrics["ssim"], metrics["lpips"])
        quality = sum(pts.values())
        budget = ref["total_seconds"]
        tp = time_points(b["total_seconds"], budget)
        rows.append(dict(
            case=case["id"], psnr=metrics["psnr"], ssim=metrics["ssim"], lpips=metrics["lpips"],
            points=pts, quality_points=quality,
            b_seconds=b["total_seconds"], budget_seconds=budget,
            ratio=b["total_seconds"] / budget, time_points=tp,
            total_points=quality + tp, valid=b["total_seconds"] < budget,
            iterations=b.get("iterations"), gaussians=b.get("num_final_gaussians"),
            a_seconds=module_a["total_seconds"], resplat_seconds=resplat["total_seconds"],
            e2e_seconds=module_a["total_seconds"] + resplat["total_seconds"] + b["total_seconds"]))
    mean = {k: statistics.mean(r[k] for r in rows)
            for k in ("psnr", "ssim", "lpips", "quality_points", "time_points", "total_points")}
    mean.update(mean_b_seconds=statistics.mean(r["b_seconds"] for r in rows),
                mean_e2e_seconds=statistics.mean(r["e2e_seconds"] for r in rows),
                all_valid=all(r["valid"] for r in rows))
    out = dict(work=str(root), rows=rows, mean=mean)
    for r in rows:
        print(f"{r['case']:<40} PSNR {r['psnr']:7.3f} SSIM {r['ssim']:.4f} LPIPS {r['lpips']:.4f} "
              f"| q {r['quality_points']:5.2f} + t {r['time_points']:5.2f} = {r['total_points']:5.2f} "
              f"| B {r['b_seconds']:6.2f}/{r['budget_seconds']:6.2f}s it {r['iterations']} N {r['gaussians']}")
    print(f"{'MEAN':<40} PSNR {mean['psnr']:7.3f} SSIM {mean['ssim']:.4f} LPIPS {mean['lpips']:.4f} "
          f"| q {mean['quality_points']:5.2f} + t {mean['time_points']:5.2f} = {mean['total_points']:5.2f} "
          f"| B {mean['mean_b_seconds']:6.2f}s E2E {mean['mean_e2e_seconds']:6.2f}s valid {mean['all_valid']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
