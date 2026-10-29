"""Summarize baseline run: metrics + per-stage timing (any env, CPU only).

Reads results/<group>/<scene>/{module_a.json(resplat.json,module_b.json,
module_b/metrics.json)} and prints/writes summary.{json,md}.

Usage: python summarize.py --results results [--out results]
"""

import argparse
import json
import os


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or args.results

    scenes = []
    for group in sorted(os.listdir(args.results)):
        gdir = os.path.join(args.results, group)
        if not os.path.isdir(gdir):
            continue
        for scene in sorted(os.listdir(gdir)):
            sdir = os.path.join(gdir, scene)
            if not os.path.isdir(sdir):
                continue
            a = load_json(os.path.join(args.work, group, scene, "module_a.json")) or {}
            r = load_json(os.path.join(sdir, "resplat", "resplat.json")) or {}
            b = load_json(os.path.join(sdir, "module_b", "module_b.json")) or {}
            m = load_json(os.path.join(sdir, "module_b", "metrics.json")) or {}
            ref = load_json(os.path.join(sdir, "reference_3dgs", "module_b.json")) or {}
            mean = m.get("mean", {})
            t_a = a.get("total_seconds")
            t_r = r.get("total_seconds")
            t_b = b.get("total_seconds")
            total = (t_a or 0) + (t_r or 0) + (t_b or 0)
            scenes.append({
                "group": group, "scene": scene,
                "pose_source": a.get("pose_source"),
                "resolution": a.get("resolution"),
                "images": a.get("num_images"),
                "gaussians_init": b.get("num_init_gaussians"),
                "gaussians_final": b.get("num_final_gaussians"),
                "time_A": t_a, "time_resplat": t_r, "time_B": t_b,
                "time_total": round(total, 1) if total else None,
                "psnr": mean.get("psnr"), "ssim": mean.get("ssim"),
                "lpips": mean.get("lpips"),
                "reference_3dgs_time": ref.get("total_seconds"),
                "reference_3dgs_psnr": ref.get("test_psnr_quick"),
            })

    lines = ["| group | scene | PSNR | SSIM | LPIPS | A(s) | resplat(s) | B(s) | total(s) | ref30k(s) | ref30k PSNR |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in scenes:
        fmt = lambda v: round(v, 1) if isinstance(v, (int, float)) else v
        lines.append(
            f"| {s['group']} | {s['scene']} | {s['psnr']} | {s['ssim']} | "
            f"{s['lpips']} | {fmt(s['time_A'])} | {fmt(s['time_resplat'])} | {fmt(s['time_B'])} | "
            f"{fmt(s['time_total'])} | {fmt(s['reference_3dgs_time'])} | "
            f"{s['reference_3dgs_psnr']} |")
    md = "\n".join(lines)

    with open(os.path.join(out_path, "summary.json"), "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    with open(os.path.join(out_path, "summary.md"), "w") as f:
        f.write("# ReSplat competition baseline — results\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
