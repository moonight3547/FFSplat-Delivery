"""Collect every research sweep summary into one comparable table.

Reads ``**/summary.json`` (module-B sweeps) and ``**/score_summary.json``
(initialization-only scores) from one or more roots and prints the rows sorted
by proxy score. Nothing is recomputed: the numbers come from the official
``baseline/evaluate.py`` that each sweep already ran.

  python -m research.collect records/artifacts-20261030 outputs/experiments
"""
import argparse
import json
from pathlib import Path


def rows_from(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    if "results" in data and "case" in data:
        for row in data["results"]:
            out.append(dict(kind="B-sweep", case=data["case"], name=path.parent.parent.name,
                            iteration=row["iteration"], gaussians=row["num_gaussians"],
                            b_seconds=row["b_seconds_estimate"], psnr=row["psnr"],
                            ssim=row["ssim"], lpips=row["lpips"],
                            quality=row["score"]["quality_points"],
                            time=row["score"]["time_points"],
                            total=row["score"]["total_points"]))
    elif "rows" in data and "work" in data:
        for row in data["rows"]:
            out.append(dict(kind="init", case=Path(data["work"]).name, name=row["name"],
                            iteration=0, gaussians=None, b_seconds=data.get("b_seconds"),
                            psnr=row["psnr"], ssim=row["ssim"], lpips=row["lpips"],
                            quality=row["score"]["quality_points"],
                            time=row["score"].get("time_points"),
                            total=row["score"].get("quality_points")))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--case", default=None, help="filter by case substring")
    parser.add_argument("--kind", default=None, choices=["B-sweep", "init"])
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    rows = []
    for root in args.roots:
        for path in sorted(Path(root).rglob("summary.json")):
            try:
                rows.extend(rows_from(path))
            except (KeyError, ValueError):
                continue
        for path in sorted(Path(root).rglob("score_summary.json")):
            try:
                rows.extend(rows_from(path))
            except (KeyError, ValueError):
                continue

    if args.case:
        rows = [r for r in rows if args.case in (r["case"] or "")]
    if args.kind:
        rows = [r for r in rows if r["kind"] == args.kind]
    rows.sort(key=lambda r: (-(r["total"] or r["quality"] or 0)))

    header = f"{'kind':8s} {'case':22s} {'variant':22s} {'it':>6s} {'N':>8s} {'B(s)':>7s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'q':>6s} {'t':>6s} {'tot':>6s}"
    print(header)
    for r in rows:
        print(f"{r['kind']:8s} {str(r['case'])[:22]:22s} {str(r['name'])[:22]:22s} "
              f"{r['iteration'] or 0:6d} {(r['gaussians'] or 0):8d} {(r['b_seconds'] or 0):7.1f} "
              f"{r['psnr']:7.3f} {r['ssim']:7.4f} {r['lpips']:7.4f} "
              f"{(r['quality'] or 0):6.2f} {(r['time'] or 0):6.2f} {(r['total'] or 0):6.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
