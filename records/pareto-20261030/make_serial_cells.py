#!/usr/bin/env python3
"""Build the serial-anchor cell list from the concurrent Pareto results.

  python3 make_serial_cells.py --results results.jsonl --out serial_cells.txt \
      [--add-steps 500,2000] [--min-per-scene 2]

Selection: for every scene take the argmax-total S, plus S=5000 (delivery), plus
any S listed in --add-steps (used to get 2 points/side for a B ~ a + r*S fit).
Cells are de-duplicated and written as "case S" lines, sorted by case then S.
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--add-steps", default="")
    p.add_argument("--require", default="5000",
                   help="S values always included (comma separated)")
    p.add_argument("--exclude-cases", default="")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.results, encoding="utf-8")
            if line.strip()]
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    must = {int(x) for x in args.require.split(",") if x.strip()}
    extra = {int(x) for x in args.add_steps.split(",") if x.strip()}
    excluded = [c.strip() for c in args.exclude_cases.split(",") if c.strip()]

    cells = {}
    for case, rs in by_case.items():
        if any(x in case for x in excluded):
            continue
        chosen = set(must) | extra
        best = max(rs, key=lambda r: r["total_points"])
        chosen.add(best["S"])
        available = {r["S"] for r in rs}
        for S in sorted(chosen):
            if S in available:
                cells[(case, S)] = best["total_points"] if S == best["S"] else None
    lines = ["%s %d" % (case, S) for (case, S) in sorted(cells)]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    argmax = {}
    for case, rs in by_case.items():
        argmax[case] = max(rs, key=lambda r: r["total_points"])["S"]
    print("cells:", len(lines))
    for case in sorted(argmax):
        print("  %-46s argmax S=%d" % (case, argmax[case]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
