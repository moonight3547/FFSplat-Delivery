#!/usr/bin/env python3
"""Steps-Pareto analyzer for the FFSplat delivery recipe.

Reads one or more sweep outputs and produces the evidence tables for
``docs/PARETO-STEPS-20261030.md``:

1. **per-scene Pareto table** - every (S, quality, B, TIME, total) point, the
   per-scene argmax total, and the quality-vs-B frontier (non-dominated points);
2. **policy comparison across the 13 scenes** -
   (a) fixed ``S=5000`` (current delivery),
   (b) best single *uniform* S,
   (c) per-scene oracle (DIAGNOSTIC ceiling only - ``AGENTS.md`` forbids
       per-scene hyper-parameter selection in the submitted pipeline),
   (d) the grid marginal-value stop rule: stop when the next interval's
       quality gain per second of module-B time falls below the TIME marginal
       value ``lambda = 20 / (0.9 * budget)`` (0.223-0.240 points/s);
3. **variance / robustness diagnostics** - which decisions move less than the
   measured run-to-run noise, which B timings look concurrency-inflated, and
   where serial anchors are needed.

Scoring is the authoritative organizer proxy (``ProblemDesc.md`` 4)::

    quality = 20*clip((PSNR-15)/25, 0, 1)
            + 20*clip((SSIM-0.70)/0.25, 0, 1)
            + 40*clip((0.25-LPIPS)/0.20, 0, 1)
    TIME    = 20*min(1, (1 - B/budget)/0.9)      [0 if B >= budget -> INVALID]
    total   = quality + TIME

Accepted input schemas (all read-only; nothing is inferred that the file does
not contain):

* ``records/pareto-20261030/results.jsonl`` (written by
  ``records/pareto-20261030/run_sweep_pareto.py``): one JSON object per
  (case, S) with ``case, S, b_seconds_estimate, psnr, ssim, lpips,
  quality_points, time_points, total_points, valid, budget_seconds,
  num_gaussians, num_train_views, lane, ...``.
* ``outputs/ocurve/<tag>/s<S>/summary.json`` (H1): ``{case, budget_seconds,
  train_record, results: [<official metric point>]}``.
* ``outputs/scene13r15/results.jsonl`` (H1 13-case paired table):
  ``{case, baseline: {...}, delivery: {...}}``; needs ``--reference`` because
  its rows do not carry S. Used only as the *measured serial anchor* for the
  S=5000 delivery arm and the baseline O0 row - not as Pareto points.
* a flat JSONL of row dicts ``{case, S|steps|iteration, b_seconds|b|...,
  psnr, ssim, lpips}`` - for hand-assembled data.
* an optional serial-timing anchor file (``--serial-anchors``): rows
  ``{case, S, b_seconds}`` measured single-card, used to correct the
  concurrency-inflated B seconds via a per-scene linear B(S) fit.

Usage::

    python records/pareto-20261030/analyze_pareto.py \
        --results records/pareto-20261030/results.jsonl \
        --reference .analyst-tmp/h1/scene13r15/results.jsonl \
        --out-json records/pareto-20261030/analysis.json \
        --out-md  records/pareto-20261030/analysis-tables.md

    # Phase-1 smoke test on the existing ``ocurve`` tree (3 hard DL3DV scenes):
    python records/pareto-20261030/analyze_pareto.py --results .analyst-tmp/h1/ocurve

Exit code is 0 on success, 2 when no usable row is found. Warnings (missing
cells, recomputed-vs-stored score mismatches, partial coverage) are written to
stderr and to ``report["warnings"]``; the analyzer never silently drops a scene.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------- constants
#: Authoritative budgets = measured same-GPU FastGS random-init 30K seconds
#: (AGENTS.md "用户确认的预算"). Full scene key -> seconds.
BUDGETS: dict[str, float] = {
    "dl3dv": 94.29,
    "hw/billboard": 99.49,
    "hw/basketball": 95.0,
    "hw/car": 95.0,
    "hw/trashbin": 95.0,
    "mipnerf360/bicycle": 95.0,
    "mipnerf360/bonsai": 95.0,
    "mipnerf360/counter": 92.57,
}

#: The sweep grid (task-2). The analyzer keeps any S it finds, but policy tables
#: are restricted to this grid by default so every S is compared on one scale.
DEFAULT_GRID: tuple[int, ...] = (500, 1000, 2000, 3000, 5000, 8000)

#: Steps of the two arms in the H1 13-case paired table (``scene_eval_r15.sh``).
REFERENCE_ARMS: dict[str, int] = {"baseline": 10000, "delivery": 5000}

TIME_DENOMINATOR = 0.9  # TIME = 20 * min(1, (1 - B/budget) / 0.9)


# ---------------------------------------------------------------- scoring
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def quality_points(psnr: float, ssim: float, lpips: float) -> dict[str, float]:
    """Authoritative quality proxy /80."""
    psnr_pts = 20.0 * clip01((psnr - 15.0) / 25.0)
    ssim_pts = 20.0 * clip01((ssim - 0.70) / 0.25)
    lpips_pts = 40.0 * clip01((0.25 - lpips) / 0.20)
    return {
        "psnr": psnr_pts,
        "ssim": ssim_pts,
        "lpips": lpips_pts,
        "quality": psnr_pts + ssim_pts + lpips_pts,
    }


def time_points(b_seconds: float | None, budget: float | None) -> float | None:
    """Authoritative TIME proxy /20 for module-B seconds ``b_seconds``."""
    if b_seconds is None or budget is None:
        return None
    if b_seconds >= budget:
        return 0.0
    return 20.0 * min(1.0, (1.0 - b_seconds / budget) / TIME_DENOMINATOR)


def marginal_value(budget: float) -> float:
    """TIME points lost per extra second of module-B time (points/s).

    ``d TIME / d B = -20 / (0.9 * budget)`` while ``B < budget`` (and while the
    ``min(1, .)`` cap is inactive, i.e. ``B >= 0.1 * budget``). 0.2234 / 0.2356
    / 0.2339 / 0.2400 for billboard / dl3dv / 95 s scenes / counter.
    """
    return 20.0 / (TIME_DENOMINATOR * budget)


def budget_for(case: str) -> float | None:
    if case in BUDGETS:
        return BUDGETS[case]
    group, _, name = case.partition("/")
    key = f"{group}/{name}"
    if key in BUDGETS:
        return BUDGETS[key]
    if group == "dl3dv":
        return BUDGETS["dl3dv"]
    if group == "hw":
        return BUDGETS["hw/billboard"] if name == "billboard" else BUDGETS["hw/car"]
    if group == "mipnerf360":
        return BUDGETS["mipnerf360/counter"] if name == "counter" else BUDGETS["mipnerf360/bicycle"]
    return None


# ---------------------------------------------------------------- data model
@dataclass
class Row:
    """One (case, S) measurement, with the proxy score recomputed from metrics."""

    case: str
    steps: int
    b_seconds: float | None
    psnr: float | None
    ssim: float | None
    lpips: float | None
    source: str = ""
    stored: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.budget = budget_for(self.case)
        if None in (self.psnr, self.ssim, self.lpips):
            self.points = None
            self.quality = self.time = self.total = None
            self.valid = False
            return
        self.points = quality_points(float(self.psnr), float(self.ssim), float(self.lpips))
        self.quality = self.points["quality"]
        self.time = time_points(self.b_seconds, self.budget)
        # An over-budget row is INVALID: AGENTS.md excludes it from every mean.
        self.valid = self.time is not None and self.b_seconds is not None and self.budget is not None \
            and self.b_seconds < self.budget
        self.total = None if self.time is None else self.quality + self.time

    def recompute(self) -> None:
        """Recompute points/time/valid/total after mutating budget or b_seconds."""
        if None in (self.psnr, self.ssim, self.lpips):
            self.points = None
            self.quality = self.time = self.total = None
            self.valid = False
            return
        self.points = quality_points(float(self.psnr), float(self.ssim), float(self.lpips))
        self.quality = self.points["quality"]
        self.time = time_points(self.b_seconds, self.budget)
        self.valid = self.time is not None and self.b_seconds is not None and self.budget is not None \
            and self.b_seconds < self.budget
        self.total = None if self.time is None else self.quality + self.time

    @property
    def lam(self) -> float | None:
        return marginal_value(self.budget) if self.budget else None


def _pick(d: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _row_from_flat(case: str, steps: int, rec: dict[str, Any], source: str, *,
                   stored: dict[str, Any] | None = None,
                   extra: dict[str, Any] | None = None) -> Row:
    b = _pick(rec, ("b_seconds_estimate", "b_seconds", "b", "total_seconds"))
    psnr = _pick(rec, ("psnr", "PSNR"))
    ssim = _pick(rec, ("ssim", "SSIM"))
    lpips = _pick(rec, ("lpips", "LPIPS", "vgg_lpips"))
    return Row(case=case, steps=int(steps), b_seconds=None if b is None else float(b),
               psnr=None if psnr is None else float(psnr),
               ssim=None if ssim is None else float(ssim),
               lpips=None if lpips is None else float(lpips),
               source=source, stored=dict(stored or {}), extra=dict(extra or {}))


def _rows_from_point(case: str, point: dict[str, Any], source: str,
                     steps_override: int | None = None) -> list[Row]:
    """Expand one ``summary.json`` point (``results[i]`` or paired arm)."""
    steps = steps_override if steps_override is not None else \
        _pick(point, ("S", "steps", "iteration"))
    if steps is None:
        return []
    extra: dict[str, Any] = {}
    for key in ("num_gaussians", "num_train_views", "num_init_gaussians",
                "budget_seconds", "lane", "wall_seconds", "cached"):
        if point.get(key) is not None:
            extra[key] = point[key]
    stored = point.get("score") if isinstance(point.get("score"), dict) else {
        "quality_points": point.get("quality_points"),
        "time_points": point.get("time_points"),
        "total_points": point.get("total_points"),
        "valid": point.get("valid"),
    }
    return [_row_from_flat(case, steps, point, source, stored=stored, extra=extra)]


def _load_jsonl_rows(path: Path, warnings: list[str]) -> tuple[list[Row], list[dict[str, Any]]]:
    rows: list[Row] = []
    paired: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path}:{lineno}: unparsable JSON line ({exc}); skipped")
            continue
        if not isinstance(rec, dict):
            continue
        case = rec.get("case")
        # (1) sweep results.jsonl / flat row: must carry case + S + metrics
        if case and _pick(rec, ("S", "steps", "iteration")) is not None:
            rows.extend(_rows_from_point(str(case), rec, path.name))
            continue
        # (2) H1 paired 13-case table: {case, baseline: {...}, delivery: {...}}
        arms = [k for k in REFERENCE_ARMS if isinstance(rec.get(k), dict)]
        if case and arms:
            paired.append(rec)
            for arm in arms:
                for row in _rows_from_point(str(case), rec[arm], path.name,
                                            steps_override=REFERENCE_ARMS[arm]):
                    row.extra["arm"] = arm
                    row.extra["paired_table"] = True
                    rows.append(row)
            continue
        # (3) summary.json-shaped record used as a JSONL line
        results = rec.get("results")
        if case and isinstance(results, list):
            for point in results:
                if isinstance(point, dict):
                    rows.extend(_rows_from_point(str(case), point, path.name))
            continue
        warnings.append(f"{path}:{lineno}: no case/S/metrics in JSON object; skipped")
    return rows, paired


def _load_recursive(path: Path, warnings: list[str]) -> tuple[list[Row], list[dict[str, Any]]]:
    rows: list[Row] = []
    paired: list[dict[str, Any]] = []
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
        if not files:
            warnings.append(f"{path}: directory contains no .json/.jsonl; nothing loaded")
        for file in files:
            sub_rows, sub_paired = load_any(file, warnings)
            rows.extend(sub_rows)
            paired.extend(sub_paired)
        return rows, paired
    if not path.exists():
        warnings.append(f"{path}: does not exist; skipped")
        return rows, paired
    if path.suffix == ".jsonl":
        return _load_jsonl_rows(path, warnings)
    # summary.json / report.json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: unparsable JSON ({exc}); skipped")
        return rows, paired
    records = data if isinstance(data, list) else [data]
    for rec in records:
        if not isinstance(rec, dict):
            continue
        case = rec.get("case") or (rec.get("identity") or {}).get("case")
        results = rec.get("results")
        if case and isinstance(results, list):
            for point in results:
                if isinstance(point, dict):
                    rows.extend(_rows_from_point(str(case), point, path.name))
        elif case and _pick(rec, ("S", "steps", "iteration")) is not None:
            rows.extend(_rows_from_point(str(case), rec, path.name))
        elif case and any(isinstance(rec.get(k), dict) for k in REFERENCE_ARMS):
            paired.append(rec)
            for arm in REFERENCE_ARMS:
                if isinstance(rec.get(arm), dict):
                    for row in _rows_from_point(str(case), rec[arm], path.name,
                                                steps_override=REFERENCE_ARMS[arm]):
                        row.extra.setdefault("arm", arm)
                        rows.append(row)
        elif results is not None and not case:
            warnings.append(f"{path}: summary without case; skipped")
    return rows, paired


def load_any(path: Path, warnings: list[str]) -> tuple[list[Row], list[dict[str, Any]]]:
    rows, paired = _load_recursive(path, warnings)
    return rows, paired


def node_budget_factors(grouped: dict[str, list[Row]],
                        reference_delivery_b: dict[str, float],
                        anchors: dict[str, dict[int, float]] | None = None,
                        anchor: int = 5000) -> tuple[dict[str, float], str]:
    """Correct the budget for the timing node.

    The self-measured budgets (94.29 / 99.49 / ...) were taken on H1; the E2 sweep
    timings come from N1. ``TIME = 20*min(1,(1-B/budget)/0.9)`` is only meaningful
    when B and budget are from the same machine class, so scale each scene's budget
    by the measured node-speed ratio::

        factor_case = B_serial(case, anchor) / B_H1_delivery(case, anchor)

    ``B_serial`` comes from the serial anchors when available (best), else from the
    sweep's anchor cell (concurrency-inflated, flagged).  Scenes without a ratio use
    the median factor.  Returns ``{case: factor}`` and a source label.
    """
    factors: dict[str, float] = {}
    for case, ref_b in reference_delivery_b.items():
        if not ref_b:
            continue
        candidates = []
        if anchors and anchors.get(case, {}).get(anchor):
            candidates.append(("serial", anchors[case][anchor] / ref_b))
        sweep_b = next((p.b_seconds for p in grouped.get(case, [])
                        if p.steps == anchor and p.b_seconds), None)
        if sweep_b:
            candidates.append(("sweep", sweep_b / ref_b))
        if candidates:
            # serial wins over the concurrency-inflated sweep cell
            factors[case] = round(candidates[0][1], 4)
    if not factors:
        return {}, "none"
    median = round(statistics.median(factors.values()), 4)
    for case in grouped:
        factors.setdefault(case, median)
    source = "serial-anchors" if (anchors and any(
        anchors.get(c, {}).get(anchor) for c in factors)) else "sweep-anchor-cell"
    return factors, source


def rescale_budgets(rows: Iterable[Row], factors: dict[str, float]) -> list[Row]:
    """Clone ``Row``s with ``budget * factor`` and recompute the proxy scores."""
    out: list[Row] = []
    for row in rows:
        factor = factors.get(row.case)
        if factor is None:
            out.append(row)
            continue
        clone = Row(case=row.case, steps=row.steps,
                    b_seconds=row.b_seconds, psnr=row.psnr, ssim=row.ssim, lpips=row.lpips,
                    source=row.source, stored=dict(row.stored), extra=dict(row.extra))
        if clone.budget is not None:
            clone.budget = round(clone.budget * factor, 4)
        clone.extra["budget_factor"] = factor
        clone.recompute()
        out.append(clone)
    return out


def load_serial_anchors(path: Path | None, warnings: list[str]) -> dict[str, dict[int, float]]:
    """Read ``{case, S, b_seconds}`` JSONL anchors (single-card timings, same node)."""
    anchors: dict[str, dict[int, float]] = {}
    if not path:
        return anchors
    if not path.exists():
        warnings.append(f"serial anchors file {path} does not exist; ignored")
        return anchors
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path}:{lineno}: unparsable anchor line ({exc}); skipped")
            continue
        case = rec.get("case")
        steps = _pick(rec, ("S", "steps", "iteration"))
        b = _pick(rec, ("b_seconds", "b_serial", "b_seconds_serial", "b_seconds_estimate"))
        if case and steps is not None and b is not None:
            anchors.setdefault(str(case), {})[int(steps)] = float(b)
        else:
            warnings.append(f"{path}:{lineno}: anchor needs case/S/b_seconds; skipped")
    return anchors


# ---------------------------------------------------------------- analysis
def group_by_case(rows: Iterable[Row]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.case, []).append(row)
    for points in grouped.values():
        points.sort(key=lambda r: r.steps)
    return grouped


def dedupe(points: list[Row]) -> list[Row]:
    """Last row wins for a duplicated (case, S); warn handled by the caller."""
    out: dict[int, Row] = {}
    for row in points:
        out[row.steps] = row
    return [out[s] for s in sorted(out)]


def frontier(points: list[Row]) -> list[Row]:
    """Non-dominated (B, quality) points: minimise B, maximise quality."""
    pts = [p for p in points if p.valid and p.quality is not None and p.b_seconds is not None]
    pts.sort(key=lambda r: (r.b_seconds, -r.quality))
    keep: list[Row] = []
    best_quality = float("-inf")
    for point in pts:
        if point.quality > best_quality:
            keep.append(point)
            best_quality = point.quality
    return keep


def argmax_total(points: list[Row]) -> Row | None:
    valid = [p for p in points if p.valid and p.total is not None]
    return max(valid, key=lambda r: r.total) if valid else None


def step_time_rate(points: list[Row], anchor: int = 5000) -> dict[str, Any] | None:
    """Fit B = a + r*S over the measured cells and express r per 1000 steps.

    ``r`` is the marginal module-B seconds per step; ``lambda * r`` is the quality
    gain per step needed for one more step to pay, i.e. the break-even slope.
    ``r * 1000 * lambda`` is the TIME cost of 1000 steps (== break-even quality
    per 1000 steps).  Uses raw (un-corrected) B; a linear fit over 6 grid points is
    insensitive to a uniform concurrency inflation, but the intercept absorbs the
    fixed load/init overhead.
    """
    pts = [(p.steps, p.b_seconds) for p in points if p.b_seconds is not None]
    if len(pts) < 2:
        return None
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    n = len(xs)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    r = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    lam = points[0].lam
    return dict(r_s_per_step=round(r, 8), r_s_per_1000=round(r * 1000, 3),
                intercept=round(mean_y - r * mean_x, 3), n=n,
                breakeven_q_per_1000=(None if lam is None else round(lam * r * 1000, 3)),
                b_at_anchor=next((p.b_seconds for p in points if p.steps == anchor), None))


def repeat_noise(primary: dict[str, list[Row]], replicate: dict[str, list[Row]]) -> dict[str, Any]:
    """Run-to-run spread from cells that were re-run with the identical config.

    For every (case, S) present in BOTH grids, compare quality and B.  This is the
    empirical justification for the "|Δ| < 1 pt is noise" rule in the doc: the
    original cells were 3-lane concurrent, the replicates single-card serial, so
    the pair also mixes timing conditions (quality is independent of that).
    """
    pairs: list[dict[str, Any]] = []
    for case in sorted(set(primary) & set(replicate)):
        by_s_primary = {p.steps: p for p in primary[case]}
        for point in replicate[case]:
            other = by_s_primary.get(point.steps)
            if other is None or point.quality is None or other.quality is None:
                continue
            pairs.append(dict(
                case=case, S=point.steps,
                primary_quality=round(other.quality, 4),
                replicate_quality=round(point.quality, 4),
                delta_quality=round(point.quality - other.quality, 4),
                primary_b=other.b_seconds, replicate_b=point.b_seconds,
                delta_total=(round(point.total - other.total, 4)
                             if point.total is not None and other.total is not None else None)))
    if not pairs:
        return dict(available=False, n_pairs=0)
    deltas = [p["delta_quality"] for p in pairs]
    mean = statistics.fmean(deltas)
    sigma = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    abs_deltas = [abs(d) for d in deltas]
    b_ratios = [p["replicate_b"] / p["primary_b"] for p in pairs
                if p["primary_b"] and p["replicate_b"]]
    return dict(available=True, n_pairs=len(pairs), mean_delta_quality=round(mean, 4),
                sigma_quality=round(sigma, 4), mean_abs_delta_quality=round(
                    statistics.fmean(abs_deltas), 4),
                max_abs_delta_quality=round(max(abs_deltas), 4),
                b_ratio_median=round(statistics.median(b_ratios), 3) if b_ratios else None,
                pairs=pairs)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom_x = sum((x - mx) ** 2 for x in xs)
    denom_y = sum((y - my) ** 2 for y in ys)
    if denom_x <= 0 or denom_y <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (denom_x * denom_y) ** 0.5


def static_rule_screen(grouped: dict[str, list[Row]], cases: Sequence[str],
                       grid: Sequence[int], anchor: int = 5000) -> dict[str, Any]:
    """Can a cheap PRE-B scene statistic predict the per-scene optimal S?

    This screens the ``(d)-static`` variant allowed by task-3: measure a statistic
    before module B (init Gaussian count / number of train views), bucket scenes by
    its median, and use the in-bucket median optimal S for every scene.  The rule
    itself uses no test information; the *screening* uses the sweep's argmax and is
    therefore in-sample - the leave-one-out column is the honest generalisation
    estimate.  With 13 scenes this is a lead, not a validated policy.
    """
    stats: dict[str, dict[str, float]] = {}
    best_S: dict[str, int] = {}
    for case in cases:
        points = grouped[case]
        best = argmax_total(points)
        if best is None:
            continue
        best_S[case] = best.steps
        candidates: dict[str, float] = {}
        first = points[0]
        for key in ("num_init_gaussians", "num_train_views"):
            value = first.extra.get(key)
            if value is not None:
                candidates[key] = float(value)
        if candidates:
            stats[case] = candidates
    if len(best_S) < 5 or not stats:
        return dict(available=False, n=len(best_S),
                    reason="need >=5 scenes with a pre-B statistic; none usable")
    usable = [key for key in ("num_init_gaussians", "num_train_views")
              if all(key in stats.get(c, {}) for c in best_S)]
    out: dict[str, Any] = dict(available=True, n=len(best_S), by_stat={})
    ranked_best = _rank([float(best_S[c]) for c in best_S])
    for key in usable:
        values = [stats[c][key] for c in best_S]
        ranked_values = _rank(values)
        rho = _pearson(ranked_values, ranked_best)
        if rho is None:
            continue
        pairs = sorted(((stats[c][key], best_S[c]) for c in best_S))
        hi = [s for _, s in pairs[len(pairs) // 2:]]
        lo = [s for _, s in pairs[:len(pairs) // 2]]
        threshold = statistics.median([v for v, _ in pairs])
        s_hi, s_lo = int(round(statistics.median(hi))), int(round(statistics.median(lo)))
        grid_list = list(grid)
        s_hi = min(grid_list, key=lambda s: abs(s - s_hi))
        s_lo = min(grid_list, key=lambda s: abs(s - s_lo))
        total_lookup = {c: {p.steps: p for p in grouped[c]} for c in best_S}

        def evaluate(assign_s: dict[str, int]) -> float | None:
            got = []
            for case, choice in assign_s.items():
                point = total_lookup[case].get(choice)
                if point is not None and point.valid and point.total is not None:
                    got.append(point.total)
            return round(statistics.fmean(got), 4) if got else None

        assign = {c: (s_hi if stats[c][key] >= threshold else s_lo) for c in best_S}
        in_sample = evaluate(assign)
        loo_totals, loo_picks = [], []
        for held in best_S:
            rest = [c for c in best_S if c != held]
            rest_pairs = sorted((stats[c][key], best_S[c]) for c in rest)
            rest_hi = [s for _, s in rest_pairs[len(rest_pairs) // 2:]]
            rest_lo = [s for _, s in rest_pairs[:len(rest_pairs) // 2]]
            rest_threshold = statistics.median([v for v, _ in rest_pairs])
            r_hi = min(grid_list, key=lambda g: abs(g - int(round(statistics.median(rest_hi)))))
            r_lo = min(grid_list, key=lambda g: abs(g - int(round(statistics.median(rest_lo)))))
            choice = r_hi if stats[held][key] >= rest_threshold else r_lo
            point = total_lookup[held].get(choice)
            if point is not None and point.valid and point.total is not None:
                loo_totals.append(point.total)
                loo_picks.append(choice)
        out["by_stat"][key] = dict(
            spearman_rho=round(rho, 3), threshold=threshold,
            S_high=s_hi, S_low=s_lo, in_sample_mean=in_sample,
            loo_mean=round(statistics.fmean(loo_totals), 4) if loo_totals else None,
            loo_n=len(loo_totals),
            loo_different=sum(1 for c, s in zip(best_S, loo_picks)
                              if s != best_S[c]))
    out["argmax_S"] = best_S
    return out


def _fit_step_seconds(points: list[Row]) -> float | None:
    """OLS slope of B vs S (s/step) over the valid measured cells."""
    pts = [(p.steps, p.b_seconds) for p in points if p.b_seconds is not None]
    if len(pts) < 2:
        return None
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def marginal_stop(points: list[Row], factor: float = 1.0,
                  smooth_b: bool = True) -> dict[str, Any]:
    """Grid marginal-value stop rule.

    Walk the step grid upward; accept one more interval when its measured
    quality gain per second of module-B time exceeds ``factor * lambda`` with
    ``lambda = 20/(0.9*budget)``.  Stop at the last accepted point.

    NOTE: on a *measured test-quality curve* this is the **ceiling** of any
    stopping rule that consumes test quality - it is the "perfect information"
    simulation.  A deployable rule must feed the same arithmetic from
    non-context train views (see the doc), so the number this emits is an upper
    bound, never a claim about the deployable rule's measured gain.
    """
    pts = [p for p in points if p.valid and p.quality is not None]
    if not pts:
        return {"ok": False, "reason": "no valid points"}
    lam = pts[0].lam
    if lam is None:
        return {"ok": False, "reason": "no budget"}
    chosen = pts[0]
    slope_b = _fit_step_seconds(pts) if smooth_b else None
    if slope_b is not None and slope_b <= 0:
        slope_b = None
    intervals = []
    for prev, nxt in zip(pts, pts[1:]):
        db = nxt.b_seconds - prev.b_seconds
        if slope_b is not None:
            # A single 500-step cell can be cold-start inflated (or lane-noise), which
            # would fake a huge/negative marginal.  The stopping rule only needs the
            # *expected* B cost of the next ΔS steps, so use the fitted slope.
            db = slope_b * (nxt.steps - prev.steps)
        dq = nxt.quality - prev.quality
        if db <= 0.0:
            # non-monotone timing: a later point that is no slower and no worse
            # dominates the earlier one - keep walking.
            if dq >= 0.0:
                chosen = nxt
                intervals.append(dict(from_S=prev.steps, to_S=nxt.steps, delta_q=dq,
                                      delta_b=db, marginal=None, decision="dominates"))
                continue
            intervals.append(dict(from_S=prev.steps, to_S=nxt.steps, delta_q=dq,
                                  delta_b=db, marginal=None, decision="stop-nonmonotone"))
            break
        slope = dq / db
        decision = "continue" if slope >= factor * lam else "stop"
        intervals.append(dict(from_S=prev.steps, to_S=nxt.steps, delta_q=round(dq, 4),
                              delta_b=round(db, 3), marginal=round(slope, 4),
                              threshold=round(factor * lam, 4), decision=decision))
        if decision == "stop":
            break
        chosen = nxt
    return {"ok": True, "chosen_S": chosen.steps, "chosen_total": round(chosen.total, 4),
            "chosen_quality": round(chosen.quality, 4), "lambda": round(lam, 4),
            "factor": factor, "intervals": intervals}


def comparison_set(grouped: dict[str, list[Row]], grid: Sequence[int],
                   anchor: int = 5000) -> tuple[list[str], str]:
    """Scenes usable for an apples-to-apples policy comparison.

    Preference order: (1) scenes with *every* grid S present and valid;
    (2) otherwise scenes with a valid point at ``anchor`` (the fixed-5000
    anchor).  The second case is a degraded comparison and is reported as such;
    it is never silently mixed with a larger set.
    """
    all_cases = [c for c in sorted(grouped) if grouped[c]]
    complete = [c for c in all_cases
                if all(any(p.steps == s and p.valid for p in grouped[c]) for s in grid)]
    if complete and len(complete) >= max(1, len(all_cases) - 1):
        # allow at most one incomplete scene (e.g. a final failed cell) so the
        # headline comparison keeps the full 13-scene set instead of shrinking
        return complete, "complete" if len(complete) == len(all_cases) else "complete-minus-one"
    anchored = [c for c in all_cases
                if any(p.steps == anchor and p.valid for p in grouped[c])]
    return anchored, ("anchor-only" if anchored else "none")


def policy_means(grouped: dict[str, list[Row]], grid: Sequence[int],
                 cases: Sequence[str] | None = None, anchor: int = 5000) -> dict[str, Any]:
    """Fixed S / best uniform S / oracle / marginal-stop means over ``cases``."""
    cases = [c for c in (cases or sorted(grouped)) if grouped.get(c)]
    per_step: dict[int, dict[str, Any]] = {}
    for steps in grid:
        totals, used, invalid = [], [], []
        for case in cases:
            point = next((p for p in grouped[case] if p.steps == steps), None)
            if point is None:
                continue
            if point.valid and point.total is not None:
                totals.append(point.total)
                used.append(case)
            else:
                invalid.append(case)
        per_step[steps] = dict(
            n=len(totals), n_cases=len(cases),
            mean_total=round(statistics.fmean(totals), 4) if totals else None,
            mean_quality=round(statistics.fmean(
                [next(p for p in grouped[c] if p.steps == steps).quality for c in used]), 4)
            if used else None,
            mean_b=round(statistics.fmean(
                [next(p for p in grouped[c] if p.steps == steps).b_seconds for c in used]), 3)
            if used else None,
            invalid_cases=invalid,
            full_coverage=len(totals) == len(cases) and len(cases) > 0,
        )
    full = {s: v for s, v in per_step.items() if v["full_coverage"]}
    best_uniform = max(full.values(), key=lambda v: v["mean_total"]) if full else None
    best_uniform_S = (next(s for s, v in per_step.items() if v is best_uniform)
                      if best_uniform else None)

    fixed = per_step.get(anchor)
    oracle_rows = []
    for case in cases:
        best = argmax_total(grouped[case])
        if best is not None:
            oracle_rows.append((case, best))
    oracle_mean = (round(statistics.fmean([b.total for _, b in oracle_rows]), 4)
                   if oracle_rows else None)

    stop_factors = (0.5, 1.0, 1.5)
    stop_sensitivity: dict[float, dict[str, Any]] = {}
    for factor in stop_factors:
        sims = {case: marginal_stop(grouped[case], factor=factor) for case in cases}
        sims = {case: s for case, s in sims.items() if s.get("ok")}
        stop_sensitivity[factor] = dict(
            mean_total=round(statistics.fmean([s["chosen_total"] for s in sims.values()]), 4)
            if sims else None,
            n=len(sims),
            chosen={case: s["chosen_S"] for case, s in sims.items()})
    stops = {case: marginal_stop(grouped[case], factor=1.0) for case in cases}
    stops = {case: s for case, s in stops.items() if s.get("ok")}
    stop_mean = (round(statistics.fmean([s["chosen_total"] for s in stops.values()]), 4)
                 if stops else None)

    def delta(value: float | None) -> float | None:
        if value is None or fixed is None or fixed["mean_total"] is None:
            return None
        return round(value - fixed["mean_total"], 4)

    policies = [
        dict(key="fixed_5000", label=f"(a) fixed S={anchor} (current delivery)",
             mean_total=fixed["mean_total"] if fixed else None,
             n=fixed["n"] if fixed else 0, n_cases=len(cases), delta=None),
        dict(key="uniform", label="(b) best single uniform S",
             S=best_uniform_S, mean_total=best_uniform["mean_total"] if best_uniform else None,
             n=best_uniform["n"] if best_uniform else 0, n_cases=len(cases),
             delta=delta(best_uniform["mean_total"] if best_uniform else None)),
        dict(key="oracle", label="(c) per-scene oracle (DIAGNOSTIC ceiling only)",
             S="per-scene", mean_total=oracle_mean, n=len(oracle_rows), n_cases=len(cases),
             delta=delta(oracle_mean)),
        dict(key="marginal_stop",
             label="(d) marginal-value stop simulated on TEST curves (optimistic, NOT deployable)",
             S="per-scene", mean_total=stop_mean, n=len(stops), n_cases=len(cases),
             delta=delta(stop_mean)),
    ]
    return dict(cases=cases, per_step=per_step, best_uniform_S=best_uniform_S,
                stop_sensitivity=stop_sensitivity,
                policies=policies, per_scene_oracle={
                    case: dict(S=b.steps, quality=round(b.quality, 3),
                               b_seconds=b.b_seconds, total=round(b.total, 3))
                    for case, b in oracle_rows},
                marginal_stops=stops)


def serial_correction(grouped: dict[str, list[Row]],
                      anchors: dict[str, dict[int, float]],
                      warnings: list[str]) -> dict[str, dict[int, float]]:
    """Estimate single-card B(S) per scene from serial anchors + the grid slope.

    The concurrency inflation acts mostly as a level shift, so the *shape* B(S) is
    better estimated from the full grid (OLS slope r over 6 cells) while the
    *level* must come from a single-card anchor.  For each scene::

        B_serial(S) = B_anchor + r_grid * (S - S_anchor)

    with the anchor closest to 5000 preferred (an anchor at S=5000 also gives the
    node-speed ratio used by --budget-scale).  Scenes with no grid slope fall back
    to the median slope across scenes with one.  Returns ``case -> {S: b_serial}``.
    """
    slopes: dict[str, float] = {}
    for case, points in grouped.items():
        r = _fit_step_seconds(points)
        if r is not None and r > 0:
            slopes[case] = r
    if not slopes:
        warnings.append("serial correction: no positive B(S) slope in any scene; cannot correct")
        return {}
    median_slope = statistics.median(slopes.values())
    out: dict[str, dict[int, float]] = {}
    for case, points in grouped.items():
        scene_anchors = anchors.get(case) or {}
        if not scene_anchors:
            continue
        # prefer the anchor closest to 5000 (node-normalisation anchor)
        s_ref = min(scene_anchors, key=lambda s: abs(s - 5000))
        b_ref = scene_anchors[s_ref]
        r = slopes.get(case, median_slope)
        out[case] = {p.steps: round(b_ref + r * (p.steps - s_ref), 3)
                     for p in points if p.b_seconds is not None}
        # measured anchors are always kept verbatim (never re-fit)
        for s_anchor, b_anchor in scene_anchors.items():
            if s_anchor in out[case]:
                out[case][s_anchor] = round(float(b_anchor), 3)
    return out


# ---------------------------------------------------------------- reporting
def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report: dict[str, Any], title: str) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines += ["## 0. 输入与覆盖", ""]
    lines.append(f"- 结果文件：`{report['inputs']['results']}`")
    if report["inputs"].get("reference"):
        lines.append(f"- 13 例串行锚表：`{report['inputs']['reference']}`")
    if report["inputs"].get("serial_anchors"):
        lines.append(f"- 串行计时锚：`{report['inputs']['serial_anchors']}`")
    lines.append(f"- 网格：{report['inputs']['grid']}")
    lines.append(f"- 场景数：{len(report['cases'])}；行数：{report['n_rows']}；"
                 f"有效行：{report['n_valid']}")
    scaling = report.get("budget_scaling") or {}
    if scaling.get("applied"):
        factors = scaling.get("factors") or {}
        spread = ""
        if factors:
            vals = sorted(factors.values())
            spread = f"（{vals[0]:.3f}–{vals[-1]:.3f}×）"
        lines.append(f"- **节点预算敏感性检查**：开（来源 {scaling.get('source')}{spread}）；"
                     f"主表仍用 E2 自身 S={report.get('fixed_steps', 5000)} 作基线，缩放结果只在 §2a 作敏感性对照")
        if scaling.get("robust") is False:
            lines.append("- **⚠ 敏感性检查结论不一致（raw vs 缩放）：S 选择对计时节点不稳健，不要只报一个版本**")
    else:
        lines.append("- **节点预算敏感性检查**：关（主表用 E2 自身 S=5000 基线，节点差已消去；"
                     "需要时加 `--budget-scale auto` 做敏感性检查）")
    comparison = report.get("comparison") or {}
    lines.append(f"- **策略对比集合**：{comparison.get('n', 0)}/{comparison.get('n_all_cases', 0)} 个场景"
                 f"（{comparison.get('kind', '?')}；只有该集合内的均值可直接比较）")
    missing = [f"{c}@{s}" for c, by_s in report["coverage"].items() for s, ok in by_s.items() if not ok]
    if missing:
        lines.append(f"- **缺失格子（显式列出，不猜测）：** {', '.join(missing)}")
    else:
        lines.append("- 缺失格子：无（全网格覆盖）")
    lines.append("")
    lines += ["## 1. 每场景 Pareto 表", "",
              "标记：`◀` = 该场景 total argmax；`·` = quality-vs-B 非支配点；"
              "`INVALID` = B ≥ budget（按 AGENTS.md 从均值中剔除）。", ""]
    lines.append("| case | S | quality | B(s) | TIME | total | 标记 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for case in report["cases"]:
        table = report["scenes"][case]
        for point in table["points"]:
            marks = []
            if point["argmax"]:
                marks.append("◀")
            if point["frontier"]:
                marks.append("·")
            if not point["valid"]:
                marks.append("**INVALID**")
            lines.append(
                f"| {case} | {point['S']} | {fmt(point['quality'])} | {fmt(point['b_seconds'])} "
                f"| {fmt(point['time'])} | {fmt(point['total'])} | {' '.join(marks)} |")
    lines += ["", "### 1b. 每场景摘要", "",
              "| case | argmax S | quality@argmax | B@argmax | total@argmax | total@S=5000 | Δ vs 5000 | frontier S |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        table = report["scenes"][case]
        lines.append(
            f"| {case} | {fmt(table['argmax_S'])} | {fmt(table['argmax_quality'])} "
            f"| {fmt(table['argmax_b'])} | {fmt(table['argmax_total'])} "
            f"| {fmt(table['total_5000'])} "
            f"| {fmt(table['delta_vs_5000'])} | {table['frontier_S']} |")
    lines.append("")
    static = report.get("static_rule") or {}
    if static.get("available"):
        lines += ["### 1c. 便宜的 pre-B 统计量能否预测 argmax S（筛查，非策略）", "",
                  "| 统计量 | Spearman ρ（与 argmax S 的秩） | 中位阈值 | 高桶 S / 低桶 S | 样本内均值 | LOO 均值 | LOO 改变选点 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for key, cell in static["by_stat"].items():
            lines.append(
                f"| {key} | {fmt(cell['spearman_rho'], 3)} | {fmt(cell['threshold'], 1)} "
                f"| {cell['S_high']} / {cell['S_low']} | {fmt(cell['in_sample_mean'], 3)} "
                f"| {fmt(cell['loo_mean'], 3)} | {cell['loo_different']}/{static['n']} |")
        lines += ["", "> 阈值是样本内选的，LOO 才是泛化估计；13 例量级下 |ρ|<0.5 基本等于没有信号，"
                  "不能据此上线 static 规则（仍需 fixed-best-uniform 兜底）。", ""]
    repeat = report.get("repeat_runs") or {}
    if repeat.get("available"):
        lines += ["### 1d. 重复运行噪声（同配置重跑格 vs 主扫描格）", "",
                  f"- 配对格数：{repeat['n_pairs']}；Δquality 均值 {fmt(repeat['mean_delta_quality'], 3)}、"
                  f"σ {fmt(repeat['sigma_quality'], 3)}、|Δ| 均值 {fmt(repeat['mean_abs_delta_quality'], 3)}、"
                  f"最大 |Δ| {fmt(repeat['max_abs_delta_quality'], 3)}",
                  f"- B 比值（重跑/主扫）中位：{fmt(repeat['b_ratio_median'], 3)}"
                  f"（<1 说明主扫描的并发确实抬高了 B）", "",
                  "| case | S | 主扫 quality | 重跑 quality | Δquality | Δtotal |",
                  "|---|---:|---:|---:|---:|---:|"]
        for pair in repeat["pairs"]:
            lines.append(f"| {pair['case']} | {pair['S']} | {fmt(pair['primary_quality'])} "
                         f"| {fmt(pair['replicate_quality'])} | {fmt(pair['delta_quality'], 3)} "
                         f"| {fmt(pair['delta_total'], 3)} |")
        lines += ["", "> 这组配对同时给出了「同配置重跑质量差」和「并发 B 抬高比」；"
                  "§7 的 <1 分噪声阈值应以 |Δ| 均值/最大值为准。", ""]
    lines += [f"## 2. 策略对比（集合内均值，fixed anchor = E2 自身 S={report.get('fixed_steps', 5000)}）", "",
              "> (a) 的基线是**扫描里同一节点/同一 init 的 S=5000 格**，不是 H1 的 13 例表"
              "（后者只作外部 sanity anchor，见 §5）。", "",
              "| 策略 | S | n/集合 | 集合内均值 total | 相对 (a) |",
              "|---|---|---:|---:|---:|"]
    for policy in report["policies"]["policies"]:
        lines.append(f"| {policy['label']} | {policy.get('S', '-')} "
                     f"| {policy['n']}/{policy.get('n_cases', '?')} "
                     f"| {fmt(policy['mean_total'], 3)} | {fmt(policy['delta'], 3)} |")
    lines.append("")
    if scaling.get("applied") and scaling.get("policies"):
        lines += ["### 2a. 节点预算敏感性（把 H1 预算按实测节点比缩放后重算，仅对照）", "",
                  "| 策略 | S | n | 均值 total（缩放后） | 相对 (a) |",
                  "|---|---|---:|---:|---:|"]
        for policy in scaling["policies"]:
            lines.append(f"| {policy['label']} | {policy.get('S', '-')} | {policy['n']} "
                         f"| {fmt(policy['mean_total'], 3)} | {fmt(policy['delta'], 3)} |")
        lines += ["", (f"结论：raw best-uniform S={scaling.get('raw_best_uniform_S')} "
                       f"(Δ={fmt(scaling.get('raw_uniform_delta'), 3)})，"
                       f"缩放后 S={scaling.get('scaled_best_uniform_S')} "
                       f"(Δ={fmt(scaling.get('scaled_uniform_delta'), 3)}) → "
                       f"{'一致，决策稳健' if scaling.get('robust') else '不一致，决策不稳健'}"), ""]
    lines += ["### 2b. 每个 S 的均匀策略均值", "",
              "| S | n/场景数 | 均值 total | 均值 quality | 均值 B(s) | INVALID 场景 |",
              "|---:|---:|---:|---:|---:|---|"]
    for steps, cell in report["policies"]["per_step"].items():
        lines.append(f"| {steps} | {cell['n']}/{cell['n_cases']} | {fmt(cell['mean_total'], 3)} "
                     f"| {fmt(cell['mean_quality'])} | {fmt(cell['mean_b'])} "
                     f"| {', '.join(cell['invalid_cases']) or '-'} |")
    lines.append("")
    lambdas = sorted({round(b["delta"] if False else v, 4)
                      for v in report["lambda_by_case"].values() if v is not None})
    sensitivity = report["policies"].get("stop_sensitivity") or {}
    if sensitivity:
        lines += ["### 2c. 边际停止阈值的敏感性（在测试曲线上模拟）", "",
                  "| 阈值倍数 γ | n | 均值 total |",
                  "|---:|---:|---:|"]
        for factor, cell in sorted(sensitivity.items()):
            lines.append(f"| {factor:g}×λ | {cell['n']} | {fmt(cell['mean_total'], 3)} |")
        lines += ["", f"> λ 越小→越晚停、越大→越早停；可部署规则的 train 曲线尚未测，"
                  f"这组数只说明「阈值选多少」对上限的影响。", ""]
    lines += ["### 2d. 每场景 B(S) 斜率与「每 1000 步的 TIME 成本」", "",
              "| case | r (s/1000 步) | 截距 (s) | 每 1000 步 TIME 成本 = 盈亏质量 | n |",
              "|---|---:|---:|---:|---:|"]
    for case in report["cases"]:
        rate = report["step_rates"].get(case)
        if not rate:
            lines.append(f"| {case} | - | - | - | - |")
            continue
        lines.append(f"| {case} | {fmt(rate['r_s_per_1000'])} | {fmt(rate['intercept'])} "
                     f"| {fmt(rate['breakeven_q_per_1000'])} | {rate['n']} |")
    if report.get("class_breakeven"):
        lines += ["", f"按数据集团中位：{report['class_breakeven']}（质量分/1000 步）"]
    lines += ["", "## 3. 边际停止规则（grid marginal-value stop）", "",
              f"阈值 λ = 20/(0.9·budget)，取值区间 {lambdas}", "",
              "> 这里的 (d) 用**测试集质量曲线**模拟停止规则，是「完美信息」的乐观上界；"
              "可部署版本必须改用非 context 的 train 视图，见 PARETO-STEPS 文档 §5。"
              "区间中的 B 增量用 grid 上的 `B=a+rS` 拟合斜率而非单格差值（并发/冷启动噪声会让"
              "单格 ΔB 出现负值，那是测量伪影）。", "",
              "| case | budget | λ | 规则选 S | 规则 total | oracle S | oracle total | 区间决策 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        stop = (report["policies"].get("marginal_stops_all") or {}).get(case)
        oracle = report["policies"]["per_scene_oracle"].get(case, {})
        if not stop:
            lines.append(f"| {case} | {fmt(report['budget_by_case'][case])} | - | - | - | "
                         f"{oracle.get('S', '-')} | {fmt(oracle.get('total'), 3)} | 无有效点 |")
            continue
        def _interval(i: dict[str, Any]) -> str:
            if i.get("marginal") is None:
                return (f"{i['from_S']}→{i['to_S']}: ΔB={fmt(i.get('delta_b'))}"
                        f"({i['decision']})")
            return (f"{i['from_S']}→{i['to_S']}: Δq/ΔB={fmt(i['marginal'])}"
                    f"{'<λ' if i['decision'] == 'stop' else ''}")

        decisions = "; ".join(_interval(i) for i in stop["intervals"]) or "-"
        lines.append(
            f"| {case} | {fmt(report['budget_by_case'][case])} | {fmt(stop['lambda'], 3)} "
            f"| {stop['chosen_S']} | {fmt(stop['chosen_total'], 3)} "
            f"| {oracle.get('S', '-')} | {fmt(oracle.get('total'), 3)} | {decisions} |")
    lines.append("")
    if report.get("serial_corrected"):
        lines += ["## 4. 串行计时校正（并发 B 只用于质量，不用于 TIME）", "",
                  "B(S) 用「串行锚水平 + 扫描 grid 的 OLS 斜率」校正（外推值，须显式标注）；"
                  "TIME 用权威预算（与 H1 13 例锚同口径；需要节点敏感性时加 `--budget-scale auto`）：", "",
                  "| 策略 | 均值 total（串行校正） | 相对 (a) |",
                  "|---|---:|---:|"]
        for policy in report["serial_corrected"]["policies"]:
            lines.append(f"| {policy['label']} | {fmt(policy['mean_total'], 3)} "
                         f"| {fmt(policy['delta'], 3)} |")
        sc = report["serial_corrected"]
        if sc.get("oracle_measured"):
            om = sc["oracle_measured"]
            lines += ["", f"- 仅用**实测锚格**的 oracle（不靠外推）：fixed {fmt(om['fixed_mean'], 3)}、"
                      f"oracle {fmt(om['mean'], 3)}、增益 **{fmt(om['gain'], 3)}**（n={om['n']}）"]
        if sc.get("fully_serial"):
            fs = sc["fully_serial"]
            lines += ["", f"- **全串行实测**（质量与 B 都来自单卡重跑）：fixed {fmt(fs['fixed_mean'], 3)}、"
                      f"oracle {fmt(fs['oracle_mean'], 3)}、增益 **{fmt(fs['gain'], 3)}**"
                      f"（n={fs['n_fixed']}/{fs['n_oracle']}）"]
        lines.append("")
    if report.get("reference"):
        ref = report["reference"]
        lines += ["## 5. 13 例串行锚（H1 `scene13r15/results.jsonl`）", "",
                  f"- baseline O0 均值 total = **{fmt(ref['baseline_mean'], 3)}**（n={ref['baseline_n']}）",
                  f"- delivery S=5000 均值 total = **{fmt(ref['delivery_mean'], 3)}**"
                  f"（n={ref['delivery_n']}）",
                  f"- Δ = {fmt(ref['delta'], 3)}", ""]
        sweep = (report["policies"]["per_step"].get(5000) or {}).get("mean_total")
        if sweep is not None and ref["delivery_mean"] is not None:
            lines.append(f"- 扫描表 S=5000 均值 = {fmt(sweep, 3)}（n={report['comparison'].get('n', 0)} 场景）；"
                         f"与 13 例锚的**逐场景交集**交叉校验见 §6（两表场景集不同，均值不可直接相减）")
            lines.append("")
    lines += ["## 6. 方差与稳健性", ""]
    for note in report["robustness"]:
        lines.append(f"- {note}")
    lines.append("")
    if report["warnings"]:
        lines += ["## 7. 警告", ""]
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- assemble
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    grid = [int(s) for s in str(args.steps_grid).split(",") if s.strip()]
    rows: list[Row] = []
    paired: list[dict[str, Any]] = []
    for path in [Path(p) for p in args.results]:
        sub_rows, sub_paired = load_any(path, warnings)
        rows.extend(sub_rows)
        paired.extend(sub_paired)
    # a paired table passed via --results is not a Pareto grid: keep only its
    # anchor role (S=10000 baseline / S=5000 delivery), never its rows in the
    # per-scene Pareto means.
    rows = [r for r in rows if not r.extra.get("paired_table")]
    replicate_rows: list[Row] = []
    if args.replicate:
        rep_rows, _ = load_any(Path(args.replicate), warnings)
        replicate_rows = [r for r in rep_rows if not r.extra.get("paired_table")
                          and r.steps in grid]
        if not replicate_rows:
            warnings.append(f"--replicate {args.replicate}: no usable grid row")
    reference: list[Row] = []
    if args.reference:
        ref_rows, _ = load_any(Path(args.reference), warnings)
        reference = [r for r in ref_rows if r.extra.get("paired_table")]
        if not reference:
            warnings.append(f"--reference {args.reference}: no paired baseline/delivery row "
                            f"found; serial anchor section stays empty")

    # keep the sweep/occurve rows only for the Pareto analysis; the paired rows
    # are the separate serial anchor and only used for the S=5000 cross-check.
    for row in rows:
        if row.steps not in grid and not row.extra.get("paired_table"):
            warnings.append(f"{row.case}: S={row.steps} is outside --steps-grid {grid}; dropped")
    rows = [r for r in rows if r.steps in grid]

    # recompute scores; flag stored-score drift (a silent unit change would be a bug)
    for row in rows:
        if row.points is None or not row.stored:
            continue
        stored_total = row.stored.get("total_points")
        if stored_total is not None and row.total is not None and abs(stored_total - row.total) > 0.05:
            warnings.append(
                f"{row.case}@{row.steps}: stored total {stored_total:.3f} != recomputed "
                f"{row.total:.3f}; using recomputed (check for a different budget/scoring)")
        stored_valid = row.stored.get("valid")
        if stored_valid is not None and bool(stored_valid) != row.valid:
            warnings.append(f"{row.case}@{row.steps}: stored valid={stored_valid} != recomputed "
                            f"{row.valid}; using recomputed")

    grouped = {case: dedupe(points) for case, points in group_by_case(
        [r for r in rows if not r.extra.get("paired_table")]).items()}
    # duplicates (same case+S twice) are a provenance problem, not a tie
    seen: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.extra.get("paired_table"):
            continue
        key = (row.case, row.steps)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    if duplicates:
        warnings.append(f"duplicate (case,S) rows, last one wins: {duplicates[:10]}"
                        f"{' ...' if len(duplicates) > 10 else ''}")
    # stored budget vs the authoritative map
    for row in rows:
        if row.extra.get("paired_table"):
            continue
        stored_budget = row.extra.get("budget_seconds")
        if stored_budget is not None and row.budget is not None \
                and abs(float(stored_budget) - row.budget) > 1e-6:
            warnings.append(f"{row.case}@{row.steps}: sweep budget {stored_budget} != "
                            f"authoritative {row.budget}; scoring uses the authoritative value")

    # ---- node budget normalisation (SENSITIVITY ONLY) ---------------------
    # The E2 S=5000 cell is the primary fixed-5000 baseline (same node/init set
    # as the rest of the grid). The H1 budgets are used for the raw TIME term,
    # which is fine while all S cells share the node/init; --budget-scale then
    # re-runs the ranking with budgets scaled by the measured node ratio as a
    # sensitivity check. A verdict that flips between raw and scaled is NOT
    # robust and must be reported as such.
    reference_delivery_b: dict[str, float] = {}
    for row in reference:
        if row.extra.get("arm") == "delivery" and row.b_seconds:
            reference_delivery_b[row.case] = float(row.b_seconds)
    anchors = load_serial_anchors(Path(args.serial_anchors) if args.serial_anchors else None,
                                  warnings)
    scale_arg = str(args.budget_scale)
    scale_factors: dict[str, float] = {}
    scale_source = "off"
    if scale_arg not in ("off", "none", "1", "1.0"):
        grouped_pre = {case: dedupe(points) for case, points in group_by_case(rows).items()}
        if scale_arg == "auto":
            scale_factors, scale_source = node_budget_factors(
                grouped_pre, reference_delivery_b, anchors, anchor=args.fixed_steps)
            if not scale_factors:
                warnings.append("budget scaling: no usable anchor (need --reference delivery B "
                                "and/or serial anchors); sensitivity check skipped")
                scale_source = "none"
        elif Path(scale_arg).exists():
            scale_factors = {str(k): float(v) for k, v in
                             json.loads(Path(scale_arg).read_text(encoding="utf-8")).items()}
            scale_source = f"file:{scale_arg}"
        else:
            scale_factors = {row.case: float(scale_arg) for row in rows}
            scale_source = f"constant:{scale_arg}"
        if scale_source == "sweep-anchor-cell":
            warnings.append("budget scaling uses the sweep's anchor cell, which is "
                            "concurrency-inflated: treat the scaled verdict as a lower bound")

    grouped = {case: dedupe(points) for case, points in group_by_case(rows).items()}
    replicate_grouped = {case: dedupe(points) for case, points in
                         group_by_case(replicate_rows).items()}
    repeat_report = repeat_noise(grouped, replicate_grouped)
    cases = sorted(grouped)
    compare_cases, compare_kind = comparison_set(grouped, grid, anchor=args.fixed_steps)
    if compare_kind == "anchor-only":
        warnings.append(
            f"not every scene has a complete grid; policy comparison uses only the "
            f"{len(compare_cases)} scenes with a valid S={args.fixed_steps} anchor "
            f"(out of {len(cases)}). This is a degraded comparison.")

    scene_reports: dict[str, Any] = {}
    coverage: dict[str, dict[int, bool]] = {}
    for case in cases:
        points = grouped[case]
        front = frontier(points)
        best = argmax_total(points)
        front_ids = {id(p) for p in front}
        scene_reports[case] = dict(
            points=[dict(S=p.steps, quality=None if p.quality is None else round(p.quality, 4),
                         b_seconds=p.b_seconds, time=p.time,
                         total=None if p.total is None else round(p.total, 4),
                         valid=p.valid, argmax=best is not None and p is best,
                         frontier=id(p) in front_ids,
                         num_gaussians=p.extra.get("num_gaussians"),
                         lane=p.extra.get("lane")) for p in points],
            argmax_S=best.steps if best else None,
            argmax_quality=None if best is None else round(best.quality, 4),
            argmax_b=best.b_seconds if best else None,
            argmax_total=None if best is None else round(best.total, 4),
            total_5000=next((round(p.total, 4) for p in points
                             if p.steps == 5000 and p.valid and p.total is not None), None),
            delta_vs_5000=(round(best.total - next(p.total for p in points
                                                   if p.steps == 5000 and p.valid), 3)
                           if best is not None and any(
                               p.steps == 5000 and p.valid and p.total is not None
                               for p in points) else None),
            frontier_S=", ".join(str(p.steps) for p in front) or "-",
        )
        coverage[case] = {s: any(p.steps == s for p in points) for s in grid}

    policies = policy_means(grouped, grid, compare_cases, anchor=args.fixed_steps)
    # per-scene simulation for the display table: all scenes, not just the set used
    # for the policy mean (a scene can have a short grid and still be interesting)
    policies["marginal_stops_all"] = {
        case: marginal_stop(grouped[case]) for case in cases}
    budget_scaling: dict[str, Any] = dict(applied=False, source="off", factors={},
                                          anchor=args.fixed_steps, policies=None,
                                          comparison=None, robust=None)
    if scale_factors:
        scaled_rows = rescale_budgets(rows, scale_factors)
        scaled_grouped = {c: dedupe(points) for c, points in group_by_case(scaled_rows).items()}
        scaled_compare, scaled_kind = comparison_set(scaled_grouped, grid,
                                                     anchor=args.fixed_steps)
        scaled = policy_means(scaled_grouped, grid, scaled_compare, anchor=args.fixed_steps)
        fixed_scaled = next(p for p in scaled["policies"] if p["key"] == "fixed_5000")
        for policy in scaled["policies"]:
            policy["delta"] = (round(policy["mean_total"] - fixed_scaled["mean_total"], 4)
                               if policy["mean_total"] is not None
                               and fixed_scaled["mean_total"] is not None else None)
        raw_uniform = next(p for p in policies["policies"] if p["key"] == "uniform")
        scaled_uniform = next(p for p in scaled["policies"] if p["key"] == "uniform")
        raw_fixed = next(p for p in policies["policies"] if p["key"] == "fixed_5000")
        same_S = raw_uniform.get("S") == scaled_uniform.get("S")

        def _sign(value: float | None) -> int:
            return 0 if value is None or abs(value) < 1e-9 else (1 if value > 0 else -1)

        same_sign = _sign(raw_uniform.get("delta")) == _sign(scaled_uniform.get("delta"))
        robust = bool(same_S and same_sign)
        budget_scaling = dict(applied=True, source=scale_source, factors=scale_factors,
                              anchor=args.fixed_steps, policies=scaled["policies"],
                              comparison=dict(kind=scaled_kind, cases=scaled_compare,
                                              n=len(scaled_compare)),
                              robust=robust,
                              raw_best_uniform_S=raw_uniform.get("S"),
                              scaled_best_uniform_S=scaled_uniform.get("S"),
                              raw_uniform_delta=raw_uniform.get("delta"),
                              scaled_uniform_delta=scaled_uniform.get("delta"))
        if not robust:
            warnings.append(
                "budget-scaling sensitivity: raw vs node-scaled ranking differ "
                f"(raw best uniform S={raw_uniform.get('S')} delta={raw_uniform.get('delta')}; "
                f"scaled S={scaled_uniform.get('S')} delta={scaled_uniform.get('delta')}); "
                "the S decision is NOT robust to the timing node")
    static_rule = static_rule_screen(grouped, compare_cases, grid, anchor=args.fixed_steps)
    step_rates = {case: step_time_rate(grouped[case], anchor=args.fixed_steps) for case in cases}
    class_rates: dict[str, list[float]] = {}
    for case, rate in step_rates.items():
        if rate and rate["breakeven_q_per_1000"] is not None:
            group = case.split("/")[0]
            class_rates.setdefault(group, []).append(rate["breakeven_q_per_1000"])
    class_rates = {k: round(statistics.median(v), 3) for k, v in class_rates.items()}

    reference_report = None
    if reference:
        base = [r for r in reference if r.extra.get("arm") == "baseline" and r.total is not None]
        deliv = [r for r in reference if r.extra.get("arm") == "delivery" and r.total is not None]
        if base and deliv:
            reference_report = dict(
                baseline_n=len(base), delivery_n=len(deliv),
                baseline_mean=round(statistics.fmean(r.total for r in base), 4),
                delivery_mean=round(statistics.fmean(r.total for r in deliv), 4),
                delta=round(statistics.fmean(r.total for r in deliv)
                            - statistics.fmean(r.total for r in base), 4),
                per_scene={r.case: dict(total=round(r.total, 3), quality=round(r.quality, 3),
                                        b_seconds=r.b_seconds)
                           for r in deliv})

    serial_corrected = None
    if anchors:
        estimated = serial_correction(grouped, anchors, warnings)
        if estimated:
            corrected = {}
            for case, by_s in estimated.items():
                corrected[case] = []
                for point in grouped[case]:
                    clone = Row(case=case, steps=point.steps, b_seconds=by_s.get(point.steps),
                                psnr=point.psnr, ssim=point.ssim, lpips=point.lpips,
                                source=point.source, extra={"serial_estimate": True})
                    corrected[case].append(clone)
            if scale_factors:
                corrected = {c: rescale_budgets(points, scale_factors)
                             for c, points in corrected.items()}
            corrected_group = {c: corrected[c] for c in corrected}
            corrected_policies = policy_means(corrected_group, grid,
                                              [c for c in compare_cases if c in corrected_group],
                                              anchor=args.fixed_steps)
            fixed = next(p for p in corrected_policies["policies"]
                         if p["key"] == "fixed_5000")
            for policy in corrected_policies["policies"]:
                policy["delta"] = (round(policy["mean_total"] - fixed["mean_total"], 4)
                                   if policy["mean_total"] is not None else None)
            # Oracle on measured anchors only: extrapolated B must not create phantom
            # argmax wins (each scene's oracle candidate AND its S=5000 cell were
            # measured serially, so no extrapolation is needed for this comparison).
            measured_rows: list[Row] = []
            for case in corrected_group:
                for point in corrected_group[case]:
                    if point.steps in anchors.get(case, {}):
                        measured_rows.append(point)
            measured_group = {c: dedupe(points) for c, points in group_by_case(measured_rows).items()}
            measured_oracle = []
            fixed_totals = []
            for case in corrected_group:
                points = measured_group.get(case, [])
                best = argmax_total(points)
                anchor_point = next((p for p in points if p.steps == args.fixed_steps), None)
                if best is not None:
                    measured_oracle.append(best.total)
                if anchor_point is not None and anchor_point.total is not None:
                    fixed_totals.append(anchor_point.total)
            measured_oracle_mean = (round(statistics.fmean(measured_oracle), 4)
                                    if measured_oracle else None)
            measured_fixed_mean = (round(statistics.fmean(fixed_totals), 4)
                                   if fixed_totals else None)
            # fully-serial variant: quality and B both from the single-card re-runs
            fully_serial = None
            if replicate_grouped:
                serial_fixed, serial_oracle = [], []
                for case in corrected_group:
                    pts = replicate_grouped.get(case, [])
                    best = argmax_total(pts)
                    anchor_point = next((p for p in pts if p.steps == args.fixed_steps), None)
                    if best is not None:
                        serial_oracle.append(best.total)
                    if anchor_point is not None and anchor_point.total is not None:
                        serial_fixed.append(anchor_point.total)
                if serial_fixed and serial_oracle:
                    fully_serial = dict(
                        fixed_mean=round(statistics.fmean(serial_fixed), 4),
                        oracle_mean=round(statistics.fmean(serial_oracle), 4),
                        n_fixed=len(serial_fixed), n_oracle=len(serial_oracle))
                    fully_serial["gain"] = round(fully_serial["oracle_mean"]
                                                 - fully_serial["fixed_mean"], 4)
            serial_corrected = dict(
                anchors=anchors, policies=corrected_policies["policies"],
                oracle_measured=dict(mean=measured_oracle_mean, fixed_mean=measured_fixed_mean,
                                     gain=(None if measured_oracle_mean is None
                                           or measured_fixed_mean is None
                                           else round(measured_oracle_mean - measured_fixed_mean, 4)),
                                     n=len(measured_oracle)),
                fully_serial=fully_serial,
                note="B(S) extrapolated from single-card anchors; not a measurement of "
                     "every cell. Oracle is restricted to measured anchor cells. "
                     "fully_serial uses the single-card re-runs for BOTH quality and B.")

    # ---- robustness notes ------------------------------------------------
    robustness: list[str] = []
    fixed = next((p for p in policies["policies"] if p["key"] == "fixed_5000"), None)
    uniform = next((p for p in policies["policies"] if p["key"] == "uniform"), None)
    oracle = next((p for p in policies["policies"] if p["key"] == "oracle"), None)
    if fixed and uniform and fixed["mean_total"] is not None and uniform["mean_total"] is not None:
        gain = uniform["mean_total"] - fixed["mean_total"]
        if abs(gain) < 1.0:
            robustness.append(
                f"best-uniform vs fixed-5000 差 {gain:+.3f} 分 < 1 分：属噪声量级，"
                f"需要 ≥3 次重复才能裁决（当前 n={uniform['n']} 场景单次）。")
        else:
            robustness.append(f"best-uniform vs fixed-5000 差 {gain:+.3f} 分。")
    if uniform and oracle and uniform["mean_total"] is not None and oracle["mean_total"] is not None:
        robustness.append(
            f"oracle 比 best-uniform 高 {oracle['mean_total'] - uniform['mean_total']:+.3f} 分："
            f"这是「允许逐场景选 S」的诊断上界，按 AGENTS.md 不可作为提交策略。")
    if fixed and fixed["n"] and fixed["n"] != fixed["n_cases"]:
        robustness.append(f"fixed S=5000 只有 {fixed['n']}/{fixed['n_cases']} 个场景有效，"
                          f"缺格场景需显式排除或补测。")
    # how many scenes would actually change under the oracle decision
    deltas = [report["delta_vs_5000"] for report in scene_reports.values()
              if report.get("delta_vs_5000") is not None]
    if deltas:
        within_noise = [d for d in deltas if abs(d) < 1.0]
        robustness.append(
            f"{len(within_noise)}/{len(deltas)} 个场景的 oracle 增益 vs S=5000 小于 1 分"
            f"（{sorted(round(d, 2) for d in deltas)}）：这些场景的 S 决策落在噪声内，"
            f"至少 3 次重复才可裁决。")
    # TIME sensitivity to noisy module-B wall clock
    all_b = [p.b_seconds for case in cases for p in grouped[case]
             if p.b_seconds is not None and p.valid]
    if all_b:
        median_b = statistics.median(all_b)
        med_budget = statistics.median([budget_for(c) or 0 for c in cases])
        if med_budget:
            lam = marginal_value(med_budget)
            robustness.append(
                f"TIME 敏感度：预算 {med_budget:.2f}s 时每 1s 模块 B ≈ {lam:.3f} 分；"
                f"中位 B={median_b:.1f}s 的 ±45% 并发膨胀 ≈ ±{0.45 * median_b * lam:.2f} 分，"
                f"足以翻转 <2 分的策略差异——并发 B 只能用于质量，不能用于策略排序。")
    # per-scene argmax S spread tells whether one uniform S can be near-oracle
    argmax_steps = [report["argmax_S"] for report in scene_reports.values()
                    if report["argmax_S"] is not None]
    if argmax_steps:
        robustness.append(f"每场景 argmax S 分布：{sorted(set(argmax_steps))}"
                          f"（中位 {statistics.median(argmax_steps):.0f}）。")
    # timing sanity: B must increase with S; flag any scene where it does not
    for case in cases:
        points = [p for p in grouped[case] if p.b_seconds is not None and p.valid]
        bad = [(a.steps, b.steps) for a, b in zip(points, points[1:])
               if b.b_seconds <= a.b_seconds]
        if bad:
            robustness.append(f"{case}: B 秒数在 {bad} 处不随 S 单调（并发/噪声），"
                              f"该场景的 TIME 结论需单卡串行复测。")
    lanes = sorted({p.extra.get("lane") for case in cases for p in grouped[case]
                    if p.extra.get("lane") is not None})
    if lanes:
        robustness.append(f"B 秒数来自 lane {lanes} 的并发扫描：TIME 项只能作相对参考，"
                          f"必须用串行锚（--serial-anchors）校正后才能下结论。")
    if reference_report and reference_report.get("per_scene"):
        # only compare scenes present in BOTH tables; otherwise the mean mixes sets
        common = []
        for case in cases:
            point = next((p for p in grouped.get(case, []) if p.steps == 5000 and p.valid), None)
            ref_scene = reference_report["per_scene"].get(case)
            if point is not None and ref_scene is not None:
                common.append((case, point.total, ref_scene["total"], point.b_seconds,
                               ref_scene["b_seconds"]))
        if common:
            sweep_mean = statistics.fmean(t[1] for t in common)
            ref_mean = statistics.fmean(t[2] for t in common)
            b_ratio = statistics.fmean(t[3] / t[4] for t in common if t[4])
            robustness.append(
                f"逐场景交集交叉校验（{len(common)} 例，S=5000）：扫描均值 {sweep_mean:.3f} vs "
                f"13 例串行锚 {ref_mean:.3f}（Δ {sweep_mean - ref_mean:+.3f}）；"
                f"扫描/H1 的 B 比值均值 {b_ratio:.3f}（≈1 表示两节点速度一致）")
        elif policies["per_step"].get(5000, {}).get("mean_total") is not None:
            robustness.append(
                "扫描表与 13 例锚没有共同场景，S=5000 绝对分不可交叉校验（场景集不同）。")

    return dict(
        title=args.title,
        fixed_steps=args.fixed_steps,
        inputs=dict(results=args.results, reference=args.reference,
                    serial_anchors=args.serial_anchors, grid=grid,
                    fixed_steps=args.fixed_steps),
        comparison=dict(kind=compare_kind, cases=compare_cases,
                        n=len(compare_cases), n_all_cases=len(cases)),
        budget_scaling=budget_scaling,
        n_rows=len(rows), n_valid=sum(1 for r in rows if r.valid),
        cases=cases, coverage=coverage, scenes=scene_reports,
        budget_by_case={c: budget_for(c) for c in cases},
        lambda_by_case={c: None if budget_for(c) is None else round(marginal_value(budget_for(c)), 4)
                        for c in cases},
        policies=policies, reference=reference_report,
        step_rates=step_rates, class_breakeven=class_rates, static_rule=static_rule,
        repeat_runs=repeat_report,
        serial_corrected=serial_corrected, robustness=robustness, warnings=warnings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", action="append", required=True, metavar="PATH",
                        help="results.jsonl, summary.json, or a directory (recursive); repeatable")
    parser.add_argument("--reference", default=None,
                        help="13-case paired table (scene13r15/results.jsonl) - serial S=5000 anchor")
    parser.add_argument("--replicate", default=None,
                        help="same-config re-run grid (results.jsonl/summary dir) to estimate "
                             "run-to-run quality noise and concurrent-B inflation")
    parser.add_argument("--serial-anchors", default=None,
                        help="JSONL of single-card {case, S, b_seconds} anchors for B(S) correction; "
                             "also used to normalise the H1-measured budget to the sweep node")
    parser.add_argument("--budget-scale", default="off",
                        help="budget-scaling SENSITIVITY check (default off): 'auto' scales "
                             "each budget by the measured node ratio (serial anchors preferred, "
                             "else the sweep anchor cell vs --reference); a constant float; or "
                             "a JSON file {case: factor}. Main tables always use the raw E2 "
                             "S=5000 baseline; a flipped verdict means the S choice is not robust")
    parser.add_argument("--fixed-steps", type=int, default=5000,
                        help="the currently-deployed fixed S used as policy (a) anchor (default 5000)")
    parser.add_argument("--steps-grid", default=",".join(map(str, DEFAULT_GRID)),
                        help="comma-separated S grid to compare (default: the task-2 sweep grid)")
    parser.add_argument("--title", default="Steps-Pareto analysis")
    parser.add_argument("--out-json", default=None, help="write the full machine-readable report")
    parser.add_argument("--out-md", default=None, help="write markdown tables here; default stdout")
    args = parser.parse_args(argv)

    report = analyze(args)
    if report["n_valid"] == 0 and not report.get("reference"):
        print("no valid row found - check --results paths/schemas", file=sys.stderr)
        return 2
    if report["n_valid"] == 0:
        print("no Pareto grid row found; emitting the 13-case serial anchor only", file=sys.stderr)
    markdown = markdown_report(report, args.title)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(markdown)
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
