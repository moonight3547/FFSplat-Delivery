"""Diagnostic only: old measured candidate times against new FastGS budgets.

GPU identities differ, so this never emits a validated competition total.
"""
import json
import math
from pathlib import Path
from statistics import mean
from competition.contracts import validate_reference


def quality_scores(metrics):
    clip = lambda x, cap: max(0., min(cap, x))
    return dict(psnr=clip((metrics['psnr']-15)*.8, 20),
                ssim=clip((metrics['ssim']-.7)*80, 20),
                lpips=clip((.25-metrics['lpips'])*200, 40))


def time_score(seconds, budget):
    if not math.isfinite(seconds) or not math.isfinite(budget) or seconds < 0 or budget <= 0:
        raise ValueError('Invalid time')
    if seconds >= budget:
        return None
    return min(20., 20*(1-seconds/budget)/.9)


def main():
    root = Path(__file__).resolve().parents[1]
    calibration = json.loads((root/'records/fastgs-random30k-calibration.json').read_text())
    history = json.loads((root/'records/historical-comparison.json').read_text())
    if not calibration['complete']:
        raise ValueError('Full calibration required')
    budgets = {r['id']: validate_reference(r['reference'])['total_seconds'] for r in calibration['rows']}
    result = dict(status='cross-GPU diagnostic estimate, not a measured same-GPU score',
                  official_score_100=None, validated_total=None, cells={})
    for name, cell in history['cells'].items():
        q = quality_scores(cell['mean_metrics'])
        scopes = {}
        for scope in ('B-only', 'E2E'):
            rows = []
            for r in cell['rows']:
                t = r['stage_seconds']['B'] if scope == 'B-only' else r['end_to_end_seconds']
                score = time_score(t, budgets[r['id']])
                rows.append(dict(id=r['id'], measured_old_gpu_seconds=t,
                                 measured_new_gpu_budget_seconds=budgets[r['id']],
                                 estimated_within_budget=score is not None, time_score_estimate=score))
            valid_estimate = all(r['estimated_within_budget'] for r in rows)
            ts = mean(r['time_score_estimate'] for r in rows) if valid_estimate else None
            scopes[scope] = dict(rows=rows, mean_time_score_estimate=ts,
                proxy_total_estimate=sum(q.values())+ts if ts is not None else None,
                validated_total=None)
        result['cells'][name] = dict(mean_metrics=cell['mean_metrics'], quality_scores=q, scopes=scopes)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
