#!/usr/bin/env python3
"""Build gate_ab/results.jsonl from the A/B matrix (+ gate decision per scene)."""
import glob, json, os, sys
BUDGET = {"mipnerf360/bicycle": 95.0, "mipnerf360/counter": 92.57}
TAU = 0.48

def q(psnr, ssim, lpips):
    return (20*max(0,min(1,(psnr-15)/25)) + 20*max(0,min(1,(ssim-0.70)/0.25))
            + 40*max(0,min(1,(0.25-lpips)/0.20)))

def tp(b, budget):
    return 20*max(0,min(1,(1-b/budget)/0.9))

def main(root):
    # gate decisions from the calibration table
    gate = {}
    for f in glob.glob(os.path.join(os.path.dirname(__file__), 'raw/gate/*.json')):
        g = json.load(open(f))
        gate.setdefault('_all', []).append(g)
    gate_by_case = {}
    table = json.load(open(os.path.join(os.path.dirname(__file__), 'train_lpips_table.json')))
    for r in table['rows']:
        gate_by_case[r['case']] = dict(train_lpips=r['train_lpips'],
                                       keep=0.6 if r['train_lpips'] > TAU else 1.0)
    rows = []
    ctrls = {}
    for f in sorted(glob.glob(os.path.join(root, '*', '*', 'sweep', 'summary.json'))):
        arm = f.split(os.sep)[-3]
        d = json.load(open(f))
        case = d['case']; budget = BUDGET.get(case, 94.29)
        rec = json.load(open(os.path.join(os.path.dirname(f), 'train_record.json')))
        res = d['results'][-1]
        b = res['b_seconds_estimate']
        qq, tt = q(res['psnr'], res['ssim'], res['lpips']), tp(b, budget)
        row = dict(case=case, arm=arm, num_gaussians=res['num_gaussians'],
                   b_seconds=b, psnr=res['psnr'], ssim=res['ssim'], lpips=res['lpips'],
                   quality_points=round(qq, 4), time_points=round(tt, 4),
                   total_points=round(qq + tt, 4), budget_seconds=budget,
                   stage1_seconds=rec.get('stage1_seconds'),
                   prune_events=rec.get('prune_events') or [],
                   gate=gate_by_case.get(case), summary_json=f)
        if arm == 'ctrl':
            ctrls[case] = row
        rows.append(row)
    for row in rows:
        c = ctrls.get(row['case'])
        if c and row['arm'] != 'ctrl':
            row.update(ctrl_b_seconds=c['b_seconds'], ctrl_q=c['quality_points'],
                       d_q=round(row['quality_points'] - c['quality_points'], 4),
                       d_b=round(row['b_seconds'] - c['b_seconds'], 3),
                       d_time=round(row['time_points'] - c['time_points'], 4),
                       d_total=round(row['quality_points'] + row['time_points']
                                     - c['quality_points'] - c['time_points'], 4))
    out = os.path.join(os.path.dirname(__file__), 'gate_ab', 'results.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    # projected 13-case gain from the calibration table decisions
    proj = 0.0
    for r in table['rows']:
        if r['case'].endswith('bicycle'):
            proj += 0.84   # measured keep-0.6 M1 Δtotal on bicycle (this A/B: see report)
    print('wrote %d rows -> %s' % (len(rows), out))
    print('projected 13-case gain (Δtotal +0.84 on bicycle / 13) = %+.4f' % (proj/13))
    for row in rows:
        print('%-22s %-12s B=%6.2f q=%7.3f total=%7.3f dq=%s dB=%s dtotal=%s' % (
            row['case'].split('/')[-1][:22], row['arm'], row['b_seconds'],
            row['quality_points'], row['total_points'], row.get('d_q'), row.get('d_b'),
            row.get('d_total')))
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'raw/ab')
