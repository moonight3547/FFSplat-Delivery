#!/usr/bin/env python3
"""Build results.jsonl for the PGSR late-window matrix (paired vs in-session ctrl)."""
import glob, json, os, sys
BUDGET = {"mipnerf360/bicycle": 95.0, "mipnerf360/counter": 92.57,
          "dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce": 94.29,
          "dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e": 94.29}

def q(psnr, ssim, lpips):
    return (20*max(0,min(1,(psnr-15)/25)) + 20*max(0,min(1,(ssim-0.70)/0.25))
            + 40*max(0,min(1,(0.25-lpips)/0.20)))

def tp(b, budget):
    return 20*max(0,min(1,(1-b/budget)/0.9))

def main(root):
    ctrls, rows = {}, []
    for f in sorted(glob.glob(os.path.join(root, '*', '*', 'sweep', 'summary.json'))):
        arm = f.split(os.sep)[-3]
        d = json.load(open(f)); case = d['case']; budget = BUDGET.get(case, 94.29)
        rec = json.load(open(os.path.join(os.path.dirname(f), 'train_record.json')))
        res = d['results'][-1]; b = res['b_seconds_estimate']
        qq, tt = q(res['psnr'], res['ssim'], res['lpips']), tp(b, budget)
        dr = rec.get('depth_reg') or {}
        row = dict(case=case, arm=arm, num_gaussians=res['num_gaussians'],
                   b_seconds=b, psnr=res['psnr'], ssim=res['ssim'], lpips=res['lpips'],
                   quality_points=round(qq, 4), time_points=round(tt, 4),
                   total_points=round(qq + tt, 4), budget_seconds=budget,
                   stage1_seconds=rec.get('stage1_seconds'),
                   depth_reg=dict(weight=dr.get('weight'), every=dr.get('every'),
                                  from_iter=dr.get('from_iter'),
                                  until_iter=dr.get('until_iter'),
                                  n_applications=len(dr.get('events') or []),
                                  seconds_total=round(sum(e.get('seconds', 0)
                                                          for e in (dr.get('events') or [])), 3),
                                  first_event=(dr.get('events') or [None])[0],
                                  last_event=(dr.get('events') or [None])[-1]),
                   summary_json=f)
        if arm == 'ctrl': ctrls[case] = row
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.jsonl')
    with open(out, 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')
    print('wrote %d rows -> %s' % (len(rows), out))
    for row in rows:
        dr = row['depth_reg']
        print('%-24s %-10s N=%-8d B=%-6.2f q=%7.3f total=%7.3f dq=%s dB=%s dtotal=%s '
              'apps=%s extra_s=%s' % (
                  row['case'].split('/')[-1][:24], row['arm'], row['num_gaussians'],
                  row['b_seconds'], row['quality_points'], row['total_points'],
                  row.get('d_q'), row.get('d_b'), row.get('d_total'),
                  dr['n_applications'], dr['seconds_total']))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'raw/matrix')
