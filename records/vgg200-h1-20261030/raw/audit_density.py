#!/usr/bin/env python3
"""Audit the native densification events of every VGG-matrix cell.

Proves the D3500 schedule claim (500->3500, interval 100, grad_abs_thresh 0.0008
=> 29 events) from each cell's own train_record.json, which the runner keeps.
"""
import json
import os
from pathlib import Path

ROOT = Path('/workspace/organizer-projects/FFSplat-Hackathon/outputs/vggab-d3500-20261030')
out = {}
for arm in ('ctrl', 'vgg100', 'vgg200'):
    for tag in sorted(os.listdir(ROOT / arm)) if (ROOT / arm).is_dir() else []:
        rec_p = ROOT / arm / tag / 'sweep' / 'train_record.json'
        if not rec_p.exists():
            continue
        d = json.loads(rec_p.read_text())
        ev = d.get('densification_events') or []
        keys = sorted(ev[0].keys()) if ev else []
        # merged trainer (>=c607ca6): each event carries before/after and
        # actual_native_parameters{mult,loss_thresh,grad_abs_thresh,grad_thresh,dense}
        native = {}
        for e in ev:
            anp = e.get('actual_native_parameters') or {}
            for k, v in anp.items():
                native.setdefault(k, v)
        bad = []
        if not ev:
            bad.append('no densification_events (cell did not use the merged trainer)')
        if native and abs(float(native.get('grad_abs_thresh', 0)) - 0.0008) > 1e-9:
            bad.append(f"grad_abs_thresh={native.get('grad_abs_thresh')} != 0.0008")
        out[f'{arm}/{tag}'] = {
            'events': len(ev),
            'first_event_iteration': ev[0].get('iteration') if ev else None,
            'last_event_iteration': ev[-1].get('iteration') if ev else None,
            'event_keys': keys,
            'native_params': native,
            'flags': bad,
            'stop_reason': d.get('stop_reason'),
            'num_gaussians': d.get('num_final_gaussians'),
            'train_seconds': d.get('train_seconds'),
        }
path = ROOT / 'density_audit.json'
path.write_text(json.dumps(out, indent=1))
n29 = sum(1 for v in out.values() if v['events'] == 29)
print(f'audited {len(out)} cells, {n29} with exactly 29 densification events')
flagged = {k: v['flags'] for k, v in out.items() if v['flags']}
print('FLAGGED:', json.dumps(flagged) if flagged else 'none')
for k, v in out.items():
    print(k, 'events=', v['events'], 'iters', v['first_event_iteration'], '->',
          v['last_event_iteration'], 'params', v['native_params'])
print('saved', path)
