#!/usr/bin/env python3
"""VGG200 e2e on H1: delivery recipe + 200 extra VGG-LPIPS steps on the DL3DV
bad cases (LPIPS just below the 0.25 scoring cliff) and mipnerf360/counter.

Arms (uniform, no per-scene hyperparameters):
  ctrl    : the exact delivery recipe used by clean13 / run_exact13_gate.py
  vgg200  : ctrl + --stage3-iters 200 --stage3-lpips-weight 0.1 (stage2 = 0)
  vgg500w005 (optional, only if vgg200 is positive): 500 steps, weight 0.05

Each case runs ctrl then vgg200 back-to-back (paired Δq/ΔB).  Strictly serial and
gated on a quiet GPU (nvidia-smi memory + foreign compute processes), because the
module-B wall clock is the scored quantity.

Usage (H1, repo root, fastgs conda env):
    python outputs/vgg200-20261030/run_vgg200.py [--only tag1,tag2] [--arms ctrl,vgg200]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/workspace/organizer-projects/FFSplat-Hackathon')
PROFILE = REPO / 'competition/profiles/compshare-20261029.json'
OUT = REPO / 'outputs/vgg200-20261030'
RESULTS = OUT / 'results.jsonl'

CASES = [
    # tag, case, work dir, budget, role, clean13_total (records/clean13-20261030/results.jsonl)
    ('dl3dv-0b5635d5',
     'dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce',
     'outputs/multi/dl3dv-0b5635d5/work', 94.29, 'primary', 31.10574259810206),
    ('dl3dv-0f622eaa',
     'dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e',
     'outputs/multi/dl3dv-0f622eaa/work', 94.29, 'primary', 31.552958455583433),
    ('dl3dv-3e4dbe41',
     'dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3',
     'outputs/multi/dl3dv-3e4dbe41/work', 94.29, 'primary', 35.03899809710881),
    ('mipnerf360-counter', 'mipnerf360/counter',
     'outputs/cand/mipnerf360-counter/work', 92.57, 'secondary', 55.17158984097854),
    # negative control: LPIPS 0.3030 > 0.25 cliff, so the LPIPS term already scores
    # 0 -> the same VGG-200 arm must NOT help here (makes the cliff argument airtight)
    ('mipnerf360-bicycle', 'mipnerf360/bicycle',
     'outputs/cand/mipnerf360-bicycle/work', 95.0, 'negative-control', 25.31011456233717),
]

ARMS = {
    'ctrl': [],
    'vgg200': ['--stage2-iters', '0', '--stage3-iters', '200',
               '--stage3-lpips-weight', '0.1'],
    'vgg500w005': ['--stage2-iters', '0', '--stage3-iters', '500',
                   '--stage3-lpips-weight', '0.05'],
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def conda_cmd(profile, runtime, *argv):
    return [profile['conda'], 'run', '--no-capture-output', '-n',
            profile['environments'][runtime], 'python', *map(str, argv)]


def loadavg1():
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return 0.0


def gpu_busy():
    # the exact-13 contamination lesson: a concurrent CPU-heavy process inflated B
    # by ~8 % while the GPU itself looked idle, so gate on load average as well
    la = loadavg1()
    if la >= 4.0:
        return True
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        used = [int(v) for v in out.split() if v.strip()]
    except Exception:
        return True
    if any(u > 1200 for u in used):
        return True
    ps = subprocess.run(['pgrep', '-af',
                         'resplat_variant|research.run_sweep|train_sweep|'
                         'render_eval|prune_gate|prune_gate|view_select|research.select_views'],
                        capture_output=True, text=True)
    foreign = [l for l in ps.stdout.splitlines()
               if 'run_vgg200' not in l and 'pgrep' not in l]
    return bool(foreign)


def wait_quiet(max_wait, log):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > max_wait:
            raise SystemExit('QUIET-WAIT TIMED OUT: GPU still busy')
        time.sleep(15)
    log.write('[quiet] GPU idle\n')
    log.flush()


def delivery_argv(case, work, init, out_dir, budget, arm):
    argv = ['-m', 'research.run_sweep', '--profile', str(PROFILE),
            '--case', case, '--work', work, '--init-ply', init,
            '--out', str(out_dir), '--budget', str(budget),
            '--iterations', '5000', '--position-lr-max-steps', '5000',
            '--opacity-reset-interval', '0', '--lambda-dssim', '0.6',
            '--scaling-lr', '0.01', '--densify-from', '500',
            '--densify-until', '1250', '--realloc-frac', '0.15',
            '--realloc-at', '1250', '--image-cache', f'{work}/image_cache',
            '--snapshots', '', '--seed', '20261015']
    argv += ARMS[arm]
    return argv


def lpips_points(lpips):
    return 40.0 * max(0.0, min(1.0, (0.25 - lpips) / 0.20))


def run_one(profile, tag, case, work, budget, arm, init):
    cell = OUT / arm / tag
    log_path = OUT / 'logs' / f'{tag}_{arm}.log'
    cell.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, 'a', encoding='utf-8')
    try:
        wait_quiet(3600, log)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(REPO)
        env.update(read(PROFILE).get('environment', {}))
        argv = conda_cmd(profile, 'fastgs', *delivery_argv(
            case, work, init, cell / 'sweep', budget, arm))
        log.write('+ ' + ' '.join(argv) + '\n')
        log.flush()
        log.write(f'[loadavg_at_start] {loadavg1():.2f}\n')
        log.flush()
        t0 = time.perf_counter()
        subprocess.run(argv, cwd=REPO, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - t0
        summary = read(cell / 'sweep' / 'summary.json')
        row = summary['results'][-1]
        sc = row['score']
        rec = summary.get('train_record', {})
        return dict(tag=tag, case=case, arm=arm, budget=budget,
                    wall_seconds=round(wall, 2),
                    loadavg_at_start=round(loadavg1(), 2),
                    psnr=row['psnr'], ssim=row['ssim'], lpips=row['lpips'],
                    psnr_points=sc['psnr_points'], ssim_points=sc['ssim_points'],
                    lpips_points=lpips_points(row['lpips']),
                    quality_points=sc['quality_points'],
                    time_points=sc['time_points'], total_points=sc['total_points'],
                    b_seconds=row['b_seconds_estimate'],
                    train_seconds=row.get('train_seconds'),
                    stage3_seconds=rec.get('stage3_seconds'),
                    stage3_iterations=rec.get('stage3_iterations'),
                    stage2_iterations=rec.get('stage2_iterations'),
                    stage3_lpips_weight=rec.get('stage3_lpips_weight'),
                    num_gaussians=row.get('num_gaussians'))
    finally:
        log.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='comma-separated tags')
    ap.add_argument('--arms', default='ctrl,vgg200')
    args = ap.parse_args(argv)
    profile = read(PROFILE)
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(',')) if args.only else None
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f'unknown arm {a}')

    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r['tag'], r['arm']))
    for tag, case, work, budget, role, clean_total in CASES:
        if only and tag not in only:
            continue
        init = f'outputs/clean13-20261030/resplat/{tag}/resplat_world.ply'
        if not (REPO / init).exists():
            print(f'[skip] {tag}: init PLY missing')
            continue
        if not (REPO / work / 'image_cache/manifest.json').exists():
            print(f'[skip] {tag}: image cache missing under {work}')
            continue
        for arm in arms:
            if (tag, arm) in done:
                print(f'[skip] {tag}/{arm} already in results.jsonl')
                continue
            print(f'[run] {tag}/{arm}  ({role})', flush=True)
            rec = run_one(profile, tag, case, work, budget, arm, init)
            rec['role'] = role
            rec['clean13_total'] = clean_total
            rec['delta_vs_clean13'] = rec['total_points'] - clean_total
            with RESULTS.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            print(f"[done] {tag}/{arm}: psnr={rec['psnr']:.3f} lpips={rec['lpips']:.4f} "
                  f"q={rec['quality_points']:.2f} B={rec['b_seconds']:.2f} "
                  f"stage3={rec['stage3_seconds']}s total={rec['total_points']:.2f}",
                  flush=True)
    # paired deltas
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    by = {(r['tag'], r['arm']): r for r in rows}
    pairs = []
    for tag, _c, _w, _b, _ro, _cl in CASES:
        if (tag, 'ctrl') in by and (tag, 'vgg200') in by:
            c, v = by[(tag, 'ctrl')], by[(tag, 'vgg200')]
            pairs.append(dict(
                tag=tag,
                lpips_before=c['lpips'], lpips_after=v['lpips'],
                delta_lpips=v['lpips'] - c['lpips'],
                lpips_points_before=c['lpips_points'],
                lpips_points_after=v['lpips_points'],
                delta_lpips_points=v['lpips_points'] - c['lpips_points'],
                delta_psnr=v['psnr'] - c['psnr'],
                delta_ssim=v['ssim'] - c['ssim'],
                delta_quality=v['quality_points'] - c['quality_points'],
                delta_b=v['b_seconds'] - c['b_seconds'],
                delta_time=v['time_points'] - c['time_points'],
                delta_total=v['total_points'] - c['total_points'],
                stage3_seconds=v['stage3_seconds']))
    if pairs:
        (OUT / 'pairs.json').write_text(json.dumps(pairs, indent=2))
        for p in pairs:
            print('[pair] ' + json.dumps(p), flush=True)
    print(f'[results] {RESULTS}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
