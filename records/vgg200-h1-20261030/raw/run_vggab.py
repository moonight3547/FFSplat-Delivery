#!/usr/bin/env python3
"""VGG100 vs VGG200 paired e2e on the 5 bad cases, on top of the *merged
BigD2500* delivery recipe.

The base recipe is NOT hard-coded: pass it as a JSON argv list so the exact
final recipe / commit can be plugged in unchanged:

    python outputs/vggab-20261030/run_vggab.py --recipe recipe.json

recipe.json = {"recipe_name": "...", "commit": "<git sha>",
               "base_flags": ["--iterations", "5000", ...]}

Arms (uniform for every case):
  ctrl   : base recipe
  vgg100 : base + --stage2-iters 0 --stage3-iters 100 --stage3-lpips-weight 0.1
  vgg200 : base + --stage2-iters 0 --stage3-iters 200 --stage3-lpips-weight 0.1

One shared ctrl per case, then vgg100, then vgg200 (paired Δ).  Serial, quiet box
(loadavg + nvidia-smi + foreign-process gate), official metrics.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/workspace/organizer-projects/FFSplat-Hackathon')
PROFILE = REPO / 'competition/profiles/compshare-20261029.json'
_DEFAULT_OUT = REPO / 'outputs/vggab-20261030'
OUT = _DEFAULT_OUT
RESULTS = OUT / 'results.jsonl'

CASES = [
    ('dl3dv-0b5635d5',
     'dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce',
     'outputs/multi/dl3dv-0b5635d5/work', 94.29, 31.10574259810206),
    ('dl3dv-0f622eaa',
     'dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e',
     'outputs/multi/dl3dv-0f622eaa/work', 94.29, 31.552958455583433),
    ('dl3dv-3e4dbe41',
     'dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3',
     'outputs/multi/dl3dv-3e4dbe41/work', 94.29, 35.03899809710881),
    ('mipnerf360-counter', 'mipnerf360/counter',
     'outputs/cand/mipnerf360-counter/work', 92.57, 55.17158984097854),
    ('mipnerf360-bicycle', 'mipnerf360/bicycle',
     'outputs/cand/mipnerf360-bicycle/work', 95.0, 25.31011456233717),
]
# Default arms reflect the CONFIRMED Codex semantics (Lead 2026-10-31): the VGG
# phase is INSIDE the same 5000 steps, full-res VGG weight 0.1, active in the last
# N steps (100 -> 4901-5000, 200 -> 4801-5000), NOT a post-5000 continuation.
# The exact flag spelling comes with the merged commit; a recipe may override the
# whole arm table via {"arms": {"ctrl": [...], "vgg100": [...], "vgg200": [...]}}.
ARMS = {
    'ctrl': [],
    'vgg100': ['--lpips-start', '4900', '--lpips-end', '5000', '--lpips-weight', '0.1'],
    'vgg200': ['--lpips-start', '4800', '--lpips-end', '5000', '--lpips-weight', '0.1'],
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def loadavg1():
    """1-min load average **inside this container**.

    IMPORTANT: ``os.getloadavg()`` on this compshare pod returns the *physical
    host's* load (13-21 from other tenants, via sysinfo) while ``/proc/loadavg``
    is lxcfs-virtualised to this container (0.0-0.2 when idle).  Using
    os.getloadavg() here would block the quiet gate forever, so read /proc.
    """
    try:
        with open('/proc/loadavg') as f:
            return float(f.read().split()[0])
    except (OSError, ValueError):
        return 0.0


def gpu_busy():
    if loadavg1() >= 4.0:
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
                         'resplat_variant|research.run_sweep|train_sweep|render_eval|'
                         'prune_gate|view_select|research.select_views'],
                        capture_output=True, text=True)
    # exclude our own/ssh/监视 wrappers: real stages are `python -m research.*`
    skip = ('run_vggab', 'pgrep', 'probe_busy', 'bash -c', 'ssh ', 'ssh:')
    foreign = [l for l in ps.stdout.splitlines()
               if not any(tok in l for tok in skip)]
    return bool(foreign)


def wait_quiet(max_wait, log):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > max_wait:
            raise SystemExit('QUIET-WAIT TIMED OUT: box not quiet')
        time.sleep(15)
    log.write(f'[quiet] loadavg={loadavg1():.2f}\n')
    log.flush()


def free_gb(path='/'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def cleanup_heavy(cell, log):
    """Delete the multi-hundred-MB intermediates of one finished run.
    Everything needed for the report (summary.json, train_record.json,
    eval/*/metrics.json) is kept; only PLYs / model dirs / rendered PNGs go."""
    freed = 0.0
    targets = []
    for sub in ('model', 'snaps'):
        targets.append(cell / 'sweep' / sub)
    for p in (cell / 'sweep' / 'eval').rglob('*.png'):
        targets.append(p)
    for p in (cell / 'sweep' / 'eval').rglob('*.jpg'):
        targets.append(p)
    for t in targets:
        if t.is_dir():
            freed += sum(f.stat().st_size for f in t.rglob('*') if f.is_file()) / 1e9
            subprocess.run(['rm', '-rf', str(t)])
        elif t.is_file():
            freed += t.stat().st_size / 1e9
            t.unlink()
    log.write(f'[cleanup] freed ~{freed:.2f} GB, free now {free_gb():.2f} GB\n')
    log.flush()
    return freed


def lpips_points(lp):
    return 40.0 * max(0.0, min(1.0, (0.25 - lp) / 0.20))


def run_one(profile, recipe, tag, case, work, budget, arm, init, arms_table):
    cell = OUT / arm / tag
    cell.mkdir(parents=True, exist_ok=True)
    lp = OUT / 'logs' / f'{tag}_{arm}.log'
    lp.parent.mkdir(parents=True, exist_ok=True)
    log = open(lp, 'a', encoding='utf-8')
    try:
        if free_gb() < 2.0:
            log.write(f'[disk] only {free_gb():.2f} GB free before run; aborting\n')
            log.flush()
            raise SystemExit('DISK LOW: <2 GB free, refusing to start a run')
        wait_quiet(3600, log)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(REPO)
        env.update(read(PROFILE).get('environment', {}))
        argv = [read(PROFILE)['conda'], 'run', '--no-capture-output', '-n',
                read(PROFILE)['environments']['fastgs'], 'python',
                '-m', 'research.run_sweep', '--profile', str(PROFILE),
                '--case', case, '--work', work, '--init-ply', init,
                '--out', str(cell / 'sweep'), '--budget', str(budget),
                '--image-cache', f'{work}/image_cache', '--snapshots', '',
                '--seed', '20261015']
        argv += [str(x) for x in recipe['base_flags']]
        argv += [str(x) for x in arms_table[arm]]
        log.write('[recipe] ' + json.dumps(recipe) + '\n')
        log.write(f'[loadavg_at_start] {loadavg1():.2f}\n')
        log.write('+ ' + ' '.join(argv) + '\n')
        log.flush()
        t0 = time.perf_counter()
        subprocess.run(argv, cwd=REPO, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - t0
        summary = read(cell / 'sweep' / 'summary.json')
        row = summary['results'][-1]
        sc, rec = row['score'], summary.get('train_record', {})
        freed = cleanup_heavy(cell, log)
        return dict(tag=tag, case=case, arm=arm, budget=budget,
                    freed_gb=round(freed, 2),
                    recipe_name=recipe.get('recipe_name'),
                    recipe_commit=recipe.get('commit'),
                    recipe_sha256=hashlib.sha256(
                        json.dumps(recipe['base_flags'], sort_keys=True).encode()
                    ).hexdigest()[:16],
                    wall_seconds=round(wall, 2),
                    loadavg_at_start=round(loadavg1(), 2),
                    psnr=row['psnr'], ssim=row['ssim'], lpips=row['lpips'],
                    lpips_points=lpips_points(row['lpips']),
                    quality_points=sc['quality_points'],
                    time_points=sc['time_points'], total_points=sc['total_points'],
                    b_seconds=row['b_seconds_estimate'],
                    train_seconds=row.get('train_seconds'),
                    stage3_seconds=rec.get('stage3_seconds'),
                    stage3_iterations=rec.get('stage3_iterations'),
                    arm_flags=[str(x) for x in arms_table[arm]],
                    num_gaussians=row.get('num_gaussians'),
                    density_audit={k: v for k, v in rec.items()
                                   if any(t in k.lower() for t in
                                          ('dens', 'event', 'grad_abs', 'mult',
                                           'loss_thresh', 'op_', 'opt_'))
                                   and isinstance(v, (int, float, str, bool))},
                    densify_until=next((str(recipe['base_flags'][i + 1])
                                        for i, x in enumerate(recipe['base_flags'])
                                        if x == '--densify-until'), None))
    finally:
        log.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--out-root', default=None,
                    help='experiment output root (default outputs/vggab-20261030)')
    ap.add_argument('--arms', default='ctrl,vgg100,vgg200')
    ap.add_argument('--only', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    global OUT, RESULTS
    if args.out_root:
        p = Path(args.out_root)
        OUT = p if p.is_absolute() else (REPO / p)
        RESULTS = OUT / 'results.jsonl'
    profile = read(PROFILE)
    recipe = read(args.recipe)
    arms_table = dict(ARMS)
    arms_table.update({k: v for k, v in (recipe.get('arms') or {}).items()})
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    only = set(args.only.split(',')) if args.only else None
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r['tag'], r['arm']))
    for tag, case, work, budget, clean in CASES:
        if only and tag not in only:
            continue
        init = f'outputs/clean13-20261030/resplat/{tag}/resplat_world.ply'
        if not (REPO / init).exists() or not (REPO / work).is_dir():
            print(f'[skip] {tag}: missing init/work'); continue
        for arm in arms:
            if (tag, arm) in done:
                print(f'[skip] {tag}/{arm} done'); continue
            if args.dry_run:
                argv = [read(PROFILE)['conda'], 'run', '--no-capture-output', '-n',
                        read(PROFILE)['environments']['fastgs'], 'python',
                        '-m', 'research.run_sweep', '--profile', str(PROFILE),
                        '--case', case, '--work', work, '--init-ply', init,
                        '--out', str(OUT / arm / tag / 'sweep'),
                        '--budget', str(budget),
                        '--image-cache', f'{work}/image_cache', '--snapshots', '',
                        '--seed', '20261015']
                argv += [str(x) for x in recipe['base_flags']]
                argv += [str(x) for x in arms_table[arm]]
                print('[dry-run] ' + ' '.join(argv))
                return 0
            print(f'[run] {tag}/{arm}', flush=True)
            rec = run_one(profile, recipe, tag, case, work, budget, arm, init,
                          arms_table)
            rec['clean13_total'] = clean
            rec['delta_vs_clean13'] = rec['total_points'] - clean
            with RESULTS.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            print(f"[done] {tag}/{arm}: psnr={rec['psnr']:.3f} lpips={rec['lpips']:.4f} "
                  f"q={rec['quality_points']:.2f} B={rec['b_seconds']:.2f} "
                  f"stage3={rec['stage3_seconds']} total={rec['total_points']:.2f}",
                  flush=True)
    # paired deltas (ctrl as the shared in-session control)
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    by = {(r['tag'], r['arm']): r for r in rows}
    pairs = []
    for tag, _c, _w, _b, _cl in CASES:
        c = by.get((tag, 'ctrl'))
        if not c:
            continue
        for arm in ('vgg100', 'vgg200'):
            v = by.get((tag, arm))
            if not v:
                continue
            pairs.append(dict(
                tag=tag, arm=arm,
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
