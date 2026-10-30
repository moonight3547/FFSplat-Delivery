#!/usr/bin/env python3
"""Next-generation 16-view selection variants for DL3DV (task-16 part B), CPU only.

Motivation: the organizer split is llffhold=8 (test = index ≡ 0 mod 8); context frames are
16 train frames uniform *by frame index*, which clusters spatially when the camera speed varies.
`arc-uniform` (uniform in cumulative camera-centre arc length) beat index-uniform on the task-14
held-out metric. This script builds next-gen variants on the same pool (train minus the fixed
8 held-out views) and writes their context sets + kernel stats for held-out triage.

Variants:
  arc-uniform        : 16 views at uniform cumulative camera-centre arc length (baseline).
  arc-posediv        : endpoints fixed; greedy max-min in a combined arc-length + rotation
                       distance  d = sqrt(Δs_norm^2 + (w*Δθ/π)^2), w=0.25
                       (equal spacing AND angular diversity).
  arc-cadence-mid    : GREY / protocol-aware. Only train frames whose sequence index ≡ 4 (mod 8)
                       (i.e. midway between test frames ≡0 mod 8) are candidates; pick 16 uniform
                       in arc length. Uses only the known split rule; no test RGB. Flagged.
  arc-cadence-adj    : GREY. Only train frames with index ≡ ±1 (mod 8) (adjacent to a test frame);
                       16 picked uniform in arc. Same flag.

Never reads test images. Output JSON holds context names, arc positions and stats.
"""
import argparse, json, os
import numpy as np

def parse_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 8 or f[1] != "PINHOLE":
            continue
        cams[int(f[0])] = dict(w=int(f[2]), h=int(f[3]), fx=float(f[4]), fy=float(f[5]))
    return cams

def q2R(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def parse_poses(path):
    out = {}
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("#"):
            i += 1
            continue
        f = line.split()
        if len(f) >= 10:
            try:
                qw, qx, qy, qz = map(float, f[1:5]); tx, ty, tz = map(float, f[5:8])
                out[f[9]] = (q2R((qw, qx, qy, qz)), np.array([tx, ty, tz]))
            except ValueError:
                pass
        i += 2
    return out

def arc_positions(centers):
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
    return L, (L / L[-1] if L[-1] > 0 else L)

def uniform_pick(t, n=16):
    targets = np.linspace(0.0, 1.0, n)
    sel = []
    for target in targets:
        for c in np.argsort(np.abs(t - target)):
            if int(c) not in sel:
                sel.append(int(c)); break
    return sel

def greedy_posediv(t, dirs, n=16, w=0.25):
    N = len(t)
    darc = np.abs(t[:, None] - t[None, :])
    ang = np.arccos(np.clip(dirs @ dirs.T, -1.0, 1.0)) / np.pi
    D = np.sqrt(darc ** 2 + (w * ang) ** 2)
    sel = [int(np.argmin(t)), int(np.argmax(t))]
    if sel[0] == sel[1]:
        sel = [0, N - 1]
    while len(sel) < n:
        dmin = D[:, sel].min(axis=1); dmin[sel] = -1
        sel.append(int(np.argmax(dmin)))
    return sel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--heldout-json", required=True, help="task-14 selection JSON (for the fixed heldout names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--variants", type=str, default="arc-uniform,arc-posediv,arc-cadence-mid,arc-cadence-adj")
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.work, "split.json")))
    images = list(split["images"])
    def as_names(lst):
        return [images[int(x)] if (isinstance(x, int) or (isinstance(x, str) and x.isdigit())) else x for x in lst]
    train = as_names(split["train"])
    heldout = json.load(open(args.heldout_json))["heldout_names"]
    held = set(heldout)
    cams = parse_cameras(os.path.join(args.work, "sparse/0/cameras.txt"))
    imag = parse_poses(os.path.join(args.work, "sparse/0/images.txt"))
    cam = cams[sorted(cams)[0]]
    train = [n for n in train if n in imag]
    centers = np.array([-imag[n][0].T @ imag[n][1] for n in train])
    # camera viewing direction in world space (camera looks along +Z convention -> use R^T·e_z)
    dirs = np.array([imag[n][0].T @ np.array([0.0, 0.0, 1.0]) for n in train])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    idx_order = np.arange(len(train))
    L, t = arc_positions(centers)

    pool = [i for i, n in enumerate(train) if n not in held]
    # For the held-out comparison the candidate sets must also avoid the held-out views:
    pool_set = set(pool)
    out = dict(scene=args.scene, work=args.work, n_train=len(train), n_context=args.n,
               heldout_names=heldout, resolution=[cam["h"], cam["w"]],
               note="candidate pools exclude the fixed 8 held-out train views (same set as task-14)",
               methods={})
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    def finish(name, pool_local_sel, pool_list, extra=None):
        names = [train[pool_list[int(k)]] for k in pool_local_sel]
        gl = [train.index(n) for n in names]
        st = dict(names=names,
                  train_local_indices=gl,
                  image_indices=[images.index(n) for n in names],
                  arc_positions=[round(float(t[g]), 5) for g in gl],
                  mean_arc_gap=float(np.mean(np.diff(sorted(t[g] for g in gl)))) if len(gl) > 1 else None,
                  rotation_span_deg=float(np.degrees(np.arccos(np.clip(np.min(dirs[gl] @ dirs[gl].T), -1, 1)))),
                  mean_pair_angle_deg=float(np.degrees(np.arccos(np.clip(
                      (dirs[gl] @ dirs[gl].T)[np.triu_indices(len(gl), 1)], -1, 1))).mean()) if len(gl) > 1 else None)
        if extra: st.update(extra)
        out["methods"][name] = st

    pool_list = pool
    sub_t = t[pool_list]
    sub_dirs = dirs[pool_list]

    if "arc-uniform" in want:
        finish("arc-uniform", uniform_pick(sub_t, args.n), pool_list)
    if "arc-posediv" in want:
        finish("arc-posediv", greedy_posediv(sub_t, sub_dirs, args.n), pool_list)
    if "arc-cadence-mid" in want:
        cand = [k for k, g in enumerate(pool_list) if (int(g) % 8) == 4]
        if len(cand) < args.n:
            cand = [k for k, g in enumerate(pool_list) if (int(g) % 8) in (3, 4, 5)]
        sub = cand
        sel = uniform_pick(t[[pool_list[k] for k in sub]], args.n)
        finish("arc-cadence-mid", [sub[s] for s in sel], pool_list,
               extra=dict(pool_rule="train index ≡ 4 (mod 8) — midway between test frames",
                          pool_size=len(sub), grey=True))
    if "arc-cadence-adj" in want:
        cand = [k for k, g in enumerate(pool_list) if (int(g) % 8) in (1, 7)]
        if len(cand) < args.n:
            cand = [k for k, g in enumerate(pool_list) if (int(g) % 8) in (1, 2, 6, 7)]
        sub = cand
        sel = uniform_pick(t[[pool_list[k] for k in sub]], args.n)
        finish("arc-cadence-adj", [sub[s] for s in sel], pool_list,
               extra=dict(pool_rule="train index ≡ ±1 (mod 8) — adjacent to a test frame",
                          pool_size=len(sub), grey=True))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    for m, st in out["methods"].items():
        print("%-18s arc_gap=%.4f rot_span=%.1f° mean_pair_angle=%.1f° %s" %
              (m, st.get("mean_arc_gap") or -1, st.get("rotation_span_deg") or -1,
               st.get("mean_pair_angle_deg") or -1, ("GREY" if st.get("grey") else "")))
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
