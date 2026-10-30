#!/usr/bin/env python3
"""Co-visibility 16-view selection for the ReSplat context set (CPU only).

Co-visibility is derived from the scene's ReSplat world model (the exported PLY):
  * sample N=30000 3D points (Gaussian centres) from the PLY;
  * project them into every TRAIN camera of the prepared work dir
    (visible = in front of camera and inside the image);
  * V[v, p] = 1 if point p is visible in train view v;
  * pairwise co-visibility J_ij = |V_i n V_j| / |V_i u V_j|  (Jaccard).

Selections returned (always 16 views, always taken from split["train"] only —
test images are never touched):
  uniform       : the existing np.linspace control (read from split.json)
  covis-maxmin  : greedy max-min Jaccard distance (start from the highest-J pair,
                  repeatedly add the view with the largest minimum distance 1-J
                  to the already-selected set; tie-break = larger visible count)
  covis-cover   : greedy marginal coverage (repeatedly add the view that covers the
                  most not-yet-covered points; tie-break = larger min Jaccard distance)
  covis-fps     : farthest-point sampling on camera centres restricted to the
                  top-60% co-visibility core (connected-component hybrid)

Writes one JSON per scene with indices into split["images"], image names, and stats.
Never uses test images or ground-truth renders.
"""
import argparse, json, os, struct, sys
import numpy as np

def parse_ply_header(f):
    props, n = [], None
    while True:
        line = f.readline().decode("ascii", "replace").strip()
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])
        elif line == "end_header":
            return props, n, f.tell()

def read_ply_xyz(path):
    with open(path, "rb") as f:
        props, n, off = parse_ply_header(f)
    with open(path, "rb") as f:
        f.seek(off)
        arr = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return xyz

def parse_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 8 or f[1] != "PINHOLE":
            continue
        cams[int(f[0])] = dict(w=int(f[2]), h=int(f[3]),
                               fx=float(f[4]), fy=float(f[5]), cx=float(f[6]), cy=float(f[7]))
    return cams

def q2R(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def parse_images(path):
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
                qw, qx, qy, qz = map(float, f[1:5])
                tx, ty, tz = map(float, f[5:8])
                cid = int(f[8]); name = f[9]
                out[name] = dict(R=q2R((qw, qx, qy, qz)), t=np.array([tx, ty, tz]), cid=cid)
            except ValueError:
                pass
        i += 2
    return out

def visibility(poses, cam, pts):
    V = np.zeros((len(poses), len(pts)), dtype=bool)
    for i, p in enumerate(poses):
        Xc = pts @ p["R"].T + p["t"]
        z = Xc[:, 2]
        u = cam["fx"] * Xc[:, 0] / np.where(np.abs(z) < 1e-9, 1e-9, z) + cam["cx"]
        v = cam["fy"] * Xc[:, 1] / np.where(np.abs(z) < 1e-9, 1e-9, z) + cam["cy"]
        V[i] = (z > 0) & (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    return V

def jaccard(V):
    W = V.astype(np.float32) @ V.astype(np.float32).T
    c = V.sum(axis=1).astype(np.float32)
    union = c[:, None] + c[None, :] - W
    J = np.where(union > 0, W / np.maximum(union, 1e-9), 0.0)
    return J, W, c

def sel_stats(idx, J, V):
    idx = list(map(int, idx))
    sel = np.array(idx)
    sub = J[np.ix_(sel, sel)]
    iu = np.triu_indices(len(sel), 1)
    union = V[sel].any(axis=0).sum()
    return dict(indices=idx,
                mean_pair_jaccard=float(sub[iu].mean()),
                median_pair_jaccard=float(np.median(sub[iu])),
                min_pair_jaccard=float(sub[iu].min()),
                union_visible_frac=float(union / max(1, int(V.any(axis=0).sum()))),
                mean_visible_points=float(V[sel].sum(axis=1).mean()))

def greedy_maxmin(J, n=16):
    N = J.shape[0]
    Jp = J.copy()
    np.fill_diagonal(Jp, -1.0)   # never pick a view twice (diagonal J=1)
    i, j = np.unravel_index(np.argmax(Jp), Jp.shape)
    if i == j:  # degenerate guard
        j = int(np.argmax(Jp[i]))
    sel = [int(i), int(j)]
    D = 1.0 - J
    while len(sel) < n:
        dmin = D[:, sel].min(axis=1)
        dmin[sel] = -1
        cand = int(np.argmax(dmin))
        sel.append(cand)
    return sel

def greedy_cover(V, J, n=16):
    N = V.shape[0]
    sel = [int(np.argmax(V.sum(axis=1)))]
    covered = V[sel[0]].copy()
    D = 1.0 - J
    while len(sel) < n:
        gain = (V & ~covered).sum(axis=1).astype(np.float64)
        gain[sel] = -1
        dmin = D[:, sel].min(axis=1)
        order = np.lexsort((-dmin, -gain))  # primary gain desc, secondary dmin desc
        cand = int(order[0])
        sel.append(cand)
        covered = covered | V[cand]
    return sel

def greedy_fps_core(J, centers, n=16, core_frac=0.6):
    N = J.shape[0]
    core_n = max(n, int(round(N * core_frac)))
    core = np.argsort(-J.mean(axis=1))[:core_n]
    sel = [int(core[0])]
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    Dc = D[np.ix_(core, core)]
    pos = {c: k for k, c in enumerate(core)}
    while len(sel) < n:
        dmin = np.array([Dc[pos[c], [pos[s] for s in sel]].min() for c in core])
        dmin[[pos[s] for s in sel]] = -1
        cand = int(core[int(np.argmax(dmin))])
        sel.append(cand)
    return sel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="prepared scene work dir (split.json + sparse/0)")
    ap.add_argument("--ply", required=True, help="scene world PLY to sample points from")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--points", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.work, "split.json")))
    images = list(split["images"])
    # v2 prepared work dirs store image NAMES (not indices) in train/test/context;
    # be tolerant: accept either names or integer indices.
    def as_names(lst):
        out = []
        for x in lst:
            out.append(images[int(x)] if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x)
        return out
    train_names = as_names(split["train"])
    train_pos = {nm: i for i, nm in enumerate(train_names)}
    cams = parse_cameras(os.path.join(args.work, "sparse/0/cameras.txt"))
    imag = parse_images(os.path.join(args.work, "sparse/0/images.txt"))
    if len(cams) != 1:
        raise SystemExit("expected a single PINHOLE camera, got %d" % len(cams))
    cam = cams[sorted(cams)[0]]

    names = train_names
    miss = [nm for nm in names if nm not in imag]
    if miss:
        raise SystemExit("missing poses for %d train images, e.g. %s" % (len(miss), miss[:3]))
    poses = [imag[nm] for nm in names]

    xyz = read_ply_xyz(args.ply)
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(xyz), size=min(args.points, len(xyz)), replace=False)
    pts = xyz[take]

    V = visibility(poses, cam, pts)
    J, W, cnt = jaccard(V)

    out = dict(scene=args.scene, work=args.work, ply=args.ply,
               n_train=len(names), n_test=len(split["test"]), n_points=int(len(pts)),
               split_context_schema="names",
               resolution=[cam["h"], cam["w"]], seed=args.seed,
               covis_method="PLY-point projection; Jaccard of per-train-view visible point sets",
               methods={})

    uniform_names = as_names(split["context"])
    uniform_local = [train_pos[nm] for nm in uniform_names]
    out["methods"]["uniform"] = sel_stats(uniform_local, J, V)
    out["methods"]["uniform"]["local_indices"] = uniform_local
    out["methods"]["uniform"]["names"] = uniform_names
    out["methods"]["uniform"]["indices"] = [images.index(nm) for nm in uniform_names]

    centers = np.array([-p["R"].T @ p["t"] for p in poses])
    for meth, sel in (("covis-maxmin", greedy_maxmin(J, args.n)),
                      ("covis-cover", greedy_cover(V, J, args.n)),
                      ("covis-fps", greedy_fps_core(J, centers, args.n))):
        st = sel_stats(sel, J, V)
        st["local_indices"] = [int(x) for x in sel]
        st["names"] = [names[int(x)] for x in sel]
        st["indices"] = [images.index(nm) for nm in st["names"]]
        out["methods"][meth] = st

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({m: {k: out["methods"][m][k] for k in
                           ("mean_pair_jaccard", "median_pair_jaccard", "min_pair_jaccard",
                            "union_visible_frac", "mean_visible_points", "local_indices")}
                      for m in out["methods"]}, indent=1))
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
