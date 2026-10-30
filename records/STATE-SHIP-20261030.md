# STATE-SHIP-20261030 — 13-scene eval readiness + ReSplat capability backup

Owner: h1-prep (shared task-1). Written 2026-10-30 (Asia/Shanghai).
Scope: unblock the N1 steps-Pareto sweep on all 13 competition scenes, and make the
H1-only ReSplat capability restorable after the pod dies.

## TL;DR

- **N1 now has all 13 prepared scenes**, 6 of them shipped H1→N1 today.
- `records/evalnode-manifest-13.json` on N1 is **validated 13/13** (work dir +
  split.json + init PLY all resolve). Identical copy versioned at
  `FFSplat-Hackathon/records/evalnode-manifest-13.json` on the Mac.
- **`<work>/image_cache` built for all 13 scenes** (`images_u8.npy` + `manifest.json`),
  so the sweep can pass `--image-cache <work>/image_cache`.
- **ReSplat capability backup** staged on H1 `/workspace/ffsplat-backup/` and copied to
  N1 `/root/gpufree-data/backup/`: conda env tarball, source trees + git bundles +
  worktree patches + patched-file list, and the checkpoint (`sha256 a72dc6d0…`).
- N1 root disk untouched (still 312M free, 99%): everything big lives under
  `/root/gpufree-data` (136G free after this work).

## 1. 13-scene inventory (what the manifest points at)

All paths are relative to `/root/projects/FFSplat-Hackathon` (= N1 repo root; `outputs`
is a symlink to `/root/gpufree-data/outputs`).

| # | case | work | init (ReSplat PLY) | budget_s | image_cache |
|---|------|------|--------------------|----------|-------------|
| 1 | hw/billboard | `outputs/O0-20261030/work/hw/billboard` | `outputs/O0-20261030/results/hw/billboard/resplat/resplat_world.ply` | 99.49 | 73M |
| 2 | mipnerf360/counter | `outputs/O0-20261030/work/mipnerf360/counter` | `outputs/O0-20261030/results/mipnerf360/counter/resplat/resplat_world.ply` | 92.57 | 380M |
| 3 | dl3dv/001dccbc…740d5f | `outputs/prep/dl3dv/001dccbc…740d5f/work` | `outputs/prep/dl3dv/001dccbc…740d5f/resplat/resplat_world.ply` | 94.29 | 465M |
| 4 | mipnerf360/bicycle | `outputs/prep/mipnerf360/bicycle/work` | `outputs/prep/mipnerf360/bicycle/resplat/resplat_world.ply` | 95.0 | 307M |
| 5 | dl3dv/0b5635d5…64cfce | `outputs/curve/dl3dv-0b5635d5…64cfce/work` | `outputs/curve/dl3dv-0b5635d5…64cfce/resplat/resplat_world.ply` | 94.29 | 536M |
| 6 | dl3dv/3e4dbe41…0cdae3 | `outputs/curve/dl3dv-3e4dbe41…0cdae3/work` | `outputs/curve/dl3dv-3e4dbe41…0cdae3/resplat/resplat_world.ply` | 94.29 | 432M |
| 7 | dl3dv/0f622eaa…6a2e | `outputs/curve/dl3dv-0f622eaa…6a2e/work` | `outputs/curve/dl3dv-0f622eaa…6a2e/resplat/resplat_world.ply` | 94.29 | 456M |
| 8 | dl3dv/2dad2d29…d38b84 | `outputs/inputs13/dl3dv-2dad2d29/work` | `outputs/inputs13/dl3dv-2dad2d29/resplat/resplat_world.ply` | 94.29 | 429M |
| 9 | dl3dv/9a451604…ece446 | `outputs/inputs13/dl3dv-9a451604/work` | `outputs/inputs13/dl3dv-9a451604/resplat/resplat_world.ply` | 94.29 | 490M |
| 10 | hw/basketball | `outputs/inputs13/hw-basketball/work` | `outputs/inputs13/hw-basketball/resplat/resplat_world.ply` | 95.0 | 89M |
| 11 | hw/car | `outputs/inputs13/hw-car/work` | `outputs/inputs13/hw-car/resplat/resplat_world.ply` | 95.0 | 89M |
| 12 | hw/trashbin | `outputs/inputs13/hw-trashbin/work` | `outputs/inputs13/hw-trashbin/resplat/resplat_world.ply` | 95.0 | 95M |
| 13 | mipnerf360/bonsai | `outputs/inputs13/mipnerf360-bonsai/work` | `outputs/inputs13/mipnerf360-bonsai/resplat/resplat_world.ply` | 95.0 | 462M |

Budgets: dl3dv/* = 94.29, hw/billboard = 99.49, mipnerf360/counter = 92.57,
hw/basketball|car|trashbin = 95.0, mipnerf360/bicycle|bonsai = 95.0.

`outputs/inputs13` is a symlink on N1:
`/root/projects/FFSplat-Hackathon/outputs/inputs13 -> /root/gpufree-data/inputs13`.
Real bytes live in `/root/gpufree-data/inputs13/<tag>/` as instructed.

## 2. H1→N1 shipment (6 missing scenes)

Source on H1 `/workspace/organizer-projects/FFSplat-Hackathon`:

| tag | H1 source dir | work files | images | ply size |
|-----|---------------|-----------|--------|----------|
| dl3dv-2dad2d29 | `outputs/multi/dl3dv-2dad2d29` | 309 | 305 | 108M |
| dl3dv-9a451604 | `outputs/multi/dl3dv-9a451604` | 352 | 348 | 106M |
| hw-basketball | `outputs/cand/hw-basketball` | 83 | 79 | 92M |
| hw-car | `outputs/cand/hw-car` | 74 | 70 | 99M |
| hw-trashbin | `outputs/cand/hw-trashbin` | 79 | 75 | 99M |
| mipnerf360-bonsai | `outputs/cand/mipnerf360-bonsai` | 296 | 292 | 122M |

Method: `tar -C <h1-scene> -cf - work resplat/resplat_world.ply | ssh ton1 … tar -xf -`
(H1→N1 direct ssh), launched with `setsid nohup` on H1 so a dropped H1 handshake does
not kill the transfer. Only `work/` and `resplat/resplat_world.ply` were copied — not the
multi-hundred-MB `.npz` next to the PLY.
Result on N1 `/root/gpufree-data/inputs13/`: all 6 dirs verified `split.json`,
`images/`, `sparse/` (2 files) and the PLY; per-tag sizes 228M–819M
(819M dl3dv-2dad2d29, 805M dl3dv-9a451604, 228M hw-basketball, 240M hw-car,
230M hw-trashbin, 813M mipnerf360-bonsai). H1 source file counts match N1 exactly
(e.g. bonsai 296/296, dl3dv-9a451604 352/352).

## 3. Manifest

- N1: `/root/projects/FFSplat-Hackathon/records/evalnode-manifest-13.json`
- Mac (versioned): `FFSplat-Hackathon/records/evalnode-manifest-13.json`
- sha256 (both, identical): `c2aac742d859f1e7c82b592c04b2788a760eeefdc37a132ad9ebf5435d9c2667`
- Format: 13 × `{case, work, init, budget_seconds}`; verified 13/13 resolve
  (`work/` dir, `work/split.json`, `init` PLY all exist).

**Repath note (important):** the pre-existing 3-scene manifest pointed
`dl3dv/001dccbc…` at `outputs/O0-20261030/work/dl3dv/001dccbc…`, which contains only
`images/` + `sparse/` — **no `split.json`/`module_a.json`** (its image_cache build
fails, rc=1). The manifest now points that case at the complete prepared copy
`outputs/prep/dl3dv/001dccbc…/{work,resplat/resplat_world.ply}` instead. The other 3
default scenes (billboard, counter) are complete in `O0-20261030` and were left as-is.

### Same-run provenance of the repathed trees (requested check)

`work/` and `resplat/resplat_world.ply` are **from the same module-A run** for each
repathed tree — they are siblings in one `prep|curve/<scene>/{work,resplat}` tree, the
PLY is written after the work files in the same run, and the image counts agree:

| tree | work/split.json + module_a.json mtime | resplat PLY mtime | module_a num_images | work/images |
|---|---|---|---|---|
| prep/dl3dv/001dccbc… | 2026-10-30 07:13:43 | 07:17:41 (+3m58s) | 330 | 330 |
| prep/mipnerf360/bicycle | 06:14:07 | 06:18:26 (+4m19s) | — | 194 |
| curve/dl3dv/0b5635… | 10:55:13 | 10:55:43 (+30s) | — | 381 |
| curve/dl3dv/0f622eaa… | 10:58:08 | 10:58:39 (+31s) | — | 324 |
| curve/dl3dv/3e4dbe… | 10:59:19 | 10:59:53 (+34s) | — | 307 |

Caveat stated for the record: `module_a.json` records the *organizer* `scene_dir` as an
H1 path (`/workspace/organizer-assets/datasets/DL3DV/001dccbc…`), i.e. this prep tree
was produced on H1 and then landed on N1 as a whole tree with mtimes preserved; the
work sibling and the PLY were never re-mixed. (The work files' own mtimes are
07:13:43; the `prep/<scene>/work` dir mtime is 22:59 only because the image_cache was
added there today.) So DL3DV module-A is deterministic and the PLY was inferred from
exactly these 330 images — no cross-run pairing. Nothing to flag.

## 4. image_cache (all 13 scenes)

Built on N1 with the fastgs env (`/opt/conda/envs/fastgs/bin/python`), module
`research.image_cache`, out dir `<work>/image_cache`:
`images_u8.npy` (uint8, `[N, H, W, 3]`) + `manifest.json`.

Verified per scene: `npy.N == len(work/images)`, dtype `uint8`, e.g. billboard
`(64, 768, 512, 3)`, counter `(240, 576, 960, 3)`, dl3dv-9a451604
`(348, 512, 960, 3)`, basketball `(79, 768, 512, 3)`, bonsai `(292, 576, 960, 3)`.
Cache sizes per scene in the table above; total ≈ 4.3G under `/root/gpufree-data`.
Cache build log: `/root/gpufree-data/image_cache_build.log` (all rc=0).

Handoff to `n1-sweep`: pass `--image-cache <work>/image_cache`; do not rebuild.

## 5. ReSplat capability backup (H1 is the only node with the env)

H1 has no big data mount at `/root/gpufree-data` and the root overlay has 132G free, so
staging is under `/workspace/ffsplat-backup/`. Backup payload:

| artifact | H1 path | size (bytes) | sha256 |
|---|---|---|---|
| conda env `ffsplat-enhance` | `/workspace/ffsplat-backup/ffsplat-enhance-env.tgz` | 7,252,564,215 (6.8 GiB) | **`cce97ba5c541892bdfe952d0ec29f4dfd51f16218df7223f0f18502bce987916`** |
| ReSplat/source trees (`FFSplat-Enhance` + `FFSplat-Eval`) | `/workspace/ffsplat-backup/ffsplat-src-FFSplat-Enhance-FFSplat-Eval.tgz` | 244,948,749 | `83e7860c553ff4fb2d723af534c7521f37098190649ef86ad2ef2ce9c2ad1623` |
| git bundles | `FFSplat-Enhance.bundle` (3,299,890 B), `FFSplat-Eval.bundle` (1,636,696 B) | 4.9M | in `source-and-checkpoint-artifacts.tgz` |
| worktree patches | `FastGS-worktree.patch` (2.1K), `gaussian-splatting-worktree.patch` (0), `resplat-worktree.patch` (0) | — | idem |
| patched-file list | `patched-files.txt` | 1,586 B | idem |
| env package lists | `env-conda-list.txt` (25,780 B), `env-pip-freeze.txt` (5,918 B) | — | idem |
| ReSplat checkpoint | `/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth` | 896,300,265 | **`a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c`** |

Second-node copies (N1 `/root/gpufree-data/backup/`, all sha256-verified on N1):

| file | size (bytes) | sha256 |
|---|---|---|
| `ffsplat-enhance-env.tgz` | 7,252,564,215 | `cce97ba5c541892bdfe952d0ec29f4dfd51f16218df7223f0f18502bce987916` (`sha256sum -c` → OK, `verify.log`) |
| `ffsplat-enhance-env.tgz.sha256` | — | rewritten to reference the N1 filename |
| `source-and-checkpoint-artifacts.tgz` (bundles, patches, patched-files list, package lists, source tarball, checkpoint sha file) | 249,333,543 | `7b37128c38ef85f69cab66c33cf75e1aa32b78931585b2712fca441bdb0738d7` |
| `resplat-base-dl3dv-540x960-view16-a72dc6d0.pth` | 896,300,265 | `a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c` |

Copy timing: env tar 14:57→15:07:20 UTC (H1), H1→N1 copy 15:07:46→15:15:05 UTC
(≈11–18 MB/s), N1 sha256 verified 23:16 CST. No split parts needed (env 13G raw,
6.8G archive; under the 25G split threshold). N1 `/root/gpufree-data` free after
backup: 121G.

### Restore instructions

```bash
# 1) conda env (restore at the SAME path — env has hardcoded prefixes/shebangs)
#    from H1's own archive:
tar -C /usr/local/miniconda3/envs -xzf /workspace/ffsplat-backup/ffsplat-enhance-env.tgz
#    or restore from N1's copy over ssh:
ssh -i ~/.ssh/ton1 -p 30130 root@183.147.142.40 \
  'cat /root/gpufree-data/backup/ffsplat-enhance-env.tgz' \
  | tar -C /usr/local/miniconda3/envs -xzf -
/usr/local/miniconda3/envs/ffsplat-enhance/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

# 2) source trees (editable installs point at /workspace/projects/...)
tar -C /workspace/projects -xzf ffsplat-src-FFSplat-Enhance-FFSplat-Eval.tgz
#    or reconstruct history from bundles + patches:
git clone /path/to/FFSplat-Enhance.bundle FFSplat-Enhance && git -C FFSplat-Enhance checkout e880064
git -C FFSplat-Enhance/third_party/FastGS apply FastGS-worktree.patch   # forward.cu / rasterizer_impl.h
#    gaussian-splatting / resplat submodules: only pointer moves, no textual patch.

# 3) checkpoint
mkdir -p /workspace/organizer-assets
cp /root/gpufree-data/backup/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth /workspace/organizer-assets/
sha256sum -c <(echo a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c  /workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth)
```

Version identity captured in `patched-files.txt`:
- FFSplat-Enhance `e8800641a0040ee10cba0c3fc7d3b9d1f20c2282`
- FFSplat-Eval `49b92545dfc42738e36e0917b4f3bfe6b2d6538f`
- submodules: FastGS `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`,
  gaussian-splatting `54c035f7834b564019656c3e3fcc3646292f727d`,
  resplat `cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1`
- modified (patched) files: FastGS
  `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu`,
  `.../rasterizer_impl.h`; gaussian-splatting `submodules/simple-knn` (pointer only).
- Both repos are editable-installed into the env
  (`__editable__.ffsplat_{enhance,eval}-0.1.0.pth`), so env + these trees + checkpoint
  = restorable ReSplat pipeline.

## 6. Verification commands used

```bash
# 13/13 manifest validity + cache presence (N1)
cd /root/projects/FFSplat-Hackathon
/opt/conda/envs/fastgs/bin/python /tmp/verify_manifest.py   # prints VALID 13/13
# cache sanity: npy shape == image count, dtype uint8, manifest.json present
/opt/conda/envs/fastgs/bin/python /tmp/cache_sanity.py
# manifest sha on both nodes
sha256sum records/evalnode-manifest-13.json   # c2aac742…
# checkpoint sha on H1/N1
sha256sum resplat-base-dl3dv-540x960-view16-a72dc6d0.pth  # a72dc6d0…
```

## 7. Notes / caveats

- H1 handshakes are flaky; all long H1 work ran under `setsid nohup` and survived.
  H1 is **UTC**, N1 is CST (+8) — log timestamps differ by 8h.
- N1 ROOT overlay is 99% full (312M). Only `/root/gpufree-data` was written
  (`inputs13/`, `image_cache` dirs under `outputs/`, `backup/`).
- The `outputs/inputs13` symlink is the only non-data change made in the N1 repo;
  no source code or recipe was modified on either node.
- Organizer raw datasets untouched.
- Env tarball integrity: H1 sha256 = N1 sha256 =
  `cce97ba5c541892bdfe952d0ec29f4dfd51f16218df7223f0f18502bce987916`; N1
  `sha256sum -c` result is `OK` in `/root/gpufree-data/backup/verify.log`,
  sizes match byte-for-byte (7,252,564,215 on both nodes).
  (The first N1-side `sha256sum -c` attempt failed only because the copied `.sha256`
  file still carried the H1 absolute path; it was rewritten to the N1 filename and
  re-verified OK.)
- No H1 GPU work was started by this task (only CPU tar/gzip/ssh); H1 GPU stays free
  for the Lead's LPIPS screening experiment.
- Exact measured recipe, `run_pipeline.py`, `research/` code and `docs/` were NOT
  touched by this task.
