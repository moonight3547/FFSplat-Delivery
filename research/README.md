# Research harness (ours, upstream untouched)

Small tools for measuring quality-vs-time curves of module B without editing the
pinned organizer code. Everything here imports `competition/upstream/baseline`
read-only.

| script | env | purpose |
|---|---|---|
| `train_sweep.py` | fastgs | FastGS continuation from a ReSplat PLY with schedule knobs and PLY snapshots |
| `render_eval.py` | fastgs | render held-out views from any PLY with the official FastGS renderer |
| `run_sweep.py` | any | driver: train, render every snapshot, score with `baseline/evaluate.py`, apply the proxy /100 formula |

`python -m research.run_sweep --case <id> --work <prepared scene> --init-ply <ply>
 --out <fresh dir> --iterations 12000 --snapshots 1000,2000,4000,8000,12000 --budget <ref30k s>`

The proxy score follows `ProblemDesc.md` 4: PSNR 20, SSIM 20, LPIPS 40, TIME 20 on
module-B seconds over the measured FastGS random-init 30K budget.
