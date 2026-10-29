#!/usr/bin/env bash
# E8 diagnostic (from the advisory): what share of module B is render / backward /
# optimizer? Only worth adapting a sparse optimizer if the optimizer step is large.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=$PWD
PY=${PY:-/opt/conda/bin/conda}; ENV=${ENV:-fastgs}; D=${D:-outputs/experiments/timebd}; mkdir -p "$D"
HW=hw/billboard; BHW=99.49; CO=mipnerf360/counter; BCO=92.57
for spec in "hw:$HW:$BHW" "co:$CO:$BCO"; do
  IFS=':' read -r s case b <<<"$spec"
  [ -f "$D/$s.json" ] && continue
  $PY run --no-capture-output -n "$ENV" python -m research.train_sweep \
    --source "outputs/O0-20261030/work/$case" --model_path "$D/$s/model" --eval -r 1 \
    --sh_degree 3 --quiet --init_ply "outputs/O0-20261030/results/$case/resplat/resplat_world.ply" \
    --iterations 5000 --snapshots "" --record "$D/$s.json" --time_breakdown \
    --position_lr_max_steps 5000 --opacity_reset_interval 0 --lambda_dssim 0.6 \
    --scaling_lr 0.01 --densify_until_iter 1250 >>"$D/$s.log" 2>&1
  rm -rf "$D/$s/model" 2>/dev/null
done
python3 -c "
import json,glob
for p in sorted(glob.glob('$D/*.json')):
    d=json.load(open(p)); t=d.get('timing_breakdown') or {}
    tot=sum(t.get(k,0) for k in ('render','backward','optimizer','grad_to_step')) or 1
    print(p.split('/')[-1], {k: (round(100*t.get(k,0)/tot,1)) for k in ('render','backward','optimizer','grad_to_step')}, 'train', d.get('train_seconds'))
"
