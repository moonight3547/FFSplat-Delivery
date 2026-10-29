"""Measure random-init FastGS30K on frozen prepared inputs, without reusing timings.

The source control may belong to another GPU: only immutable geometry is reused.
The output records the actual current hardware; it is not a same-GPU re-score of
old candidates unless their hardware identity independently matches.
"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from competition.contracts import read_json, sha256, validate_reference, write_json
from competition.pairing import clone_work, verify_inputs
from competition.run import check_upstream
from competition.runtime import bind_python

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("geometry_control", "work", "profile"):
        p.add_argument("--"+key.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.work.exists():
        raise FileExistsError("Choose a fresh calibration directory; preserve partial evidence")
    old = read_json(args.geometry_control/"run_identity.json")
    profile = read_json(args.profile)
    upstream = ROOT/"competition/upstream"
    check_upstream(upstream, old["benchmark"]["upstream"]["commit"])
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True):
        raise ValueError("Commit source before measurement")
    hardware = dict(host=socket.gethostname(), gpu_inventory=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip())
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU has active compute processes; do not interfere with collaborators")
    identity = dict(reference_trainer="fastgs", iterations=30000, initialization="random",
        timing_revision="fastgs-random30k-user-confirmed-20261029",
        hardware=hardware, profile=profile,
        geometry_control_identity_sha256=sha256(args.geometry_control/"run_identity.json"),
        geometry_control_hardware=old["hardware"],
        same_hardware_as_geometry_control=hardware == old["hardware"],
        reused_timings=False, geometry_source_benchmark=old["benchmark"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(args.work/"calibration_identity.json", identity)
    environment = dict(os.environ, **profile.get("environment", {}))
    environment.update(PYTHONPATH=str(ROOT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    base = [profile["conda"], "run", "--no-capture-output", "-n", profile["environments"]["fastgs"], "python"]
    subprocess.run(bind_python(base+["-m", "competition.doctor", "--runtime", "fastgs"]),
        cwd=ROOT, env=environment, check=True)
    rows = []
    for case in old["benchmark"]["cases"]:
        work = args.work/"work"/case["id"]
        clone_work(args.geometry_control/"work"/case["id"], work)
        frozen = verify_inputs(work)
        out = args.work/"results"/case["id"]/"reference_fastgs"
        argv = base+[str(upstream/"baseline/module_b/train_from_ply_fastgs.py"),
            "-s", str(work), "-m", str(out), "--eval", "-r", "1", "--sh_degree", "3",
            "--iterations", "30000", "--position_lr_max_steps", "30000",
            "--densify_until_iter", "15000", "--init_random", "--quiet"]
        log = args.work/"logs"/case["id"]/"reference.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Start FastGS random30K: {case['id']}", flush=True)
        start = time.perf_counter()
        with log.open("x") as stream:
            subprocess.run(bind_python(argv), cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        wall = time.perf_counter()-start
        record = validate_reference(read_json(out/"module_b.json"))
        verify_inputs(work)
        write_json(log.with_suffix(".json"), dict(argv=argv, status="complete", subprocess_wall_seconds=wall))
        rows.append(dict(id=case["id"], geometry_sha256=frozen["geometry_sha256"],
            reference=record, reference_sha256=sha256(out/"module_b.json"), subprocess_wall_seconds=wall))
        write_json(args.work/"calibration.json", dict(identity=identity, rows=rows,
            complete=len(rows)==len(old["benchmark"]["cases"])))
        print(f"Complete: {case['id']}: {record['total_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
