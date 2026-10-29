"""Inspect tensor integrity on CPU with weights_only=True; never build the model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .assets import checkpoint_record, ROOT
from .contracts import read_json, write_json


def audit(path):
    # NumPy first avoids duplicate OpenMP initialization in the Windows controller.
    import numpy  # noqa: F401
    import torch
    record = checkpoint_record(path, read_json(ROOT/"upstream-lock.json"))
    with zipfile.ZipFile(path) as archive:
        record["zip_root"] = archive.namelist()[0].split("/")[0]
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = value.get("state_dict", value)
    schema = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Invalid or nonfinite state tensor: {key}")
        schema[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), elements=tensor.numel())
    if not schema: raise ValueError("Empty model state")
    record.update(top_level_keys=list(value) if "state_dict" in value else ["flat_state_dict"],
                  tensors=len(schema), tensor_elements=sum(t["elements"] for t in schema.values()),
                  tensor_schema_sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                  filename_hash_prefix_matches=record["sha256"].startswith(Path(path).stem.rsplit("-", 1)[-1]),
                  weights_only_load="CPU passed; every state tensor finite",
                  model_architecture_load="pending organizer model/CUDA runtime")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    record = audit(args.checkpoint)
    write_json(args.output, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
