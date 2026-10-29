"""CPU-only protocol checks shared by organizer launch and migration adapters."""
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_split(split):
    names = split["images"]
    if not names or names != sorted(set(names)):
        raise ValueError("Images must have unique sorted names")
    if any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Image names must be plain filenames")
    if split["llffhold"] != 8:
        raise ValueError("Organizer llffhold must be 8")
    if split["test"] != names[::8] or split["train"] != [n for i, n in enumerate(names) if i % 8]:
        raise ValueError("Organizer train/test partition changed")
    context = split["context"]
    if len(context) != min(16, len(split["train"])) or len(set(context)) != len(context):
        raise ValueError("Expected up to 16 unique context views")
    if not set(context) <= set(split["train"]):
        raise ValueError("Context contains held-out views")
    h, w = split["resolution"]
    if any(type(x) is not int or x < 64 or x % 64 for x in (h, w)):
        raise ValueError("Organizer dimensions must be positive multiples of 64")
    if max(h, w) > (768 if h > w else 960):
        raise ValueError("Organizer resolution limit exceeded")
    return split


def remaining_b_seconds(reference_seconds, time_a, time_resplat, fraction=0.8):
    """Explicit E2E allocation; no old first20 ratio or TIME score assumption."""
    values = (reference_seconds, time_a, time_resplat, fraction)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite time allocation")
    if reference_seconds <= 0 or min(time_a, time_resplat) < 0 or not 0 < fraction < 1:
        raise ValueError("Invalid E2E time allocation")
    remaining = fraction * reference_seconds - time_a - time_resplat
    if remaining <= 0:
        raise ValueError("A + ReSplat already exhausts the declared E2E target")
    return remaining


def validate_reference(ref):
    """Reject classic, continuation, partial and nonfinite timing references."""
    if (ref.get("trainer") != "fastgs" or type(ref.get("iterations")) is not int
            or ref["iterations"] != 30000 or not str(ref.get("init_mode", "")).startswith("random:")):
        raise ValueError("Reference must be FastGS random-init full30000 (not classic3DGS or ReSplat continuation)")
    seconds = ref.get("total_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid FastGS30K reference time")
    return ref
