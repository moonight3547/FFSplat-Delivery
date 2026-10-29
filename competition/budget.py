"""Pure time-budget and LR/CTF controller; no quality-dependent stopping."""
import math


def validate_settings(cfg):
    numeric = {
        "seed", "e2e_fraction", "max_updates", "cap_floor", "reserve_fraction", "reserve_seconds",
        "step_safety", "position_lr_initial", "position_lr_final", "lr_end_fraction",
        "topology_stop_fraction", "refine_every", "min_opacity", "noise_lr", "ssim_weight",
        "opacity_regularization", "scale_regularization",
    }
    if set(cfg) != numeric | {"policy", "ctf", "adam_lrs"}:
        raise ValueError("Unknown/missing MCMC settings")
    if any(isinstance(cfg[k], bool) or not isinstance(cfg[k], (int, float)) or
           not math.isfinite(cfg[k]) for k in numeric):
        raise ValueError("MCMC settings must be finite numbers")
    for k in ("seed", "max_updates", "cap_floor", "refine_every"):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == "seed" else 1):
            raise ValueError(f"Invalid {k}")
    for k in ("e2e_fraction", "reserve_fraction", "lr_end_fraction", "topology_stop_fraction", "min_opacity"):
        if not 0 < cfg[k] < 1: raise ValueError(f"Invalid {k}")
    for k in ("reserve_seconds", "position_lr_initial", "position_lr_final"):
        if cfg[k] <= 0: raise ValueError(f"Invalid {k}")
    for k in ("noise_lr", "opacity_regularization", "scale_regularization"):
        if cfg[k] < 0: raise ValueError(f"Invalid {k}")
    if cfg["step_safety"] < 1 or not 0 <= cfg["ssim_weight"] <= 1:
        raise ValueError("Invalid loss/safety setting")
    if cfg["position_lr_final"] > cfg["position_lr_initial"]:
        raise ValueError("Position LR must decay")
    stages = cfg["ctf"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("Missing CTF stages")
    for row in stages:
        if set(row) != {"until", "scale"} or any(type(v) not in (int, float) or not math.isfinite(v) for v in row.values()):
            raise ValueError("Invalid CTF stage")
        if not 0 < row["until"] <= 1 or not 0 < row["scale"] <= 1:
            raise ValueError("Invalid CTF range")
    if stages[-1] != dict(until=1., scale=1.) or any(
            b["until"] <= a["until"] or b["scale"] < a["scale"] for a, b in zip(stages, stages[1:])):
        raise ValueError("CTF must increase to full resolution")
    if set(cfg["adam_lrs"]) != {"scales", "quats", "dc", "rest", "opacities"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in cfg["adam_lrs"].values()):
        raise ValueError("Invalid Adam LR settings")
    return cfg


class Deadline:
    def __init__(self, allocation, setup_seconds, cfg):
        self.cfg = validate_settings(cfg)
        if not all(math.isfinite(x) for x in (allocation, setup_seconds)) or allocation <= 0 or setup_seconds < 0:
            raise ValueError("Invalid measured timing")
        self.reserve = max(cfg["reserve_seconds"], allocation*cfg["reserve_fraction"])
        self.allowed = max(0., allocation-setup_seconds-self.reserve)
        self.observations = []

    def observe(self, seconds, factor, count):
        if not math.isfinite(seconds) or seconds <= 0 or not 0 < factor <= 1 or count <= 0:
            raise ValueError("Invalid observed step")
        self.observations.append((seconds, factor, count))
        self.observations = self.observations[-128:]

    def plan(self, elapsed, steps, count):
        cfg = self.cfg
        progress = min(1., max(elapsed/self.allowed if self.allowed else 1., steps/cfg["max_updates"]))
        factor = next((s["scale"] for s in cfg["ctf"] if progress < s["until"]), 1.)
        # Scale conservatively before a resolution/count increase; retain topology-step peaks.
        predicted = max((seconds*max(1., (factor/old_factor)**2)*max(1., count/old_count)
                         for seconds, old_factor, old_count in self.observations), default=0.)*cfg["step_safety"]
        lr = math.exp(math.log(cfg["position_lr_initial"])+min(1., progress/cfg["lr_end_fraction"])*
                      math.log(cfg["position_lr_final"]/cfg["position_lr_initial"]))
        reason = ("max_updates" if steps >= cfg["max_updates"] else
                  "time_allocation" if elapsed >= self.allowed else
                  "next_step_reserve" if predicted >= self.allowed-elapsed else None)
        return dict(progress=progress, scale=factor, position_lr=lr, predicted_step_seconds=predicted,
                    stop_reason=reason)
