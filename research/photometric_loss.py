"""Uniform EDGE component; retain the original loss scale and ADC thresholds."""


def add_uniform_l1(base_loss, raw_l1, weight):
    return base_loss if weight == 0 else base_loss + weight * raw_l1
