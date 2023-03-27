import math

PI = math.pi
PI_OVER_2 = math.pi / 2


def fidelity_to_prob_max_mixed(fid: float) -> float:
    return (1 - fid) * 4.0 / 3.0


def prob_max_mixed_to_fidelity(prob: float) -> float:
    return 1 - 0.75 * prob
