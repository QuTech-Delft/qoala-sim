import math
from typing import List

import numpy as np
from netsquid.qubits import ketstates, qubitapi
from netsquid.qubits.qubit import Qubit

B00_DENS = np.outer(ketstates.b00, ketstates.b00)
B01_DENS = np.outer(ketstates.b01, ketstates.b01)
B10_DENS = np.outer(ketstates.b10, ketstates.b10)
B11_DENS = np.outer(ketstates.b11, ketstates.b11)

S00_DENS = np.outer(ketstates.s00, ketstates.s00)
S01_DENS = np.outer(ketstates.s01, ketstates.s01)
S10_DENS = np.outer(ketstates.s10, ketstates.s10)
S11_DENS = np.outer(ketstates.s11, ketstates.s11)

ONE_MAX_MIXED = np.array([[0.5, 0], [0, 0.5]])
TWO_MAX_MIXED = np.array(
    [
        [0.25, 0, 0, 0],
        [0, 0.25, 0, 0],
        [0, 0, 0.25, 0],
        [0, 0, 0, 0.25],
    ]
)

PI = math.pi
PI_OVER_2 = math.pi / 2


def fidelity_to_prob_max_mixed(num_qubits: int, fid: float) -> float:
    if num_qubits == 1:
        return (1 - fid) * 2.0
    elif num_qubits == 2:
        return (1 - fid) * 4.0 / 3.0
    else:
        raise ValueError


def prob_max_mixed_to_fidelity(num_qubits: int, prob: float) -> float:
    if num_qubits == 1:
        return 1 - 0.5 * prob
    elif num_qubits == 2:
        return 1 - 0.75 * prob
    else:
        raise ValueError


def has_state(qubit: Qubit, state: np.ndarray, margin: float = 0.001) -> bool:
    fidelity = qubitapi.fidelity(qubit, state, squared=True)
    return math.isclose(1.0, fidelity, abs_tol=margin)


def has_multi_state(
    qubits: List[Qubit], state: np.ndarray, margin: float = 0.001
) -> bool:
    fidelity = qubitapi.fidelity(qubits, state, squared=True)
    return math.isclose(1.0, fidelity, abs_tol=margin)


def density_matrices_equal(
    state1: np.ndarray, state2: np.ndarray, margin: float = 0.001
) -> bool:
    distance = np.linalg.norm(state1 - state2, 1) / 2
    return math.isclose(0, distance, abs_tol=margin)


def has_max_mixed_state(qubit: Qubit, margin: float = 0.001) -> bool:
    max_mixed = np.array([[0.5, 0], [0, 0.5]])
    qubit_state = qubitapi.reduced_dm(qubit)
    return density_matrices_equal(qubit_state, max_mixed, margin)


def has_max_mixed_two_qubit_state(q0: Qubit, q1: Qubit, margin: float = 0.001) -> bool:
    max_mixed = TWO_MAX_MIXED
    qubit_state = qubitapi.reduced_dm([q0, q1])
    return density_matrices_equal(qubit_state, max_mixed, margin)
