import numpy as np

from qoala.util.math import density_matrices_equal


def test_1():
    state1 = np.array([[1, 0], [0, 0]])
    state2 = np.array([[0.99999999, 0], [0, 0]])
    assert density_matrices_equal(state1, state2)

    state1 = np.array([[1, 0], [0, 0]])
    state2 = np.array([[0.99999999, 0], [0, 0.1]])
    assert not density_matrices_equal(state1, state2)

    state1 = np.array([[1, 0], [0, 0]])
    state2 = np.array([[0, 0], [0, 1]])
    assert not density_matrices_equal(state1, state2)


if __name__ == "__main__":
    test_1()
