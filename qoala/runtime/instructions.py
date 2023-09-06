import functools as ft
from typing import List, Tuple

import numpy as np
from netsquid.components.instructions import Instruction
from netsquid.components.qmemory import QuantumMemory
from netsquid.qubits import operators as ops
from scipy.linalg import expm


class IMeasAll(Instruction):
    """
    Measurement instruction for all qubits in a quantum memory.
    """

    @property
    def name(self) -> str:
        """instruction name."""
        return "measure_all"

    @property
    def num_positions(self) -> int:
        """number of targeted memory positions. If -1, number is unrestricted."""
        return -1

    def execute(
        self, quantum_memory: QuantumMemory, positions: List[int], **kwargs
    ) -> List[List[int]]:
        """Execute instruction on a quantum memory.

        :param quantum_memory: NetSquid Quantum memory to execute instruction on.
        :param positions: Memory positions_of_connections to execute instruction on.
        """
        result = []
        for pos in positions:
            result.append(quantum_memory.measure(positions=[pos])[0][0])
        return [result]


class IRotationAllGate(Instruction):
    """
    Rotation instruction for all qubits in a quantum memory.
    """

    def __init__(self, name: str, axis: Tuple[int, int, int]) -> None:
        self._name = name
        self._axis = axis

    @property
    def name(self) -> str:
        """instruction name."""
        return self._name

    @property
    def num_positions(self) -> int:
        """number of targeted memory positions. If -1, number is unrestricted."""
        return -1

    def execute(
        self,
        quantum_memory: QuantumMemory,
        positions: List[int],
        angle: float,
        **kwargs
    ) -> None:
        """Execute instruction on a quantum memory.

        :param quantum_memory: NetSquid Quantum memory to execute instruction on.
        :param positions: Memory positions_of_connections to execute instruction on.
        """
        print("executing rotation instruction with angle: ", angle)
        operator = ops.create_rotation_op(angle=angle, rotation_axis=self._axis)
        for pos in positions:
            quantum_memory.operate(positions=pos, operator=operator)


class IBichromaticGate(Instruction):
    """
    Bichromatic gate instruction.
    """

    @property
    def name(self) -> str:
        """instruction name."""
        return "bichromatic"

    @property
    def num_positions(self) -> int:
        """number of targeted memory positions. If -1, number is unrestricted."""
        return -1

    def construct_operator(self, n: int, angle: float) -> ops.Operator:
        """Construct operator which is applied by the gate.
        Used by execute method.

        :param n: Number of qubits to apply operator on.
        :param angle: Angle of rotation.

        """
        sub_matrices = []
        X = np.array([[0, 1], [1, 0]])
        for i in range(2, n + 1):
            for j in range(1, i):
                identity1 = np.eye(2 ** (j - 1))
                identity2 = np.eye(2 ** (i - j - 1))
                identity3 = np.eye(2 ** (n - i))
                sub_matrix_parts = [identity1, X, identity2, X, identity3]
                sub_matrix = ft.reduce(np.kron, sub_matrix_parts)
                sub_matrices.append(sub_matrix)

        matrix = expm(-1j * angle / 2 * np.sum(sub_matrices, axis=0))
        operator = ops.Operator(name="bichromatic_operator", matrix=matrix)
        return operator

    def execute(
        self,
        quantum_memory: QuantumMemory,
        positions: List[int],
        angle: float,
        **kwargs
    ) -> None:
        """Execute instruction on a quantum memory.

        :param quantum_memory: NetSquid Quantum memory to execute instruction on.
        :param positions: Memory positions to do instruction on. Can be empty.
        """

        operator = self.construct_operator(n=len(positions), angle=angle)
        print("operator: ", operator)
        quantum_memory.operate(positions=positions, operator=operator)


INSTR_MEASURE_ALL = IMeasAll()
INSTR_ROT_X_ALL = IRotationAllGate("x_rot_all_gate", axis=(1, 0, 0))
INSTR_ROT_Y_ALL = IRotationAllGate("y_rot_all_gate", axis=(0, 1, 0))
INSTR_ROT_Z_ALL = IRotationAllGate("z_rot_all_gate", axis=(0, 0, 1))
INSTR_BICHROMATIC = IBichromaticGate()
