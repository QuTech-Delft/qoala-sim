from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from netqasm.lang.subroutine import Subroutine

from qoala.lang.hostlang import IqoalaVector


@dataclass
class RoutineMetadata:
    """
    Metadata about a routine that is used to determine which
    virtual qubits are used and which ones are kept after
    finishing the routine.

    :param qubit_use: IDs in unit module of virtual qubits that are
    used in this routine.
    :param qubit_keep: IDs in unit module of virtual qubits that still have a state
    that should be kept after finishing this routine.
    """

    qubit_use: List[int]
    qubit_keep: List[int]

    @classmethod
    def use_none(cls) -> RoutineMetadata:
        """
        Convenience method for creating a RoutineMetadata object
        that uses no virtual qubits and keeps no virtual qubits.

        :return: RoutineMetadata object that uses no virtual qubits and keeps no virtual qubits.
        """
        return RoutineMetadata([], [])

    @classmethod
    def free_all(cls, ids: List[int]) -> RoutineMetadata:
        """
        Convenience method for creating a RoutineMetadata object
        that uses all given virtual qubits and keeps none of them.
        Basically, frees the given virtual qubits.

        :param ids: IDs in unit module of virtual qubits that are
        :return: RoutineMetadata object that uses all given virtual qubits and keeps none of them.
        """
        return RoutineMetadata(ids, [])


@dataclass(frozen=True)
class LocalRoutine:
    """
    A Local Routine is for quantum operations that are executed on a single node. The quantum operations are stored
    in a NetQasm Subroutine object.

    :param name: Name of the routine.
    :param subroutine: NetQasm Subroutine object that contains the quantum operations.
    :param return_vars: List of variables that are returned by the routine. These can be either strings or
    vector objects.
    :param metadata: Metadata about the routine that is used to determine which virtual qubits are used and which ones
    are kept after finishing the routine.
    :param request_name: Name of the request that called this routine.
    """

    name: str
    subroutine: Subroutine
    return_vars: List[Union[str, IqoalaVector]]
    metadata: RoutineMetadata
    request_name: Optional[str] = None

    def get_return_size(self) -> int:
        """
        Returns the size of the return vector of this routine. This is the sum of the sizes of all return variables.
        Strings have size 1, vectors have size equal to their size.

        :return: Combined size of the return variables.
        """
        size = 0
        for v in self.return_vars:
            if isinstance(v, IqoalaVector):
                assert isinstance(v.size, int)
                size += v.size
            else:
                size += 1
        return size

    def serialize(self) -> str:
        s = f"SUBROUTINE {self.name}"
        s += f"\nparams: {', '.join(self.subroutine.arguments)}"
        s += f"\nreturns: {', '.join(str(v) for v in self.return_vars)}"
        s += f"\nuses: {', '.join(str(q) for q in self.metadata.qubit_use)}"
        s += f"\nkeeps: {', '.join(str(q) for q in self.metadata.qubit_keep)}"
        s += f"\nrequest: {str(self.request_name or '')}"
        s += "\nNETQASM_START\n"
        s += self.subroutine.print_instructions()
        s += "\nNETQASM_END"
        return s

    def __str__(self) -> str:
        s = "\n"
        for value in self.return_vars:
            s += f"return {str(value)}\n"
        s += "NETQASM_START\n"
        s += self.subroutine.print_instructions()
        s += "\nNETQASM_END"
        return s
