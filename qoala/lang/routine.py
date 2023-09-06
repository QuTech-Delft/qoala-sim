from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from netqasm.lang.subroutine import Subroutine

from qoala.lang.hostlang import IqoalaVector


@dataclass
class RoutineMetadata:
    # IDs in unit module of virtual qubits that are
    # used in this routine
    qubit_use: List[int]

    # IDs in unit module of virtual qubits that still have a state
    # that should be kept after finishing this routine
    qubit_keep: List[int]

    @classmethod
    def use_none(cls) -> RoutineMetadata:
        return RoutineMetadata([], [])

    @classmethod
    def free_all(cls, ids: List[int]) -> RoutineMetadata:
        return RoutineMetadata(ids, [])


@dataclass(frozen=True)
class LocalRoutine:
    name: str
    subroutine: Subroutine
    return_vars: List[Union[str, IqoalaVector]]
    metadata: RoutineMetadata
    request_name: Optional[str] = None

    def get_return_size(self) -> int:
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
