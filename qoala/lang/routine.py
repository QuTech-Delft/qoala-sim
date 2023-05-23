from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from netqasm.lang.subroutine import Subroutine


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


@dataclass(eq=True, frozen=True)
class LrReturnVector:
    name: str
    size: int

    def __str__(self) -> str:
        return f"{self.name}<{self.size}>"


class LocalRoutine:
    def __init__(
        self,
        name: str,
        subrt: Subroutine,
        return_vars: List[Union[str, LrReturnVector]],
        metadata: RoutineMetadata,
        request_name: Optional[str] = None,
    ) -> None:
        self._name = name
        self._subrt = subrt
        self._return_vars = return_vars
        self._metadata = metadata
        self._request_name = request_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def subroutine(self) -> Subroutine:
        return self._subrt

    @property
    def return_vars(self) -> List[Union[str, LrReturnVector]]:
        return self._return_vars

    @property
    def metadata(self) -> RoutineMetadata:
        return self._metadata

    @property
    def request_name(self) -> Optional[str]:
        return self._request_name

    def get_return_size(self) -> int:
        size = 0
        for v in self.return_vars:
            if isinstance(v, LrReturnVector):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocalRoutine):
            return NotImplemented
        return (
            self.name == other.name
            and self.subroutine == other.subroutine
            and self.metadata == other.metadata
            and self.return_vars == other.return_vars
        )
