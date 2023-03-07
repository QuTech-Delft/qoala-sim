from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from netqasm.lang.subroutine import Subroutine

from qoala.lang.hostlang import IqoalaSharedMemLoc


@dataclass
class RoutineMetadata:
    # IDs in unit module of virtual qubits that are
    # used in this routine
    qubit_use: List[int]

    # IDs in unit module of virtual qubits that still have a state
    # that should be kept after finishing this routine
    qubit_keep: List[int]

    @classmethod
    def frees_all(cls, ids: List[int]) -> RoutineMetadata:
        return RoutineMetadata(ids, [])


class LocalRoutine:
    def __init__(
        self,
        name: str,
        subrt: Subroutine,
        return_map: Dict[str, IqoalaSharedMemLoc],
        metadata: RoutineMetadata,
        request_name: Optional[str] = None,
    ) -> None:
        self._name = name
        self._subrt = subrt
        self._return_map = return_map
        self._request_name = request_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def subroutine(self) -> Subroutine:
        return self._subrt

    @property
    def return_map(self) -> Dict[str, IqoalaSharedMemLoc]:
        return self._return_map

    @property
    def request_name(self) -> Optional[str]:
        return self._request_name

    def serialize(self) -> str:
        s = f"SUBROUTINE {self.name}"
        s += f"\nparams: {', '.join(self.subroutine.arguments)}"
        rm = self.return_map  # just to make next line fit on one line
        s += f"\nreturns: {', '.join(f'{v} -> {k}' for k, v in rm.items())}"
        s += "\nNETQASM_START\n"
        s += self.subroutine.print_instructions()
        s += "\nNETQASM_END"
        return s

    def __str__(self) -> str:
        s = "\n"
        for key, value in self.return_map.items():
            s += f"return {str(value)} -> {key}\n"
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
            and self.return_map == other.return_map
        )
