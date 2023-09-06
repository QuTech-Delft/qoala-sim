from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qoala.lang.ehi import UnitModule
from qoala.lang.program import QoalaProgram


@dataclass
class ProgramInput:
    values: Dict[str, Any]

    @classmethod
    def empty(cls) -> ProgramInput:
        return ProgramInput({})


@dataclass
class ProgramResult:
    values: Dict[str, Any]


@dataclass
class BatchInfo:
    """Description of a batch of program instances that should be executed."""

    program: QoalaProgram
    unit_module: UnitModule
    inputs: List[ProgramInput]  # dict of inputs for each iteration
    num_iterations: int
    deadline: float


@dataclass
class ProgramInstance:
    """A program instantiated with Program Inputs and a Unit Module"""

    pid: int
    program: QoalaProgram
    inputs: ProgramInput
    unit_module: UnitModule


@dataclass
class ProgramBatch:
    batch_id: int
    info: BatchInfo
    instances: List[ProgramInstance]


@dataclass
class BatchResult:
    batch_id: int
    results: List[ProgramResult]
    timestamps: List[Optional[Tuple[float, float]]]  # start, end

    @property
    def durations(self) -> List[float]:
        assert all(entry is not None for entry in self.timestamps)
        return [end - start for (start, end) in self.timestamps]  # type: ignore
