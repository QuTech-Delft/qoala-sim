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
class ProgramCopies:
    """A program along with a list of inputs to instantiate with.
    Each input is assigned to one program instance.
    Very similar to `BatchInfo`, but does not depend on `UnitModule`
    and does not allow for contradiction between number of inputs and iterations."""

    program: QoalaProgram
    inputs: list[ProgramInput]

    @staticmethod
    def from_input_copies(
        program: QoalaProgram,
        input: ProgramInput = ProgramInput.empty(),
        iterations: int = 1,
    ):
        return ProgramCopies(program, [input] * iterations)

    @staticmethod
    def from_varied_inputs(
        program: QoalaProgram,
        *inputs: ProgramInput,
    ):
        return ProgramCopies(program, list(inputs))

    @property
    def iterations(self):
        return len(self.inputs)


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
