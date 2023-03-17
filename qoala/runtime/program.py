from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qoala.lang.program import IqoalaProgram
from qoala.runtime.schedule import ProgramTaskList


class ProgramContext(abc.ABC):
    pass


@dataclass
class ProgramInput:
    values: Dict[str, Any]


@dataclass
class ProgramResult:
    values: Dict[str, Any]


@dataclass
class BatchInfo:
    """Description of a batch of program instances that should be executed."""

    program: IqoalaProgram
    inputs: List[ProgramInput]  # dict of inputs for each iteration
    num_iterations: int
    deadline: float
    tasks: ProgramTaskList
    num_qubits: int  # TODO: replace this by unit module


@dataclass
class ProgramInstance:
    """A running program"""

    pid: int
    program: IqoalaProgram
    inputs: ProgramInput
    tasks: ProgramTaskList


@dataclass
class ProgramBatch:
    batch_id: int
    info: BatchInfo
    instances: List[ProgramInstance]


@dataclass
class BatchResult:
    batch_id: int
    results: List[ProgramResult]


# TODO: move below classes to qoala.lang?
# or to qoala.sim ??
@dataclass
class RequestRoutineParams:
    pass


@dataclass
class RequestRoutineResult:
    meas_outcomes: Optional[List[int]]

    @classmethod
    def empty(cls) -> RequestRoutineResult:
        return RequestRoutineResult(meas_outcomes=None)


@dataclass
class CallbackRoutineParams:
    pass


@dataclass
class LocalRoutineParams:
    pass


@dataclass
class LocalRoutineResult:
    pass
