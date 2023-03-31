from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict

from qoala.lang.program import IqoalaProgram


class InstructionType(Enum):
    CL = 0
    CC = auto()
    QL = auto()
    QC = auto()


@dataclass
class ProgramTaskList:
    program: IqoalaProgram
    tasks: Dict[int, Any]  # task index -> task

    @classmethod
    def empty(cls, program: IqoalaProgram) -> ProgramTaskList:
        return ProgramTaskList(program, {})
