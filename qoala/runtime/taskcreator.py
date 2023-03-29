from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from qoala.lang.hostlang import BasicBlockType, RunRequestOp, RunSubroutineOp
from qoala.lang.program import IqoalaProgram


class TaskExecutionMode(Enum):
    ROUTINE_ATOMIC = 0
    ROUTINE_SPLIT = auto()


class TriggerType(Enum):
    PREV_TASK_DONE = 0
    TIMESTAMP = auto()
    SCHEDULER_SIGNAL = auto()
    OTHER_SIGNAL = auto()


class RoutineType(Enum):
    LOCAL = 0
    REQUEST = auto()


@dataclass
class CpuTask:
    pid: int
    block_name: str


@dataclass
class CpuSchedule:
    # list of (time -> task) entries
    # if time is None, it means "no time restriction", which in general means:
    # execute immediately after previous task
    tasks: List[Tuple[Optional[float], CpuTask]]  # time -> task

    @classmethod
    def no_constraints(cls, tasks: List[CpuTask]) -> CpuSchedule:
        return CpuSchedule(tasks=[(None, t) for t in tasks])


@dataclass
class QpuTask:
    pid: int
    routine_type: RoutineType
    block_name: str


@dataclass
class QpuSchedule:
    tasks: List[Tuple[Optional[float], QpuTask]]  # time -> task

    @classmethod
    def no_constraints(cls, tasks: List[QpuTask]) -> QpuSchedule:
        return QpuSchedule(tasks=[(None, t) for t in tasks])


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode

    def from_program(
        self, program: IqoalaProgram, pid: int
    ) -> Tuple[List[CpuTask], List[QpuTask]]:
        if self._mode == TaskExecutionMode.ROUTINE_ATOMIC:
            return self._from_program_routine_atomic(program, pid)
        else:
            raise NotImplementedError

    def _from_program_routine_atomic(
        self, program: IqoalaProgram, pid: int
    ) -> Tuple[List[CpuTask], List[QpuTask]]:
        cpu_tasks: List[CpuTask] = []
        qpu_tasks: List[QpuTask] = []

        for block in program.blocks:
            if block.typ == BasicBlockType.HOST:
                cputask = CpuTask(pid, block.name)
                cpu_tasks.append(cputask)
            elif block.typ == BasicBlockType.LR:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                qputask = QpuTask(pid, RoutineType.LOCAL, block.name)
                qpu_tasks.append(qputask)
            elif block.typ == BasicBlockType.RR:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunRequestOp)
                qputask = QpuTask(pid, RoutineType.REQUEST, block.name)
                qpu_tasks.append(qputask)

        return cpu_tasks, qpu_tasks
