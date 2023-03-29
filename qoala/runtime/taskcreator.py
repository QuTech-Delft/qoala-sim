from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from netqasm.lang.instr import core

from qoala.lang.ehi import ExposedHardwareInfo, NetworkEhi
from qoala.lang.hostlang import BasicBlockType, RunRequestOp, RunSubroutineOp
from qoala.lang.program import IqoalaProgram
from qoala.lang.routine import LocalRoutine


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


@dataclass(eq=True, frozen=True)
class CpuTask:
    pid: int
    block_name: str
    duration: Optional[float] = None
    max_time: Optional[float] = None


@dataclass
class CpuSchedule:
    # list of (time -> task) entries
    # if time is None, it means "no time restriction", which in general means:
    # execute immediately after previous task
    tasks: List[Tuple[Optional[float], CpuTask]]  # time -> task

    @classmethod
    def no_constraints(cls, tasks: List[CpuTask]) -> CpuSchedule:
        return CpuSchedule(tasks=[(None, t) for t in tasks])


@dataclass(eq=True, frozen=True)
class QpuTask:
    pid: int
    routine_type: RoutineType
    block_name: str
    duration: Optional[float] = None
    max_time: Optional[float] = None


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
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
    ) -> Tuple[List[CpuTask], List[QpuTask]]:
        if self._mode == TaskExecutionMode.ROUTINE_ATOMIC:
            return self._from_program_routine_atomic(program, pid, ehi, network_ehi)
        else:
            raise NotImplementedError

    def _from_program_routine_atomic(
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
    ) -> Tuple[List[CpuTask], List[QpuTask]]:
        cpu_tasks: List[CpuTask] = []
        qpu_tasks: List[QpuTask] = []

        for block in program.blocks:
            if block.typ == BasicBlockType.HOST:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                cputask = CpuTask(pid, block.name, duration)
                cpu_tasks.append(cputask)
            elif block.typ == BasicBlockType.LR:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    routine = program.local_routines[instr.subroutine]
                    duration = self._compute_lr_duration(ehi, routine)
                else:
                    duration = None
                qputask = QpuTask(pid, RoutineType.LOCAL, block.name, duration)
                qpu_tasks.append(qputask)
            elif block.typ == BasicBlockType.RR:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunRequestOp)
                if network_ehi is not None:
                    # TODO: refactor!!
                    epr_time = list(network_ehi.links.values())[0].duration
                    routine = program.request_routines[instr.req_routine]
                    duration = epr_time * routine.request.num_pairs
                else:
                    duration = None
                qputask = QpuTask(pid, RoutineType.REQUEST, block.name, duration)
                qpu_tasks.append(qputask)

        return cpu_tasks, qpu_tasks

    def _compute_lr_duration(
        self, ehi: ExposedHardwareInfo, routine: LocalRoutine
    ) -> float:
        duration = 0.0
        # TODO: refactor this
        for instr in routine.subroutine.instructions:
            if (
                type(instr)
                in [
                    core.SetInstruction,
                    core.StoreInstruction,
                    core.LoadInstruction,
                    core.LeaInstruction,
                ]
                or isinstance(instr, core.BranchBinaryInstruction)
                or isinstance(instr, core.BranchUnaryInstruction)
                or isinstance(instr, core.JmpInstruction)
                or isinstance(instr, core.ClassicalOpInstruction)
                or isinstance(instr, core.ClassicalOpModInstruction)
            ):
                duration += ehi.latencies.qnos_instr_time
            else:
                duration += ehi.find_single_gate(0, type(instr)).duration
        return duration
