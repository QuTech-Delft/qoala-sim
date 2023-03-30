from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from symbol import compound_stmt
from typing import Dict, List, Optional, Tuple, Union

from netqasm.lang.instr import core

from qoala.lang.ehi import ExposedHardwareInfo, NetworkEhi
from qoala.lang.hostlang import (
    BasicBlockType,
    ReceiveCMsgOp,
    RunRequestOp,
    RunSubroutineOp,
)
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
    typ: BasicBlockType
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

    @classmethod
    def consecutive(cls, task_list: CpuQpuTaskList) -> CpuSchedule:
        cpu_tasks: List[Tuple[Optional[float], CpuTask]] = []

        time = 0.0
        for task in task_list.tasks:
            if isinstance(task, CpuTask):
                cpu_tasks.append((time, task))
            else:
                assert isinstance(task, QpuTask)
            # TODO: is a None value for duration even allowed here?
            if task.duration:
                time += task.duration
        return CpuSchedule(cpu_tasks)


@dataclass(eq=True, frozen=True)
class QpuTask:
    pid: int
    routine_type: RoutineType
    block_name: str
    typ: BasicBlockType
    duration: Optional[float] = None
    max_time: Optional[float] = None


@dataclass
class QpuSchedule:
    tasks: List[Tuple[Optional[float], QpuTask]]  # time -> task

    @classmethod
    def no_constraints(cls, tasks: List[QpuTask]) -> QpuSchedule:
        return QpuSchedule(tasks=[(None, t) for t in tasks])

    @classmethod
    def consecutive(cls, task_list: CpuQpuTaskList) -> QpuSchedule:
        qpu_tasks: List[Tuple[Optional[float], QpuTask]] = []

        time = 0.0
        for task in task_list.tasks:
            if isinstance(task, QpuTask):
                qpu_tasks.append((time, task))
            else:
                assert isinstance(task, CpuTask)
            # TODO: is a None value for duration even allowed here?
            if task.duration:
                time += task.duration
        return QpuSchedule(qpu_tasks)


class ScheduleWriter:
    def __init__(self, schedule: CpuQpuSchedule) -> None:
        self._timeline = "time "
        self._cpu_task_str = "CPU  "
        self._qpu_task_str = "QPU  "
        self._cpu_tasks = schedule.cpu_schedule.tasks
        self._qpu_tasks = schedule.qpu_schedule.tasks

    def _add_cpu_entry(
        self, cpu_time: Optional[float], cpu_task: CpuTask
    ) -> Tuple[str, str]:
        width = max(len(cpu_task.block_name), len(str(cpu_time))) + 7
        if cpu_time is None:
            cpu_time = "<none>"
        self._timeline += f"{cpu_time:<{width}}"
        cpu_name = f"{cpu_task.block_name} ({cpu_task.typ.name})"
        self._cpu_task_str += f"{cpu_name:<{width}}"
        self._qpu_task_str += " " * width

    def _add_qpu_entry(
        self, qpu_time: Optional[float], qpu_task: CpuTask
    ) -> Tuple[str, str]:
        width = max(len(qpu_task.block_name), len(str(qpu_time))) + 7
        if qpu_time is None:
            qpu_time = "<none>"
        self._timeline += f"{qpu_time:<{width}}"
        qpu_name = f"{qpu_task.block_name} ({qpu_task.typ.name})"
        self._cpu_task_str += " " * width
        self._qpu_task_str += f"{qpu_name:<{width}}"

    def _add_double_entry(
        self, time: Optional[float], cpu_task: CpuTask, qpu_task: QpuTask
    ) -> Tuple[str, str]:
        width = (
            max(len(cpu_task.block_name), len(qpu_task.block_name), len(str(time))) + 7
        )
        self._timeline += f"{time:<{width}}"
        cpu_name = f"{cpu_task.block_name} ({cpu_task.typ.name})"
        self._cpu_task_str += f"{cpu_name:<{width}}"
        qpu_name = f"{qpu_task.block_name} ({qpu_task.typ.name})"
        self._qpu_task_str += f"{qpu_name:<{width}}"

    def write(self) -> str:
        cpu_index = 0
        qpu_index = 0
        cpu_done = False
        qpu_done = False
        while True:
            try:
                cpu_time, cpu_task = self._cpu_tasks[cpu_index]
            except IndexError:
                cpu_done = True
            try:
                qpu_time, qpu_task = self._qpu_tasks[qpu_index]
            except IndexError:
                qpu_done = True
            if cpu_done and qpu_done:
                break
            if qpu_done:  # cpu_done is False
                cpu_index += 1
                self._add_cpu_entry(cpu_time, cpu_task)
            elif cpu_done:  # qpu_done is False
                qpu_index += 1
                self._add_qpu_entry(qpu_time, qpu_task)
            else:  # both not done
                if qpu_time is None:
                    cpu_index += 1
                    self._add_cpu_entry(cpu_time, cpu_task)
                elif cpu_time is None:
                    qpu_index += 1
                    self._add_qpu_entry(qpu_time, qpu_task)
                elif cpu_time < qpu_time:
                    cpu_index += 1
                    self._add_cpu_entry(cpu_time, cpu_task)
                elif qpu_time < cpu_time:
                    qpu_index += 1
                    self._add_qpu_entry(qpu_time, qpu_task)
                else:  # times equal
                    cpu_index += 1
                    qpu_index += 1
                    self._add_double_entry(cpu_time, cpu_task, qpu_task)
        return self._timeline + "\n" + self._cpu_task_str + "\n" + self._qpu_task_str


@dataclass
class CpuQpuSchedule:
    cpu_schedule: CpuSchedule
    qpu_schedule: QpuSchedule

    @classmethod
    def consecutive_with_QC_constraint(
        cls,
        task_list: CpuQpuTaskList,
        qc_slots: List[float],
        cc_buffer: float,
        free_after_index: Optional[int] = None,
    ) -> CpuQpuSchedule:
        cpu_tasks: List[Tuple[Optional[float], CpuTask]] = []
        qpu_tasks: List[Tuple[Optional[float], QpuTask]] = []

        # Get QC task indices
        qc_indices = []
        for i, task in enumerate(task_list.tasks):
            if isinstance(task, QpuTask) and task.routine_type == RoutineType.REQUEST:
                qc_indices.append(i)

        # Naive approach: first schedule all tasks consecutively. Then, move the first
        # QC task forward by X until it aligns with a qc_slot. Also move all tasks
        # coming fater this QC task by X (so they are still consecutive).
        # Repeat for the next QC task in the list.

        # list of timestamps for each task (same order as tasks in task_list)
        timestamps: List[float] = []
        time = 0.0
        for i, task in enumerate(task_list.tasks):
            if free_after_index is not None and i >= free_after_index:
                timestamps.append(None)
            else:
                timestamps.append(time)
                if task.duration:
                    time += task.duration
                    if task.typ == BasicBlockType.CC:
                        time += cc_buffer

        for index in qc_indices:
            for qc_slot in qc_slots:
                if qc_slot >= timestamps[index]:
                    delta = qc_slot - timestamps[index]
                    for i in range(index, len(timestamps)):
                        if timestamps[i] is not None:
                            timestamps[i] += delta
                    break

        for i, task in enumerate(task_list.tasks):
            time = timestamps[i]
            if isinstance(task, CpuTask):
                cpu_tasks.append((time, task))
            else:
                assert isinstance(task, QpuTask)
                qpu_tasks.append((time, task))

        return CpuQpuSchedule(CpuSchedule(cpu_tasks), QpuSchedule(qpu_tasks))

    def __str__(self) -> str:
        return ScheduleWriter(self).write()


@dataclass
class CpuQpuTaskList:
    tasks: List[Union[CpuTask, QpuTask]]

    def cpu_tasks(self) -> List[CpuTask]:
        return [task for task in self.tasks if isinstance(task, CpuTask)]

    def qpu_tasks(self) -> List[QpuTask]:
        return [task for task in self.tasks if isinstance(task, QpuTask)]


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode

    def from_program(
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
    ) -> CpuQpuTaskList:
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
    ) -> CpuQpuTaskList:
        tasks: List[Union[CpuTask, QpuTask]] = []

        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                cputask = CpuTask(pid, block.name, block.typ, duration)
                tasks.append(cputask)
            elif block.typ == BasicBlockType.CC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, ReceiveCMsgOp)
                if ehi is not None:
                    duration = ehi.latencies.host_peer_latency * 5
                else:
                    duration = None
                cputask = CpuTask(pid, block.name, block.typ, duration)
                tasks.append(cputask)
            elif block.typ == BasicBlockType.QL:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    local_routine = program.local_routines[instr.subroutine]
                    duration = self._compute_lr_duration(ehi, local_routine)
                else:
                    duration = None
                qputask = QpuTask(
                    pid, RoutineType.LOCAL, block.name, block.typ, duration
                )
                tasks.append(qputask)
            elif block.typ == BasicBlockType.QC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunRequestOp)
                if network_ehi is not None:
                    # TODO: refactor!!
                    epr_time = list(network_ehi.links.values())[0].duration
                    req_routine = program.request_routines[instr.req_routine]
                    duration = epr_time * req_routine.request.num_pairs
                else:
                    duration = None
                qputask = QpuTask(
                    pid, RoutineType.REQUEST, block.name, block.typ, duration
                )
                tasks.append(qputask)

        return CpuQpuTaskList(tasks)

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
                # TODO: can we always use index 0 ??
                if info := ehi.find_single_gate(0, type(instr)):
                    duration += info.duration
                # TODO: can we always use indices 0, 1 ??
                elif info := ehi.find_multi_gate([0, 1], type(instr)):
                    duration += info.duration
                else:
                    raise RuntimeError
        return duration
