from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
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


@dataclass(eq=True, frozen=True)
class BlockTask:
    pid: int
    block_name: str
    typ: BasicBlockType
    duration: Optional[float] = None
    max_time: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.block_name} ({self.typ.name}), dur={self.duration}"


class ScheduleWriter:
    def __init__(self, schedule: TaskSchedule) -> None:
        self._timeline = "time "
        self._cpu_task_str = "CPU  "
        self._qpu_task_str = "QPU  "
        self._cpu_entries = schedule.cpu_schedule.entries
        self._qpu_entries = schedule.qpu_schedule.entries
        self._entry_width = 12

    def _entry_content(self, task: BlockTask) -> str:
        if task.duration:
            return f"{task.block_name} ({task.typ.name}, {int(task.duration)})"
        else:
            return f"{task.block_name} ({task.typ.name})"

    def _add_cpu_entry(
        self, cpu_time: Optional[float], cpu_task: BlockTask
    ) -> Tuple[str, str]:
        width = max(len(cpu_task.block_name), len(str(cpu_time))) + self._entry_width
        if cpu_time is None:
            cpu_time = "<none>"
        self._timeline += f"{cpu_time:<{width}}"
        entry = self._entry_content(cpu_task)
        self._cpu_task_str += f"{entry:<{width}}"
        self._qpu_task_str += " " * width

    def _add_qpu_entry(
        self, qpu_time: Optional[float], qpu_task: BlockTask
    ) -> Tuple[str, str]:
        width = max(len(qpu_task.block_name), len(str(qpu_time))) + self._entry_width
        if qpu_time is None:
            qpu_time = "<none>"
        self._timeline += f"{qpu_time:<{width}}"
        entry = self._entry_content(qpu_task)
        self._cpu_task_str += " " * width
        self._qpu_task_str += f"{entry:<{width}}"

    def _add_double_entry(
        self, time: Optional[float], cpu_task: BlockTask, qpu_task: BlockTask
    ) -> Tuple[str, str]:
        width = (
            max(len(cpu_task.block_name), len(qpu_task.block_name), len(str(time)))
            + self._entry_width
        )
        self._timeline += f"{time:<{width}}"
        cpu_entry = self._entry_content(cpu_task)
        qpu_entry = self._entry_content(qpu_task)
        self._cpu_task_str += f"{cpu_entry:<{width}}"
        self._qpu_task_str += f"{qpu_entry:<{width}}"

    def write(self) -> str:
        cpu_index = 0
        qpu_index = 0
        cpu_done = False
        qpu_done = False
        while True:
            try:
                cpu_entry = self._cpu_entries[cpu_index]
                cpu_task = cpu_entry.task
                cpu_time = cpu_entry.timestamp
            except IndexError:
                cpu_done = True
            try:
                qpu_entry = self._qpu_entries[qpu_index]
                qpu_task = qpu_entry.task
                qpu_time = qpu_entry.timestamp
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
class TaskScheduleEntry:
    task: BlockTask
    timestamp: Optional[float] = None
    prev: Optional[BlockTask] = None

    def is_cpu_task(self) -> bool:
        return self.task.typ == BasicBlockType.CL or self.task.typ == BasicBlockType.CC

    def is_qpu_task(self) -> bool:
        return self.task.typ == BasicBlockType.QL or self.task.typ == BasicBlockType.QC


@dataclass
class TaskSchedule:
    entries: List[TaskScheduleEntry]

    @classmethod
    def consecutive(
        cls,
        task_list: List[BlockTask],
        qc_slots: Optional[List[float]] = None,
        cc_buffer: float = 0,
        free_after_index: Optional[int] = None,
    ) -> TaskSchedule:
        entries: List[TaskScheduleEntry] = []

        if qc_slots is None:
            qc_slots = []

        # Get QC task indices
        qc_indices = []
        for i, task in enumerate(task_list):
            if task.typ == BasicBlockType.QC:
                qc_indices.append(i)

        for task in task_list:
            entry = TaskScheduleEntry(task=task, timestamp=None, prev=None)
            entries.append(entry)

        for i in range(len(entries) - 1):
            e1 = entries[i]
            e2 = entries[i + 1]
            if e1.is_cpu_task() != e2.is_cpu_task():
                e2.prev = e1.task

        return TaskSchedule(entries)

    @classmethod
    def consecutive_timestamps(
        cls,
        task_list: List[BlockTask],
        qc_slots: Optional[List[float]] = None,
        cc_buffer: float = 0,
        free_after_index: Optional[int] = None,
    ) -> TaskSchedule:
        entries: List[TaskScheduleEntry] = []

        if qc_slots is None:
            qc_slots = []

        # Get QC task indices
        qc_indices = []
        for i, task in enumerate(task_list):
            if task.typ == BasicBlockType.QC:
                qc_indices.append(i)

        # Naive approach: first schedule all tasks consecutively. Then, move the first
        # QC task forward by X until it aligns with a qc_slot. Also move all tasks
        # coming fater this QC task by X (so they are still consecutive).
        # Repeat for the next QC task in the list.

        # list of timestamps for each task (same order as tasks in task_list)
        timestamps: List[float] = []
        time = 0.0
        for i, task in enumerate(task_list):
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

        for i, task in enumerate(task_list):
            time = timestamps[i]
            entries.append(TaskScheduleEntry(task, time, None))

        return TaskSchedule(entries)

    @property
    def cpu_schedule(self) -> TaskSchedule:
        entries = [e for e in self.entries if e.is_cpu_task()]
        return TaskSchedule(entries)

    @property
    def qpu_schedule(self) -> TaskSchedule:
        entries = [e for e in self.entries if e.is_qpu_task()]
        return TaskSchedule(entries)

    def __str__(self) -> str:
        return ScheduleWriter(self).write()


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode

    def from_program(
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
    ) -> List[BlockTask]:
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
    ) -> List[BlockTask]:
        tasks: List[BlockTask] = []

        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                cputask = BlockTask(pid, block.name, block.typ, duration)
                tasks.append(cputask)
            elif block.typ == BasicBlockType.CC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, ReceiveCMsgOp)
                if ehi is not None:
                    duration = ehi.latencies.host_peer_latency
                else:
                    duration = None
                cputask = BlockTask(pid, block.name, block.typ, duration)
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
                qputask = BlockTask(pid, block.name, block.typ, duration)
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
                qputask = BlockTask(pid, block.name, block.typ, duration)
                tasks.append(qputask)

        return tasks

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
