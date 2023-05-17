from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple

from qoala.lang.hostlang import BasicBlockType
from qoala.runtime.task import BlockTask, QoalaTask, TaskGraph, TaskInfo


@dataclass
class LinkSlotInfo:
    offset1: float  # time for first pair
    offset2: float  # time for second pair
    period: float  # time until next cycle (with again two pairs)


@dataclass
class QcSlotInfo:
    links: Dict[int, LinkSlotInfo]


@dataclass
class NetworkSchedule:
    bin_length: int
    first_bin: int
    bin_period: int

    def next_bin(self, time: int) -> int:
        offset = time - self.first_bin
        next_bin_index = ceil(offset / self.bin_period)
        next_bin_start = next_bin_index * self.bin_period
        return next_bin_start + self.first_bin


@dataclass
class StaticSchedule:
    entries: List[TaskInfo]

    @classmethod
    def _compute_timestamps(
        cls, task_list: List[BlockTask], qc_slot_info: Optional[QcSlotInfo]
    ) -> List[float]:
        # Get QC task indices
        qc_indices: List[int] = []
        for i, task in enumerate(task_list):
            if task.typ == BasicBlockType.QC:
                qc_indices.append(i)

        # list of timestamps for each task (same order as tasks in task_list)
        timestamps: List[float] = []
        # Naive approach: first schedule all tasks consecutively. Then, move the first
        # QC task forward by X until it aligns with a qc_slot. Also move all tasks
        # coming fater this QC task by X (so they are still consecutive).
        # Repeat for the next QC task in the list.
        time = 0.0
        for i, task in enumerate(task_list):
            timestamps.append(time)
            if task.duration:
                time += task.duration

        if qc_slot_info is not None:
            # remote ID -> (slot, offsetX)
            curr_slots: Dict[int, Tuple[float, int]] = {
                remote_id: (qc_slot_info.links[remote_id].offset1, 1)
                for remote_id in qc_slot_info.links.keys()
            }

            for index in qc_indices:
                remote_id = task_list[index].remote_id
                assert remote_id is not None
                info = qc_slot_info.links[remote_id]
                curr_slot, offset_index = curr_slots[remote_id]
                while curr_slot <= timestamps[index]:
                    if offset_index == 1:
                        offset_index = 2
                        curr_slot += info.offset2 - info.offset1
                    else:
                        offset_index = 1
                        curr_slot += info.period - (info.offset2 - info.offset1)
                curr_slots[remote_id] = curr_slot, offset_index

                delta = curr_slot - timestamps[index]
                for i in range(index, len(timestamps)):
                    if timestamps[i] is not None:
                        timestamps[i] += delta
        return timestamps

    @classmethod
    def consecutive_block_tasks(
        cls, task_graphs: List[TaskGraph], qc_slot_info: Optional[QcSlotInfo] = None
    ) -> StaticSchedule:
        entries: List[TaskInfo] = []

        task_list: List[BlockTask] = []
        for graph in task_graphs:
            tasks = [tinfo.task for tinfo in graph.get_tasks().values()]
            assert all(isinstance(task, BlockTask) for task in tasks)
            task_list.extend(tasks)  # type: ignore

        if qc_slot_info is not None:
            timestamps = cls._compute_timestamps(task_list, qc_slot_info)

        for i, task in enumerate(task_list):
            if qc_slot_info is not None and task.typ == BasicBlockType.QC:
                time = timestamps[i]
            else:
                time = None
            entry = TaskInfo.only_task(task)
            entry.start_time = time
            entries.append(entry)

        for i in range(len(entries) - 1):
            e1 = entries[i]
            e2 = entries[i + 1]
            if e1.is_cpu_task() != e2.is_cpu_task():
                e2.predecessors.append(e1.task.task_id)

        return StaticSchedule(entries)

    @classmethod
    def linear_graph(cls, task_graphs: List[TaskGraph]) -> StaticSchedule:
        entries: List[TaskInfo] = []

        for task_graph in task_graphs:
            # Make a copy so we can alter it without affecting the original graph.
            graph = deepcopy(task_graph)  # TODO: refactor

            while len(graph.get_tasks()) > 0:
                next_task_id = graph.get_roots()[0]
                next_task = graph.get_tinfo(next_task_id).task
                graph.remove_task(next_task_id)
                entry = TaskInfo.only_task(next_task)
                entries.append(entry)

            for i in range(len(entries) - 1):
                e1 = entries[i]
                e2 = entries[i + 1]
                if e1.is_cpu_task() != e2.is_cpu_task():
                    e2.predecessors.append(e1.task.task_id)

        return StaticSchedule(entries)

    @classmethod
    def consecutive_block_tasks_with_timestamps(
        cls,
        task_graphs: List[TaskGraph],
        qc_slot_info: Optional[QcSlotInfo] = None,
    ) -> StaticSchedule:
        entries: List[TaskInfo] = []

        task_list: List[BlockTask] = []
        for graph in task_graphs:
            tasks = [tinfo.task for tinfo in graph.get_tasks().values()]
            assert all(isinstance(task, BlockTask) for task in tasks)
            task_list.extend(tasks)  # type: ignore

        timestamps = cls._compute_timestamps(task_list, qc_slot_info)

        for i, task in enumerate(task_list):
            time = timestamps[i]
            entry = TaskInfo.only_task(task)
            entry.start_time = time
            entries.append(entry)

        return StaticSchedule(entries)

    @property
    def cpu_schedule(self) -> StaticSchedule:
        entries = [e for e in self.entries if e.is_cpu_task()]
        return StaticSchedule(entries)

    @property
    def qpu_schedule(self) -> StaticSchedule:
        entries = [e for e in self.entries if e.is_qpu_task()]
        return StaticSchedule(entries)

    def __str__(self) -> str:
        return ScheduleWriter(self).write()


class ScheduleWriter:
    def __init__(self, schedule: StaticSchedule) -> None:
        self._timeline = "time "
        self._cpu_task_str = "CPU  "
        self._qpu_task_str = "QPU  "
        self._cpu_entries = schedule.cpu_schedule.entries
        self._qpu_entries = schedule.qpu_schedule.entries
        self._entry_width = 12

    def _entry_content(self, task: QoalaTask) -> str:
        if not isinstance(task, BlockTask):
            # TODO implement writing non-BlockTasks
            return ""
        if task.duration:
            return f"{task.block_name} ({task.typ.name}, {int(task.duration)})"
        else:
            return f"{task.block_name} ({task.typ.name})"

    def _add_cpu_entry(self, cpu_time: Optional[float], cpu_task: QoalaTask) -> None:
        if not isinstance(cpu_task, BlockTask):
            # TODO implement writing non-BlockTasks
            return
        width = max(len(cpu_task.block_name), len(str(cpu_time))) + self._entry_width
        if cpu_time is None:
            cpu_time = "<none>"  # type: ignore
        self._timeline += f"{cpu_time:<{width}}"
        entry = self._entry_content(cpu_task)
        self._cpu_task_str += f"{entry:<{width}}"
        self._qpu_task_str += " " * width

    def _add_qpu_entry(self, qpu_time: Optional[float], qpu_task: QoalaTask) -> None:
        if not isinstance(qpu_task, BlockTask):
            # TODO implement writing non-BlockTasks
            return
        width = max(len(qpu_task.block_name), len(str(qpu_time))) + self._entry_width
        if qpu_time is None:
            qpu_time = "<none>"  # type: ignore
        self._timeline += f"{qpu_time:<{width}}"
        entry = self._entry_content(qpu_task)
        self._cpu_task_str += " " * width
        self._qpu_task_str += f"{entry:<{width}}"

    def _add_double_entry(
        self, time: Optional[float], cpu_task: QoalaTask, qpu_task: QoalaTask
    ) -> None:
        if not isinstance(cpu_task, BlockTask) or not isinstance(qpu_task, BlockTask):
            # TODO implement writing non-BlockTasks
            return
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
                cpu_time = cpu_entry.start_time
            except IndexError:
                cpu_done = True
            try:
                qpu_entry = self._qpu_entries[qpu_index]
                qpu_task = qpu_entry.task
                qpu_time = qpu_entry.start_time
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
