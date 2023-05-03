from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from click import Option
from netqasm.lang.instr import core
from numpy import isin

from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.lang.hostlang import (
    BasicBlock,
    BasicBlockType,
    ReceiveCMsgOp,
    RunRequestOp,
    RunSubroutineOp,
)
from qoala.lang.program import QoalaProgram
from qoala.lang.request import CallbackType
from qoala.lang.routine import LocalRoutine


class TaskExecutionMode(Enum):
    ROUTINE_ATOMIC = 0
    ROUTINE_SPLIT = auto()


class ProcessorType(Enum):
    CPU = 0
    QPU = auto()


class QoalaTask:
    def __init__(
        self,
        task_id: int,
        processor_type: ProcessorType,
        pid: int,
        duration: Optional[float] = None,
    ) -> None:
        self._task_id = task_id
        self._processor_type = processor_type
        self._pid = pid
        self._duration = duration

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def processor_type(self) -> ProcessorType:
        return self._processor_type

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QoalaTask):
            return NotImplemented
        return (
            self.task_id == other.task_id
            and self.processor_type == other.processor_type
            and self.pid == other.pid
            and self.duration == other.duration
        )


class HostLocalTask(QoalaTask):
    def __init__(
        self, task_id: int, pid: int, block_name: str, duration: Optional[float] = None
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
        )
        self._block_name = block_name

    @property
    def block_name(self) -> str:
        return self._block_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HostLocalTask):
            return NotImplemented
        return super().__eq__(other) and self.block_name == other.block_name


class HostEventTask(QoalaTask):
    def __init__(
        self, task_id: int, pid: int, block_name: str, duration: Optional[float] = None
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
        )
        self._block_name = block_name

    @property
    def block_name(self) -> str:
        return self._block_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HostEventTask):
            return NotImplemented
        return super().__eq__(other) and self.block_name == other.block_name


class LocalRoutineTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        block_name: str,
        shared_ptr: int,
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
        )
        self._block_name = block_name
        self._shared_ptr = shared_ptr

    @property
    def block_name(self) -> str:
        return self._block_name

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocalRoutineTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.block_name == other.block_name
            and self.shared_ptr == self.shared_ptr
        )


class PreCallTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        block_name: str,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
        )
        self._block_name = block_name
        self._shared_ptr = shared_ptr

    @property
    def block_name(self) -> str:
        return self._block_name

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreCallTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.block_name == other.block_name
            and self.shared_ptr == self.shared_ptr
        )


class PostCallTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        block_name: str,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
        )
        self._block_name = block_name
        self._shared_ptr = shared_ptr

    @property
    def block_name(self) -> str:
        return self._block_name

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PostCallTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.block_name == other.block_name
            and self.shared_ptr == other.shared_ptr
        )


class SinglePairTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        pair_index: int,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
        )
        self._pair_index = pair_index
        self._shared_ptr = shared_ptr

    @property
    def pair_index(self) -> int:
        return self._pair_index

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinglePairTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.pair_index == other.pair_index
            and self.shared_ptr == other.shared_ptr
        )


class MultiPairTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
        )
        self._shared_ptr = shared_ptr

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiPairTask):
            return NotImplemented
        return super().__eq__(other) and self.shared_ptr == other.shared_ptr


class SinglePairCallbackTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        callback_name: str,
        pair_index: int,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
        )
        self._callback_name = callback_name
        self._pair_index = pair_index
        self._shared_ptr = shared_ptr

    @property
    def callback_name(self) -> str:
        return self._callback_name

    @property
    def pair_index(self) -> int:
        return self._pair_index

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinglePairCallbackTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.callback_name == other.callback_name
            and self.pair_index == other.pair_index
            and self.shared_ptr == other.shared_ptr
        )


class MultiPairCallbackTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        callback_name: str,
        shared_ptr: int,  # used to identify shared (with other tasks) lrcall/rrcall objects
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
        )
        self._callback_name = callback_name
        self._shared_ptr = shared_ptr

    @property
    def callback_name(self) -> str:
        return self._callback_name

    @property
    def shared_ptr(self) -> int:
        return self._shared_ptr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiPairCallbackTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.callback_name == other.callback_name
            and self.shared_ptr == other.shared_ptr
        )


class BlockTask(QoalaTask):
    def __init__(
        self,
        task_id: int,
        pid: int,
        block_name: str,
        typ: BasicBlockType,
        duration: Optional[float] = None,
        max_time: Optional[float] = None,
        remote_id: Optional[int] = None,
    ) -> None:
        if typ == BasicBlockType.CL or typ == BasicBlockType.CC:
            processor_type = ProcessorType.CPU
        else:
            processor_type = ProcessorType.QPU
        super().__init__(task_id=task_id, processor_type=processor_type, pid=pid)
        self._block_name = block_name
        self._typ = typ
        self._duration = duration
        self._max_time = max_time
        self._remote_id = remote_id

    @property
    def block_name(self) -> str:
        return self._block_name

    @property
    def typ(self) -> BasicBlockType:
        return self._typ

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @property
    def max_time(self) -> Optional[float]:
        return self._max_time

    @property
    def remote_id(self) -> Optional[int]:
        return self._remote_id

    def __str__(self) -> str:
        return f"{self.block_name} ({self.typ.name}), dur={self.duration}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockTask):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.block_name == other.block_name
            and self.typ == other.typ
            and self.duration == other.duration
            and self.max_time == other.max_time
            and self.remote_id == other.remote_id
        )


@dataclass
class TaskInfo:
    task: QoalaTask
    predecessors: List[int]
    ext_predecessors: List[int]
    deadline: Optional[int]
    rel_deadlines: Dict[int, int]
    ext_rel_deadlines: Dict[int, int]

    @classmethod
    def only_task(cls, task: QoalaTask) -> TaskInfo:
        return TaskInfo(task, [], [], None, {}, {})


@dataclass(eq=True)
class TaskGraph:
    """DAG of Tasks."""

    def __init__(self, tasks: Optional[Dict[int, TaskInfo]] = None) -> None:
        if tasks is None:
            self._tasks: Dict[int, TaskInfo] = {}
        else:
            self._tasks = tasks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskGraph):
            raise NotImplemented
        return all(
            tinfo1 == tinfo2
            for tinfo1, tinfo2 in zip(self._tasks.values(), other._tasks.items())
        )

    def add_tasks(self, tasks: List[QoalaTask]) -> None:
        for task in tasks:
            self._tasks[task.task_id] = TaskInfo.only_task(task)

    def add_precedences(self, precedences: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x precedes y (y should execute after x)
        for (x, y) in precedences:
            assert x in self._tasks and y in self._tasks
            self._tasks[y].predecessors.append(x)

    def add_ext_precedences(self, precedences: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x (which is not in this graph) precedes y
        # (which is in this graph)
        for (x, y) in precedences:
            assert x not in self._tasks and y in self._tasks
            self._tasks[y].ext_predecessors.append(x)

    def add_deadlines(self, deadlines: List[Tuple[int, int]]) -> None:
        for (x, d) in deadlines:
            assert x in self._tasks
            self._tasks[x].deadline = d

    def add_rel_deadlines(self, deadlines: List[Tuple[Tuple[int, int], int]]) -> None:
        # entry ((x, y), d) means
        # task y must start at most time d time units after task x has finished
        for ((x, y), d) in deadlines:
            assert x in self._tasks and y in self._tasks
            self._tasks[y].rel_deadlines[x] = d

    def add_ext_rel_deadlines(
        self, deadlines: List[Tuple[Tuple[int, int], int]]
    ) -> None:
        # entry ((x, y), d) means
        # task y must start at most time d time units after task x has finished
        for ((x, y), d) in deadlines:
            assert x not in self._tasks and y in self._tasks  # x is external
            self._tasks[y].ext_rel_deadlines[x] = d

    def get_tinfo(self, id: int) -> TaskInfo:
        assert id in self._tasks
        return self._tasks[id]

    def get_roots(self, ignore_external: bool = False) -> List[int]:
        # Return all (IDs of) tasks that have no predecessors

        if ignore_external:
            return [
                i for i, tinfo in self._tasks.items() if len(tinfo.predecessors) == 0
            ]
        else:
            return [
                i
                for i, tinfo in self._tasks.items()
                if len(tinfo.predecessors) == 0 and len(tinfo.ext_predecessors) == 0
            ]

    def remove_task(self, id: int) -> None:
        assert id in self.get_roots()
        self._tasks.pop(id)

        # Remove precedences of successor tasks
        for t in self._tasks.values():
            if id in t.predecessors:
                t.predecessors.pop(id)

        # Change relative deadlines to absolute ones
        for t in self._tasks.values():
            if id in t.rel_deadlines:
                t.deadline = t.rel_deadlines.pop(id)

    def get_cpu_graph(self) -> TaskGraph:
        return self.partial_graph(ProcessorType.CPU)

    def get_qpu_graph(self) -> TaskGraph:
        return self.partial_graph(ProcessorType.QPU)

    def cross_predecessors(self, task_id: int, immediate: bool = True) -> Set[int]:
        # Return all (IDs of) tasks that are predecessors that run on
        # the other processor (CPU/QPU).
        # If immediate = False, return all closest such predecessor, even if they are
        # no immediate parents.
        # If immediate = True, return only immediate parents with a different processor
        # type.
        # TODO: remove items from result set when they are ancestors of other items
        # in the set (in which case they are redundant)
        proc_type = self.get_tinfo(task_id).task.processor_type
        cross_preds = set()

        for pred in self.get_tinfo(task_id).predecessors:
            pred_type = self.get_tinfo(pred).task.processor_type
            if pred_type != proc_type:
                cross_preds.add(pred)  # immediate parent of different type
            elif not immediate:
                cross_preds = cross_preds.union(
                    self.cross_predecessors(pred, immediate)
                )
        return cross_preds

    def double_cross_predecessors(self, task_id: int) -> Set[int]:
        # Return all (IDs of) tasks that are the closest predecessors that run on
        # the same processor (CPU/QPU) but where there are tasks of the other processor
        # type inbetween (in the precedence chain).

        # For the first step: only check immediate parents that have different type.
        # Parents with same type already induce a normal precedence constraint in the
        # partial graph.
        cross_preds = self.cross_predecessors(task_id, immediate=True)
        double_cross_preds = set()
        for cp in cross_preds:
            # For each different-type parent, find the nearest ancestor of the original
            # type.
            double_cross_preds = double_cross_preds.union(
                self.cross_predecessors(cp, immediate=False)
            )
        return double_cross_preds

    def partial_graph(self, proc_type: ProcessorType) -> TaskGraph:
        # Filter tasks with the correct type.
        partial_tasks: Dict[int, TaskInfo] = {
            i: tinfo
            for i, tinfo in self._tasks.items()
            if tinfo.task.processor_type == proc_type
        }

        # Precedence constraints.
        # Move predecessor tasks that have been removed to ext_predecessors.
        for tinfo in self._tasks.values():
            # Keep predecessors if they are still in the graph.
            tinfo.predecessors = [
                pred for pred in tinfo.predecessors if pred in partial_tasks
            ]
            # Move others to ext_predecessors.
            tinfo.ext_predecessors = [
                pred for pred in tinfo.predecessors if pred not in partial_tasks
            ]

        # Precedence constraints for same-processor tasks that used to have a
        # precedence chain of other-processor tasks in between them.
        for tid, tinfo in partial_tasks.items():
            for pred in self.double_cross_predecessors(tid):
                if pred not in tinfo.predecessors:
                    tinfo.predecessors.append(pred)

        # Relative deadlines.
        for tinfo in self._tasks.values():
            # Keep rel_deadline to pred if pred is still in the graph.
            tinfo.rel_deadlines = {
                pred: dl
                for pred, dl in tinfo.rel_deadlines.items()
                if pred in partial_tasks
            }
            # Move others to ext_predecessors.
            tinfo.ext_rel_deadlines = {
                pred: dl
                for pred, dl in tinfo.rel_deadlines.items()
                if pred not in partial_tasks
            }

        return TaskGraph(partial_tasks)


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode
        self._task_id_counter = 0
        self._graph = TaskGraph()

    def unique_id(self) -> int:
        id = self._task_id_counter
        self._task_id_counter += 1
        return id

    def from_program(
        self,
        program: QoalaProgram,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        remote_id: Optional[int] = None,
    ) -> TaskGraph:
        if self._mode == TaskExecutionMode.ROUTINE_ATOMIC:
            self._build_routine_atomic(program, pid, ehi, network_ehi, remote_id)
        else:
            self._build_routine_split(program, pid, ehi, network_ehi, remote_id)

        return self._graph

    def _build_routine_atomic(
        self,
        program: QoalaProgram,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        remote_id: Optional[int] = None,
    ) -> None:
        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                task_id = self.unique_id()
                cputask = BlockTask(task_id, pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = cputask
            elif block.typ == BasicBlockType.CC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, ReceiveCMsgOp)
                if ehi is not None:
                    duration = ehi.latencies.host_peer_latency
                else:
                    duration = None
                task_id = self.unique_id()
                cputask = BlockTask(task_id, pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = cputask
            elif block.typ == BasicBlockType.QL:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    local_routine = program.local_routines[instr.subroutine]
                    duration = self._compute_lr_duration(ehi, local_routine)
                else:
                    duration = None
                task_id = self.unique_id()
                qputask = BlockTask(task_id, pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = qputask
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

                task_id = self.unique_id()
                qputask = BlockTask(
                    task_id, pid, block.name, block.typ, duration, remote_id=remote_id
                )
                self._graph.tasks[task_id] = qputask

        # We assume a linear program, where the task graph is a 1-dimensional chain of
        # block tasks. Hence we can trivially add a single dependency to each task.
        # We also make use of the fact that all tasks have incrementing indices
        # starting at 0.
        for i in range(1, len(self._graph.tasks)):
            # task (block) i should come after task (block) i - 1
            self._graph.precedences.append((i - 1, i))

    def _build_routine_split(
        self,
        program: QoalaProgram,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        remote_id: Optional[int] = None,
    ) -> None:
        prev_block_task_id: Optional[int] = None
        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                task_id = self.unique_id()
                self._graph.tasks[task_id] = HostLocalTask(
                    task_id, pid, block.name, duration
                )
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.precedences.append((prev_block_task_id, task_id))
                prev_block_task_id = task_id
            elif block.typ == BasicBlockType.CC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, ReceiveCMsgOp)
                if ehi is not None:
                    duration = ehi.latencies.host_peer_latency
                else:
                    duration = None
                task_id = self.unique_id()
                self._graph.tasks[task_id] = HostEventTask(
                    task_id, pid, block.name, duration
                )
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.precedences.append((prev_block_task_id, task_id))
                prev_block_task_id = task_id
            elif block.typ == BasicBlockType.QL:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    local_routine = program.local_routines[instr.subroutine]
                    lr_duration = self._compute_lr_duration(ehi, local_routine)
                    pre_duration = ehi.latencies.host_instr_time
                    post_duration = ehi.latencies.host_instr_time
                else:
                    lr_duration = None
                    pre_duration = None
                    post_duration = None

                precall_id = self.unique_id()
                # Use a unique "pointer" or identifier which is used at runtime to point
                # to shared data. The PreCallTask will store the lrcall object
                # to this location, such that the LR- and postcall task can
                # access this object using the shared pointer.
                shared_ptr = precall_id  # just use this task id so we know it's unique
                precall_task = PreCallTask(
                    precall_id, pid, block.name, shared_ptr, pre_duration
                )
                self._graph.tasks[precall_id] = precall_task

                lr_id = self.unique_id()
                qputask = LocalRoutineTask(
                    lr_id, pid, block.name, shared_ptr, lr_duration
                )
                self._graph.tasks[lr_id] = qputask

                postcall_id = self.unique_id()
                postcall_task = PostCallTask(
                    postcall_id, pid, block.name, shared_ptr, post_duration
                )
                self._graph.tasks[postcall_id] = postcall_task

                # LR task should come after precall task
                self._graph.precedences.append((precall_id, lr_id))
                # postcall task should come after LR task
                self._graph.precedences.append((lr_id, postcall_id))

                # Tasks for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    # First task for this block is precall task.
                    self._graph.precedences.append((prev_block_task_id, precall_id))
                # Last task for this block is postcall task.
                prev_block_task_id = postcall_id
            elif block.typ == BasicBlockType.QC:
                precall_id, postcall_id = self._build_from_qc_task_routine_split(
                    program, block, pid, ehi, network_ehi
                )
                # Tasks for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    # First task for QC block is precall task.
                    self._graph.precedences.append((prev_block_task_id, precall_id))
                # Last task for QC block is postcall task.
                prev_block_task_id = postcall_id  # (not precall_id !)

    def _build_from_qc_task_routine_split(
        self,
        program: QoalaProgram,
        block: BasicBlock,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
    ) -> Tuple[int, int]:
        """Returns (precall_id, post_call_id)"""
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)
        req_routine = program.request_routines[instr.req_routine]
        callback = req_routine.callback

        if ehi is not None:
            # TODO: make more accurate!
            pre_duration = ehi.latencies.host_instr_time
            post_duration = ehi.latencies.host_instr_time
            cb_duration = ehi.latencies.qnos_instr_time
        else:
            pre_duration = None
            post_duration = None
            cb_duration = None

        if network_ehi is not None:
            pair_duration = list(network_ehi.links.values())[0].duration
            multi_duration = pair_duration * req_routine.request.num_pairs
        else:
            pair_duration = None
            multi_duration = None

        precall_id = self.unique_id()
        # Use a unique "pointer" or identifier which is used at runtime to point
        # to shared data. The PreCallTask will store the lrcall or rrcall object
        # to this location, such that the pair- callback- and postcall tasks can
        # access this object using the shared pointer.
        shared_ptr = precall_id  # just use this task id so we know it's unique
        precall_task = PreCallTask(
            precall_id, pid, block.name, shared_ptr, pre_duration
        )
        self._graph.tasks[precall_id] = precall_task

        postcall_id = self.unique_id()
        postcall_task = PostCallTask(
            postcall_id, pid, block.name, shared_ptr, post_duration
        )
        self._graph.tasks[postcall_id] = postcall_task

        if req_routine.callback_type == CallbackType.WAIT_ALL:
            rr_id = self.unique_id()
            rr_task = MultiPairTask(rr_id, pid, shared_ptr, multi_duration)
            self._graph.tasks[rr_id] = rr_task
            # RR task should come after precall task
            self._graph.precedences.append((precall_id, rr_id))

            if callback is not None:
                cb_id = self.unique_id()
                cb_task = MultiPairCallbackTask(
                    cb_id, pid, callback, shared_ptr, cb_duration
                )
                self._graph.tasks[cb_id] = cb_task
                # callback task should come after RR task
                self._graph.precedences.append((rr_id, cb_id))
                # postcall task should come after callback task
                self._graph.precedences.append((cb_id, postcall_id))
            else:  # no callback
                # postcall task should come after RR task
                self._graph.precedences.append((rr_id, postcall_id))

        else:
            assert req_routine.callback_type == CallbackType.SEQUENTIAL
            for i in range(req_routine.request.num_pairs):
                rr_pair_id = self.unique_id()
                rr_pair_task = SinglePairTask(
                    rr_pair_id, pid, i, shared_ptr, pair_duration
                )
                self._graph.tasks[rr_pair_id] = rr_pair_task
                # RR pair task should come after precall task.
                # Note: the RR pair tasks do not have precedence
                # constraints among each other.
                self._graph.precedences.append((precall_id, rr_pair_id))
                if callback is not None:
                    pair_cb_id = self.unique_id()
                    pair_cb_task = SinglePairCallbackTask(
                        pair_cb_id, pid, callback, i, shared_ptr, cb_duration
                    )
                    self._graph.tasks[pair_cb_id] = pair_cb_task
                    # Callback task for pair should come after corresponding
                    # RR pair task. Note: the pair callback tasks do not have
                    # precedence constraints among each other.
                    self._graph.precedences.append((rr_pair_id, pair_cb_id))
                    # postcall task should come after callback task
                    self._graph.precedences.append((pair_cb_id, postcall_id))
                else:  # no callback
                    # postcall task should come after RR task
                    self._graph.precedences.append((rr_pair_id, postcall_id))

        return precall_id, postcall_id

    def _compute_lr_duration(self, ehi: EhiNodeInfo, routine: LocalRoutine) -> float:
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
                max_duration = -1.0
                # TODO: gate duration depends on which qubit!!
                # currently we always take the worst case scenario but this is not ideal
                for i in ehi.single_gate_infos.keys():
                    if info := ehi.find_single_gate(i, type(instr)):
                        max_duration = max(max_duration, info.duration)

                for multi in ehi.multi_gate_infos.keys():
                    if info := ehi.find_multi_gate(multi.qubit_ids, type(instr)):
                        max_duration = max(max_duration, info.duration)

                if max_duration != -1:
                    duration += max_duration
                else:
                    raise RuntimeError
        return duration
