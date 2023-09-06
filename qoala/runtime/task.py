from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from netqasm.lang.instr import core
from netqasm.lang.operand import Template

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
from qoala.runtime.program import ProgramInstance


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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(pid={self.pid}, tid={self.task_id})"

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

    def is_epr_task(self) -> bool:
        return isinstance(self, SinglePairTask) or isinstance(self, MultiPairTask)

    def is_event_task(self) -> bool:
        return isinstance(self, HostEventTask)

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
        self,
        task_id: int,
        pid: int,
        block_name: str,
        duration: Optional[float] = None,
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


@dataclass
class TaskInfo:
    task: QoalaTask
    predecessors: Set[int]
    ext_predecessors: Set[int]
    successors: Set[int]
    deadline: Optional[int]
    rel_deadlines: Dict[int, int]
    ext_rel_deadlines: Dict[int, int]
    start_time: Optional[float]

    @classmethod
    def only_task(cls, task: QoalaTask) -> TaskInfo:
        return TaskInfo(task, set(), set(), set(), None, {}, {}, None)

    def is_cpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.CPU

    def is_qpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.QPU


@dataclass
class TaskGraph:
    """DAG of Tasks."""

    def __init__(self, tasks: Optional[Dict[int, TaskInfo]] = None) -> None:
        if tasks is None:
            self._tasks: Dict[int, TaskInfo] = {}
        else:
            self._tasks = tasks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskGraph):
            raise NotImplementedError
        return self._tasks == other._tasks

    def __str__(self) -> str:
        return "\n".join(f"{i}: {t}" for i, t in self._tasks.items())

    def add_tasks(self, tasks: List[QoalaTask]) -> None:
        for task in tasks:
            self._tasks[task.task_id] = TaskInfo.only_task(task)

    def add_precedences(self, precedences: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x precedes y (y should execute after x)
        for (x, y) in precedences:
            assert x in self._tasks and y in self._tasks
            self._tasks[y].predecessors.add(x)
            self._tasks[x].successors.add(y)

    def update_successors(self) -> None:
        # Make sure all `successors` of all tinfos match all predecessors
        for tid, tinfo in self.get_tasks().items():
            for pred in tinfo.predecessors:
                pred_tinfo = self.get_tinfo(pred)
                if tid not in pred_tinfo.successors:
                    pred_tinfo.successors.add(tid)

    def add_ext_precedences(self, precedences: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x (which is not in this graph) precedes y
        # (which is in this graph)
        for (x, y) in precedences:
            assert x not in self._tasks and y in self._tasks
            self._tasks[y].ext_predecessors.add(x)

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

    def get_tasks(self) -> Dict[int, TaskInfo]:
        return self._tasks

    def get_tinfo(self, id: int) -> TaskInfo:
        assert id in self._tasks
        return self._tasks[id]

    def task_exists_for_pid(self, pid: int) -> bool:
        for tid, tinfo in self._tasks.items():
            if tinfo.task.pid == pid:
                return True
        return False

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

    def get_tasks_blocked_only_on_external(self) -> List[int]:
        return [
            i
            for i, tinfo in self._tasks.items()
            if len(tinfo.predecessors) == 0 and len(tinfo.ext_predecessors) > 0
        ]

    def get_epr_roots(self, ignore_external: bool = False) -> List[int]:
        roots = self.get_roots(ignore_external)
        return [r for r in roots if self.get_tinfo(r).task.is_epr_task()]

    def get_event_roots(self, ignore_external: bool = False) -> List[int]:
        roots = self.get_roots(ignore_external)
        return [r for r in roots if self.get_tinfo(r).task.is_event_task()]

    def linearize(self) -> List[int]:
        # Returns None if not linear
        if len(self.get_tasks()) == 0:
            return []  # empty graph is linear

        roots = self.get_roots()
        if len(roots) != 1:
            raise RuntimeError("Task Graph cannot be linearized")

        chain: List[int] = [roots[0]]
        for _ in range(len(self._tasks) - 1):
            successors = self.get_tinfo(chain[-1]).successors
            if len(successors) != 1:
                raise RuntimeError("Task Graph cannot be Linearized")
            successor = successors.pop()
            chain.append(successor)
            successors.add(successor)
        return chain

    def remove_task(self, id: int) -> None:
        assert id in self.get_roots(ignore_external=True)
        tinfo = self._tasks.pop(id)

        # Remove precedences of successor tasks
        for succ in tinfo.successors:
            succ_info = self.get_tinfo(succ)
            assert id in succ_info.predecessors
            succ_info.predecessors.remove(id)

        # Change relative deadlines to absolute ones
        for t in self._tasks.values():
            if id in t.rel_deadlines:
                t.deadline = t.rel_deadlines.pop(id)

    def decrease_deadlines(self, amount: int) -> None:
        for tinfo in self._tasks.values():
            if tinfo.deadline is not None:
                tinfo.deadline -= amount

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
        double_cross_preds: Set[int] = set()
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
            i: deepcopy(tinfo)
            for i, tinfo in self._tasks.items()
            if tinfo.task.processor_type == proc_type
        }

        # Precedence constraints.
        # Move predecessor tasks that have been removed to ext_predecessors.
        for tinfo in partial_tasks.values():
            # Keep predecessors if they are still in the graph.
            new_predecessors = {
                pred for pred in tinfo.predecessors if pred in partial_tasks
            }
            # Move others to ext_predecessors.
            new_ext_predecessors = {
                pred for pred in tinfo.predecessors if pred not in partial_tasks
            }
            tinfo.predecessors = new_predecessors
            tinfo.ext_predecessors = new_ext_predecessors
            # Clear successors. Will be filled in at the end of this function.
            tinfo.successors.clear()

        # Precedence constraints for same-processor tasks that used to have a
        # precedence chain of other-processor tasks in between them.
        for tid, tinfo in partial_tasks.items():
            for pred in self.double_cross_predecessors(tid):
                if pred not in tinfo.predecessors:
                    tinfo.predecessors.add(pred)

            # Relative deadlines.
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

        partial_graph = TaskGraph(partial_tasks)
        # Fill in successors by taking opposite of predecessors.
        partial_graph.update_successors()
        return partial_graph


class TaskGraphBuilder:
    @classmethod
    def linear_tasks(cls, tasks: List[QoalaTask]) -> TaskGraph:
        tinfos: List[TaskInfo] = [TaskInfo.only_task(task) for task in tasks]

        for i in range(len(tinfos) - 1):
            t1 = tinfos[i]
            t2 = tinfos[i + 1]
            t2.predecessors.add(t1.task.task_id)

        graph = TaskGraph(tasks={t.task.task_id: t for t in tinfos})
        graph.update_successors()
        return graph

    @classmethod
    def linear_tasks_with_start_times(
        cls, tasks: List[Tuple[QoalaTask, Optional[int]]]
    ) -> TaskGraph:
        tinfos: List[TaskInfo] = []
        for task, start_time in tasks:
            tinfo = TaskInfo.only_task(task)
            tinfo.start_time = start_time
            tinfos.append(tinfo)

        for i in range(len(tinfos) - 1):
            t1 = tinfos[i]
            t2 = tinfos[i + 1]
            t2.predecessors.add(t1.task.task_id)

        graph = TaskGraph(tasks={t.task.task_id: t for t in tinfos})
        graph.update_successors()
        return graph

    @classmethod
    def merge(cls, graphs: List[TaskGraph]) -> TaskGraph:
        merged_tinfos = {}
        for graph in graphs:
            for tid, tinfo in graph.get_tasks().items():
                merged_tinfos[tid] = tinfo

        merged = TaskGraph(merged_tinfos)
        merged.update_successors()
        return merged

    @classmethod
    def merge_linear(cls, graphs: List[TaskGraph]) -> TaskGraph:
        merged_tinfos = {}
        for graph in graphs:
            for tid, tinfo in graph.get_tasks().items():
                merged_tinfos[tid] = deepcopy(tinfo)

        merged = TaskGraph(merged_tinfos)

        for i in range(1, len(graphs)):
            chain1 = graphs[i - 1].linearize()
            chain2 = graphs[i].linearize()
            # Add precedence between last task of graph1 and first task of graph2
            precedence = (chain1[-1], chain2[0])
            merged.add_precedences([precedence])

        merged.update_successors()
        return merged

    @classmethod
    def from_program(
        cls,
        program: QoalaProgram,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        first_task_id: int = 0,
        prog_input: Optional[Dict[str, int]] = None,
    ) -> TaskGraph:
        return QoalaGraphFromProgramBuilder(first_task_id).build(
            program, pid, ehi, network_ehi, prog_input
        )


class TaskDurationEstimator:
    @classmethod
    def lr_duration(cls, ehi: EhiNodeInfo, routine: LocalRoutine) -> float:
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

                if ehi.all_qubit_gate_infos is not None:
                    for gate_info in ehi.all_qubit_gate_infos:
                        max_duration = max(max_duration, gate_info.duration)

                if max_duration != -1:
                    duration += max_duration
                else:
                    raise RuntimeError(
                        f"Gate {type(instr)} not found in EHI. Cannot calculate duration of containing block."
                    )
        return duration


class TaskGraphFromBlockBuilder:
    def __init__(self):
        self._task_id_counter: int = 0

    def unique_id(self) -> int:
        task_id = self._task_id_counter
        self._task_id_counter += 1
        return task_id

    def build(
        self,
        program_instance: ProgramInstance,
        block_index: int,
        network_ehi: Optional[EhiNetworkInfo] = None,
    ) -> TaskGraph:
        graph = TaskGraph()
        block = program_instance.program.blocks[block_index]
        pid = program_instance.pid
        local_routines = program_instance.program.local_routines
        request_routines = program_instance.program.request_routines
        ehi = program_instance.unit_module.info
        prog_input = program_instance.inputs.values

        if block.typ == BasicBlockType.CL:
            if ehi is not None:
                duration = ehi.latencies.host_instr_time * len(block.instructions)
            else:
                duration = None
            task_id = self.unique_id()
            graph.add_tasks([HostLocalTask(task_id, pid, block.name, duration)])
        elif block.typ == BasicBlockType.CC:
            assert len(block.instructions) == 1
            instr = block.instructions[0]
            assert isinstance(instr, ReceiveCMsgOp)

            if ehi is not None:
                duration = ehi.latencies.host_peer_latency
            else:
                duration = None
            task_id = self.unique_id()
            graph.add_tasks([HostEventTask(task_id, pid, block.name, duration)])
        elif block.typ == BasicBlockType.QL:
            assert len(block.instructions) == 1
            instr = block.instructions[0]
            assert isinstance(instr, RunSubroutineOp)
            if ehi is not None:
                local_routine = local_routines[instr.subroutine]
                lr_duration = TaskDurationEstimator.lr_duration(ehi, local_routine)
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
            graph.add_tasks([precall_task])

            lr_id = self.unique_id()
            qputask = LocalRoutineTask(lr_id, pid, block.name, shared_ptr, lr_duration)
            graph.add_tasks([qputask])

            postcall_id = self.unique_id()
            postcall_task = PostCallTask(
                postcall_id, pid, block.name, shared_ptr, post_duration
            )
            graph.add_tasks([postcall_task])

            # LR task should come after precall task
            graph.get_tinfo(lr_id).predecessors.add(precall_id)
            # postcall task should come after LR task
            graph.get_tinfo(postcall_id).predecessors.add(lr_id)
        elif block.typ == BasicBlockType.QC:
            assert len(block.instructions) == 1
            instr = block.instructions[0]
            assert isinstance(instr, RunRequestOp)
            req_routine = request_routines[instr.req_routine]
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
                num_pairs = req_routine.request.num_pairs
                if isinstance(num_pairs, Template):
                    assert prog_input is not None
                    num_pairs = prog_input[num_pairs.name]
                multi_duration = pair_duration * num_pairs
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
            graph.add_tasks([precall_task])

            postcall_id = self.unique_id()
            postcall_task = PostCallTask(
                postcall_id, pid, block.name, shared_ptr, post_duration
            )
            graph.add_tasks([postcall_task])

            if req_routine.callback_type == CallbackType.WAIT_ALL:
                rr_id = self.unique_id()
                rr_task = MultiPairTask(rr_id, pid, shared_ptr, multi_duration)
                graph.add_tasks([rr_task])
                # RR task should come after precall task
                graph.get_tinfo(rr_id).predecessors.add(precall_id)

                if callback is not None:
                    cb_id = self.unique_id()
                    cb_task = MultiPairCallbackTask(
                        cb_id, pid, callback, shared_ptr, cb_duration
                    )
                    graph.add_tasks([cb_task])
                    # callback task should come after RR task
                    graph.get_tinfo(cb_id).predecessors.add(rr_id)
                    # postcall task should come after callback task
                    graph.get_tinfo(postcall_id).predecessors.add(cb_id)
                else:  # no callback
                    # postcall task should come after RR task
                    graph.get_tinfo(postcall_id).predecessors.add(rr_id)

            else:
                assert req_routine.callback_type == CallbackType.SEQUENTIAL

                num_pairs = req_routine.request.num_pairs
                if isinstance(num_pairs, Template):
                    assert prog_input is not None
                    num_pairs = prog_input[num_pairs.name]

                for i in range(num_pairs):
                    rr_pair_id = self.unique_id()
                    rr_pair_task = SinglePairTask(
                        rr_pair_id, pid, i, shared_ptr, pair_duration
                    )
                    graph.add_tasks([rr_pair_task])
                    # RR pair task should come after precall task.
                    # Note: the RR pair tasks do not have precedence
                    # constraints among each other.
                    graph.get_tinfo(rr_pair_id).predecessors.add(precall_id)
                    if callback is not None:
                        pair_cb_id = self.unique_id()
                        pair_cb_task = SinglePairCallbackTask(
                            pair_cb_id, pid, callback, i, shared_ptr, cb_duration
                        )
                        graph.add_tasks([pair_cb_task])
                        # Callback task for pair should come after corresponding
                        # RR pair task. Note: the pair callback tasks do not have
                        # precedence constraints among each other.
                        graph.get_tinfo(pair_cb_id).predecessors.add(rr_pair_id)
                        # postcall task should come after callback task
                        graph.get_tinfo(postcall_id).predecessors.add(pair_cb_id)
                    else:  # no callback
                        # postcall task should come after RR task
                        graph.get_tinfo(postcall_id).predecessors.add(rr_pair_id)
        return graph


class QoalaGraphFromProgramBuilder:
    def __init__(self, first_task_id: int = 0) -> None:
        self._first_task_id = first_task_id
        self._task_id_counter = first_task_id
        self._graph = TaskGraph()
        self._block_to_task_map: Dict[str, int] = {}  # blk name -> task ID

    def unique_id(self) -> int:
        id = self._task_id_counter
        self._task_id_counter += 1
        return id

    def build(
        self,
        program: QoalaProgram,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        prog_input: Optional[Dict[str, int]] = None,
    ) -> TaskGraph:
        prev_block_task_id: Optional[int] = None
        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                task_id = self.unique_id()
                self._graph.add_tasks(
                    [HostLocalTask(task_id, pid, block.name, duration)]
                )
                self._block_to_task_map[block.name] = task_id
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.get_tinfo(task_id).predecessors.add(prev_block_task_id)
                if block.deadlines is not None:
                    for blk, dl in block.deadlines.items():
                        other_task = self._block_to_task_map[blk]
                        self._graph.get_tinfo(task_id).rel_deadlines[other_task] = dl
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
                self._graph.add_tasks(
                    [HostEventTask(task_id, pid, block.name, duration)]
                )
                self._block_to_task_map[block.name] = task_id
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.get_tinfo(task_id).predecessors.add(prev_block_task_id)
                prev_block_task_id = task_id
            elif block.typ == BasicBlockType.QL:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    local_routine = program.local_routines[instr.subroutine]
                    lr_duration = TaskDurationEstimator.lr_duration(ehi, local_routine)
                    pre_duration = ehi.latencies.host_instr_time
                    post_duration = ehi.latencies.host_instr_time
                else:
                    lr_duration = None
                    pre_duration = None
                    post_duration = None

                deadlines: Dict[int, int] = {}  # other task ID -> relative deadline
                if block.deadlines is not None:
                    for blk, dl in block.deadlines.items():
                        other_task = self._block_to_task_map[blk]
                        deadlines[other_task] = dl

                precall_id = self.unique_id()
                # Use a unique "pointer" or identifier which is used at runtime to point
                # to shared data. The PreCallTask will store the lrcall object
                # to this location, such that the LR- and postcall task can
                # access this object using the shared pointer.
                shared_ptr = precall_id  # just use this task id so we know it's unique
                precall_task = PreCallTask(
                    precall_id, pid, block.name, shared_ptr, pre_duration
                )
                self._graph.add_tasks([precall_task])
                for other_task, dl in deadlines.items():
                    self._graph.get_tinfo(precall_id).rel_deadlines[other_task] = dl

                lr_id = self.unique_id()
                qputask = LocalRoutineTask(
                    lr_id, pid, block.name, shared_ptr, lr_duration
                )
                self._graph.add_tasks([qputask])

                postcall_id = self.unique_id()
                postcall_task = PostCallTask(
                    postcall_id, pid, block.name, shared_ptr, post_duration
                )
                self._graph.add_tasks([postcall_task])
                self._block_to_task_map[block.name] = postcall_id

                # LR task should come after precall task
                self._graph.get_tinfo(lr_id).predecessors.add(precall_id)
                # postcall task should come after LR task
                self._graph.get_tinfo(postcall_id).predecessors.add(lr_id)

                # Tasks for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    # First task for this block is precall task.
                    self._graph.get_tinfo(precall_id).predecessors.add(
                        prev_block_task_id
                    )
                # Last task for this block is postcall task.
                prev_block_task_id = postcall_id
            elif block.typ == BasicBlockType.QC:
                precall_id, postcall_id = self._build_from_qc_task_routine_split(
                    program, block, pid, ehi, network_ehi, prog_input
                )
                # Tasks for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    # First task for QC block is precall task.
                    self._graph.get_tinfo(precall_id).predecessors.add(
                        prev_block_task_id
                    )
                # Last task for QC block is postcall task.
                prev_block_task_id = postcall_id  # (not precall_id !)

        self._graph.update_successors()
        return self._graph

    def _build_from_qc_task_routine_split(
        self,
        program: QoalaProgram,
        block: BasicBlock,
        pid: int,
        ehi: Optional[EhiNodeInfo] = None,
        network_ehi: Optional[EhiNetworkInfo] = None,
        prog_input: Optional[Dict[str, int]] = None,
    ) -> Tuple[int, int]:
        """Returns (precall_id, post_call_id)"""
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)
        req_routine = program.request_routines[instr.req_routine]
        callback_name = req_routine.callback

        if ehi is not None:
            # TODO: make more accurate!
            pre_duration = ehi.latencies.host_instr_time
            post_duration = ehi.latencies.host_instr_time
            if callback_name is not None:
                callback = program.local_routines[callback_name]
                cb_duration = TaskDurationEstimator.lr_duration(ehi, callback)
        else:
            pre_duration = None
            post_duration = None
            cb_duration = None

        if network_ehi is not None:
            pair_duration = list(network_ehi.links.values())[0].duration
            num_pairs = req_routine.request.num_pairs
            if isinstance(num_pairs, Template):
                assert prog_input is not None
                num_pairs = prog_input[num_pairs.name]
            multi_duration = pair_duration * num_pairs
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
        self._graph.add_tasks([precall_task])

        postcall_id = self.unique_id()
        postcall_task = PostCallTask(
            postcall_id, pid, block.name, shared_ptr, post_duration
        )
        self._graph.add_tasks([postcall_task])
        self._block_to_task_map[block.name] = postcall_id

        if req_routine.callback_type == CallbackType.WAIT_ALL:
            rr_id = self.unique_id()
            rr_task = MultiPairTask(rr_id, pid, shared_ptr, multi_duration)
            self._graph.add_tasks([rr_task])
            # RR task should come after precall task
            self._graph.get_tinfo(rr_id).predecessors.add(precall_id)

            if callback_name is not None:
                cb_id = self.unique_id()
                cb_task = MultiPairCallbackTask(
                    cb_id, pid, callback_name, shared_ptr, cb_duration
                )
                self._graph.add_tasks([cb_task])
                # callback task should come after RR task
                self._graph.get_tinfo(cb_id).predecessors.add(rr_id)
                # postcall task should come after callback task
                self._graph.get_tinfo(postcall_id).predecessors.add(cb_id)
            else:  # no callback
                # postcall task should come after RR task
                self._graph.get_tinfo(postcall_id).predecessors.add(rr_id)

        else:
            assert req_routine.callback_type == CallbackType.SEQUENTIAL

            num_pairs = req_routine.request.num_pairs
            if isinstance(num_pairs, Template):
                assert prog_input is not None
                num_pairs = prog_input[num_pairs.name]

            for i in range(num_pairs):
                rr_pair_id = self.unique_id()
                rr_pair_task = SinglePairTask(
                    rr_pair_id, pid, i, shared_ptr, pair_duration
                )
                self._graph.add_tasks([rr_pair_task])
                # RR pair task should come after precall task.
                # Note: the RR pair tasks do not have precedence
                # constraints among each other.
                self._graph.get_tinfo(rr_pair_id).predecessors.add(precall_id)
                if callback_name is not None:
                    pair_cb_id = self.unique_id()
                    pair_cb_task = SinglePairCallbackTask(
                        pair_cb_id, pid, callback_name, i, shared_ptr, cb_duration
                    )
                    self._graph.add_tasks([pair_cb_task])
                    # Callback task for pair should come after corresponding
                    # RR pair task. Note: the pair callback tasks do not have
                    # precedence constraints among each other.
                    self._graph.get_tinfo(pair_cb_id).predecessors.add(rr_pair_id)
                    # postcall task should come after callback task
                    self._graph.get_tinfo(postcall_id).predecessors.add(pair_cb_id)
                else:  # no callback
                    # postcall task should come after RR task
                    self._graph.get_tinfo(postcall_id).predecessors.add(rr_pair_id)

        return precall_id, postcall_id
