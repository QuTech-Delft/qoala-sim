from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class ProcessorType(Enum):
    CPU = 0
    QPU = auto()


class QoalaTask:
    """Base class for Qoala tasks."""

    def __init__(
        self,
        task_id: int,
        processor_type: ProcessorType,
        pid: int,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        self._task_id = task_id
        self._processor_type = processor_type
        self._pid = pid
        self._duration = duration
        self._critical_section = critical_section

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(pid={self.pid}, tid={self.task_id})"
        if not self.is_epr_task() and hasattr(self, "block_name"):
            s += f"block={self.block_name}"  # type: ignore
        return s

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

    @property
    def critical_section(self) -> Optional[int]:
        return self._critical_section

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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        self,
        task_id: int,
        pid: int,
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.CPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
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
    deadline_set: bool = False

    @classmethod
    def only_task(cls, task: QoalaTask) -> TaskInfo:
        return TaskInfo(task, set(), set(), set(), None, {}, {}, None)

    def is_cpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.CPU

    def is_qpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.QPU


@dataclass
class TaskGraph:
    """DAG of Tasks.

    Nodes are TaskInfo objects, which point to a Task object and
    optionally to more info like deadlines, successors, etc.
    """

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
        for x, y in precedences:
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
        for x, y in precedences:
            assert x not in self._tasks and y in self._tasks
            self._tasks[y].ext_predecessors.add(x)

    def add_deadlines(self, deadlines: List[Tuple[int, int]]) -> None:
        for x, d in deadlines:
            assert x in self._tasks
            self._tasks[x].deadline = d

    def add_rel_deadlines(self, deadlines: List[Tuple[Tuple[int, int], int]]) -> None:
        # entry ((x, y), d) means
        # task y must start at most time d time units after task x has finished
        for (x, y), d in deadlines:
            assert x in self._tasks and y in self._tasks
            self._tasks[y].rel_deadlines[x] = d

    def add_ext_rel_deadlines(
        self, deadlines: List[Tuple[Tuple[int, int], int]]
    ) -> None:
        # entry ((x, y), d) means
        # task y must start at most time d time units after task x has finished
        for (x, y), d in deadlines:
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
                raise RuntimeError(
                    f"Task Graph cannot be Linearized: number of successors is {len(successors)}"
                )
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
            new_rel_deadlines = {
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
            tinfo.rel_deadlines = new_rel_deadlines

        partial_graph = TaskGraph(partial_tasks)
        # Fill in successors by taking opposite of predecessors.
        partial_graph.update_successors()
        return partial_graph
