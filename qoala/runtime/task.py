from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        self._task_id = task_id
        self._processor_type = processor_type
        self._pid = pid
        self._duration = duration
        self._critical_section = critical_section
        self._block_name = block_name

    def __str__(self) -> str:
        fields = [
            f"task_type={self.__class__.__name__}",
            f"pid={self.pid}",
            f"tid={self.task_id}",
            f"block={self.block_name}",
        ]
        return " ".join(fields)

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
    def block_name(self) -> str:
        return self._block_name

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
            block_name=block_name,
        )

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
            block_name=block_name,
        )

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
            block_name=block_name,
        )
        self._shared_ptr = shared_ptr

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
            block_name=block_name,
        )
        self._shared_ptr = shared_ptr

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
            block_name=block_name,
        )
        self._shared_ptr = shared_ptr

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
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
            block_name=block_name,
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
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
            block_name=block_name,
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
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
            block_name=block_name,
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
        block_name: str,
        duration: Optional[float] = None,
        critical_section: Optional[int] = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            processor_type=ProcessorType.QPU,
            pid=pid,
            duration=duration,
            critical_section=critical_section,
            block_name=block_name,
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


class PrecedenceKind(Enum):
    DEPENDENCY = auto()
    PREDECESSOR = auto()
    PREV_COMM = auto()
    PREV_ENT = auto()


@dataclass
class TaskPrecedences:
    predecessors: Set[int] = field(default_factory=set)
    dependencies: Set[int] = field(default_factory=set)
    prev_comm: int | None = None
    prev_ent: int | None = None

    def is_empty(self) -> bool:
        return (
            len(self.predecessors) == 0
            and len(self.dependencies) == 0
            and self.prev_comm is None
            and self.prev_ent is None
        )

    def get_ids(self) -> Set[int]:
        ids = self.predecessors | self.dependencies

        if self.prev_comm is not None:
            ids.add(self.prev_comm)
        if self.prev_ent is not None:
            ids.add(self.prev_ent)

        return ids


@dataclass
class TaskInfo:
    task: QoalaTask
    precedences: TaskPrecedences
    ext_precedences: TaskPrecedences
    deadline: Optional[int]
    rel_deadlines: Dict[int, int]
    ext_rel_deadlines: Dict[int, int]
    start_time: Optional[float]
    deadline_set: bool = False

    @classmethod
    def only_task(cls, task: QoalaTask) -> TaskInfo:
        return TaskInfo(task, TaskPrecedences(), TaskPrecedences(), None, {}, {}, None)

    def is_cpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.CPU

    def is_qpu_task(self) -> bool:
        return self.task.processor_type == ProcessorType.QPU


@dataclass
class TaskGraph:
    """DAG of Tasks.

    Nodes are TaskInfo objects, which point to a Task object and
    optionally to more info like deadlines, etc.
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

    @property
    def tasks(self) -> Dict[int, TaskInfo]:
        return self._tasks

    def add_tasks(self, tasks: List[QoalaTask]) -> None:
        for task in tasks:
            self._tasks[task.task_id] = TaskInfo.only_task(task)

    def add_dependencies(self, dependencies: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x precedes y (y should execute after x)
        for x, y in dependencies:
            assert x in self._tasks and y in self._tasks
            self._tasks[y].precedences.dependencies.add(x)

    def add_ext_dependencies(self, dependencies: List[Tuple[int, int]]) -> None:
        # an entry (x, y) means that x (which is not in this graph) precedes y
        # (which is in this graph)
        for x, y in dependencies:
            assert x not in self._tasks and y in self._tasks
            self._tasks[y].ext_precedences.dependencies.add(x)

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
        # Return all (IDs of) tasks that have no precedences

        if ignore_external:
            return [
                i for i, tinfo in self._tasks.items() if tinfo.precedences.is_empty()
            ]
        else:
            return [
                i
                for i, tinfo in self._tasks.items()
                if tinfo.precedences.is_empty() and tinfo.ext_precedences.is_empty()
            ]

    def get_leaves(self) -> List[int]:
        # All tasks ID
        all_ids = set(self._tasks)

        # Tasks that appear in any other task's precedences
        referenced_tasks = set()
        for _, task in self._tasks.items():
            p = task.precedences

            referenced_tasks.update(p.predecessors)
            referenced_tasks.update(p.dependencies)

            # prev_comm and prev_ent are single integers (or maybe None/invalid values)
            # Ensure they are valid task IDs before adding
            for related_id in (p.prev_comm, p.prev_ent):
                if related_id in all_ids:
                    referenced_tasks.add(related_id)

        # Leaves = those not referenced in anyone else's precedences
        return list(all_ids - referenced_tasks)

    def get_tasks_blocked_only_on_external(self) -> List[int]:
        return [
            i
            for i, tinfo in self._tasks.items()
            if tinfo.precedences.is_empty() and not tinfo.ext_precedences.is_empty()
        ]

    def get_epr_roots(self, ignore_external: bool = False) -> List[int]:
        roots = self.get_roots(ignore_external)
        return [r for r in roots if self.get_tinfo(r).task.is_epr_task()]

    def get_event_roots(self, ignore_external: bool = False) -> List[int]:
        roots = self.get_roots(ignore_external)
        return [r for r in roots if self.get_tinfo(r).task.is_event_task()]

    def cancel_task(self, id: int) -> None:
        """Cancel a task regardless of root status.

        Used to remove tasks for non-taken conditional branches in the
        static scheduler.  Unlike ``remove_task``, this preserves
        transitive dependencies: any successor that depended on the
        cancelled task inherits the cancelled task's own dependencies so
        that the dependency chain is not broken.
        """
        if id not in self._tasks:
            return
        cancelled_info = self._tasks.pop(id)
        cp = cancelled_info.precedences
        cep = cancelled_info.ext_precedences

        for succ_info in self._tasks.values():
            p = succ_info.precedences
            if id in p.dependencies:
                p.dependencies.remove(id)
                # Inherit the cancelled task's dependencies (transitive).
                # Only add IDs that still exist in this graph or that the
                # successor can resolve via ext_precedences.
                p.dependencies.update(d for d in cp.dependencies if d in self._tasks)
                succ_info.ext_precedences.dependencies.update(
                    d for d in cp.dependencies if d not in self._tasks
                )
                succ_info.ext_precedences.dependencies.update(cep.dependencies)
            if id in p.predecessors:
                p.predecessors.clear()
            if p.prev_comm == id:
                p.prev_comm = cp.prev_comm
            if p.prev_ent == id:
                p.prev_ent = cp.prev_ent

            ep = succ_info.ext_precedences
            if id in ep.dependencies:
                ep.dependencies.remove(id)
                ep.dependencies.update(cep.dependencies)
                # Also inherit internal deps of cancelled task as ext deps
                ep.dependencies.update(
                    d for d in cp.dependencies if d not in self._tasks
                )
                succ_info.precedences.dependencies.update(
                    d for d in cp.dependencies if d in self._tasks
                )
            if id in ep.predecessors:
                ep.predecessors.clear()
            if ep.prev_comm == id:
                ep.prev_comm = cep.prev_comm
            if ep.prev_ent == id:
                ep.prev_ent = cep.prev_ent

    def remove_task(self, id: int) -> None:
        assert id in self.get_roots(ignore_external=True)
        _ = self._tasks.pop(id)

        # Remove precedences of referenced_tasks tasks
        for succ_id, succ_info in self._tasks.items():
            p = succ_info.precedences

            # Remove from dependencies if present
            if id in p.dependencies:
                p.dependencies.remove(id)

            # Clear predecessors if 'id' is among them because at least one predecessor
            # needs to be executed, not all of them
            if id in p.predecessors:
                p.predecessors.clear()

            # Nullify prev_comm and prev_ent if they reference 'id'
            if p.prev_comm == id:
                p.prev_comm = None
            if p.prev_ent == id:
                p.prev_ent = None

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

    def get_preceding_task_sources(
        self, task_id: int
    ) -> dict[int, set[PrecedenceKind]]:
        p = self.get_tinfo(task_id).precedences
        sources: dict[int, set[PrecedenceKind]] = {}

        for pid in p.predecessors:
            sources.setdefault(pid, set()).add(PrecedenceKind.PREDECESSOR)
        for did in p.dependencies:
            sources.setdefault(did, set()).add(PrecedenceKind.DEPENDENCY)
        if p.prev_comm is not None:
            sources.setdefault(p.prev_comm, set()).add(PrecedenceKind.PREV_COMM)
        if p.prev_ent is not None:
            sources.setdefault(p.prev_ent, set()).add(PrecedenceKind.PREV_ENT)

        return sources

    def cross_precedences(
        self, task_id: int, immediate: bool = True
    ) -> dict[int, set[PrecedenceKind]]:
        # Return all (IDs of) tasks that are precedences that run on
        # the other processor (CPU/QPU).
        # If immediate = False, return all closest such precedence, even if they are
        # no immediate parents.
        # If immediate = True, return only immediate parents with a different processor
        # type.
        # TODO: remove items from result set when they are ancestors of other items
        # in the set (in which case they are redundant)
        proc_type = self.get_tinfo(task_id).task.processor_type
        cross_preds: dict[int, set[PrecedenceKind]] = {}

        for pred, kinds in self.get_preceding_task_sources(task_id).items():
            pred_type = self.get_tinfo(pred).task.processor_type
            if pred_type != proc_type:
                cross_preds.setdefault(pred, set()).update(
                    kinds
                )  # immediate parent of different type
            elif not immediate:
                nested = self.cross_precedences(pred, immediate)
                for nid, n_kinds in nested.items():
                    cross_preds.setdefault(nid, set()).update(n_kinds)

        return cross_preds

    def double_cross_precedences(self, task_id: int) -> dict[int, set[PrecedenceKind]]:
        # Return all (IDs of) tasks that are the closest precedences that run on
        # the same processor (CPU/QPU) but where there are tasks of the other processor
        # type in between (in the precedence chain).

        # For the first step: only check immediate parents that have different type.
        # Parents with same type already induce a normal precedence constraint in the
        # partial graph.
        result: dict[int, set[PrecedenceKind]] = {}
        first_level = self.cross_precedences(task_id, immediate=True)

        for cp in first_level:
            # For each different-type parent, find the nearest ancestor of the original
            # type.
            second_level = self.cross_precedences(cp, immediate=False)
            for tid, kinds in second_level.items():
                result.setdefault(tid, set()).update(kinds)

        return result

    def partial_graph(self, proc_type: ProcessorType) -> TaskGraph:
        # Filter tasks with the correct type.
        partial_tasks: Dict[int, TaskInfo] = {
            i: deepcopy(tinfo)
            for i, tinfo in self._tasks.items()
            if tinfo.task.processor_type == proc_type
        }

        # Precedence constraints.
        # Move precdence tasks that have been removed to ext_precedences.
        for tinfo in partial_tasks.values():
            p = tinfo.precedences

            # Split internal vs. external precedences
            internal_ids = {pred for pred in p.get_ids() if pred in partial_tasks}
            external_ids = p.get_ids() - internal_ids

            # Keep precedences if they are still in the graph.
            new_precedences = TaskPrecedences(
                predecessors=p.predecessors & internal_ids,
                dependencies=p.dependencies & internal_ids,
                prev_comm=p.prev_comm if p.prev_comm in internal_ids else None,
                prev_ent=p.prev_ent if p.prev_ent in internal_ids else None,
            )

            # Move others to ext_precedences.
            new_ext_precedences = TaskPrecedences(
                predecessors=p.predecessors & external_ids,
                dependencies=p.dependencies & external_ids,
                prev_comm=p.prev_comm if p.prev_comm in external_ids else None,
                prev_ent=p.prev_ent if p.prev_ent in external_ids else None,
            )

            tinfo.precedences = new_precedences
            tinfo.ext_precedences = new_ext_precedences

        # Precedence constraints for same-processor tasks that used to have a
        # precedence chain of other-processor tasks in between them.
        for tid, tinfo in partial_tasks.items():
            new_preds = self.double_cross_precedences(tid)
            for pred, kinds in new_preds.items():
                if PrecedenceKind.PREDECESSOR in kinds:
                    tinfo.precedences.predecessors.add(pred)
                if PrecedenceKind.DEPENDENCY in kinds:
                    tinfo.precedences.dependencies.add(pred)
                if PrecedenceKind.PREV_COMM in kinds:
                    tinfo.precedences.prev_comm = pred
                if PrecedenceKind.PREV_ENT in kinds:
                    tinfo.precedences.prev_ent = pred

            # Relative deadlines.
            # Keep rel_deadline to pred if pred is still in the graph.
            new_rel_deadlines = {
                pred: dl
                for pred, dl in tinfo.rel_deadlines.items()
                if pred in partial_tasks
            }
            # Move others to ext_dependencies.
            tinfo.ext_rel_deadlines = {
                pred: dl
                for pred, dl in tinfo.rel_deadlines.items()
                if pred not in partial_tasks
            }
            tinfo.rel_deadlines = new_rel_deadlines

        partial_graph = TaskGraph(partial_tasks)
        return partial_graph
