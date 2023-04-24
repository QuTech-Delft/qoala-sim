from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from click import Option
from netqasm.lang.instr import core

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
from qoala.runtime.message import RrCallTuple


class TaskExecutionMode(Enum):
    ROUTINE_ATOMIC = 0
    ROUTINE_SPLIT = auto()


@dataclass(eq=True, frozen=True)
class QoalaTask:
    task_id: int
    # memory: Any  # TODO needed?


@dataclass(eq=True, frozen=True)
class PreCallTask(QoalaTask):
    pid: int
    block_name: str


@dataclass(eq=True, frozen=True)
class PostCallTask(QoalaTask):
    pid: int
    block_name: str
    rrcall: Optional[RrCallTuple]


@dataclass(eq=True, frozen=True)
class SinglePairTask(QoalaTask):
    pid: int
    pair_index: int
    rrcall: Optional[RrCallTuple]


@dataclass(eq=True, frozen=True)
class MultiPairTask(QoalaTask):
    pid: int
    rrcall: Optional[RrCallTuple]


@dataclass(eq=True, frozen=True)
class SinglePairCallbackTask(QoalaTask):
    pid: int
    callback_name: str
    pair_index: int
    rrcall: Optional[RrCallTuple]


@dataclass(eq=True, frozen=True)
class MultiPairCallbackTask(QoalaTask):
    pid: int
    callback_name: str
    rrcall: Optional[RrCallTuple]


@dataclass(eq=True, frozen=True)
class BlockTask(QoalaTask):
    pid: int
    block_name: str
    typ: BasicBlockType
    duration: Optional[float] = None
    max_time: Optional[float] = None
    remote_id: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.block_name} ({self.typ.name}), dur={self.duration}"


@dataclass(eq=True, frozen=True)
class TaskGraph:
    """DAG of Tasks."""

    tasks: Dict[int, QoalaTask]  # "nodes"

    # an entry (x, y) means that x depends on y (x should execute after y)
    # also known as "precedence constraints"
    dependencies: List[Tuple[int, int]]  # "edges"

    @classmethod
    def empty(cls) -> TaskGraph:
        return TaskGraph(tasks={}, dependencies=[])

    def predecessors(self, task_id: int) -> List[int]:
        # Return all (IDs of) tasks that are direct predecessors of the given task (ID)
        return [y for (x, y) in self.dependencies if x == task_id]

    def leaves(self) -> List[int]:
        # Return all (IDs of) tasks that have no predecessors
        return [i for i in self.tasks.keys() if len(self.predecessors(i)) == 0]


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode
        self._task_id_counter = 0
        self._graph = TaskGraph.empty()

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
    ) -> TaskGraph:
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
            self._graph.dependencies.append((i, i - 1))

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
                cputask = BlockTask(task_id, pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = cputask
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.dependencies.append((task_id, prev_block_task_id))
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
                cputask = BlockTask(task_id, pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = cputask
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.dependencies.append((task_id, prev_block_task_id))
                prev_block_task_id = task_id
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
                qputask = BlockTask(pid, block.name, block.typ, duration)
                self._graph.tasks[task_id] = qputask
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    self._graph.dependencies.append((task_id, prev_block_task_id))
                prev_block_task_id = task_id
            elif block.typ == BasicBlockType.QC:
                precall_id, postcall_id = self._build_from_qc_task_routine_split(
                    program, block, pid, network_ehi
                )
                # Task for this block should come after task for previous block
                # (Assuming linear program!)
                if prev_block_task_id is not None:
                    # First task for QC block is precall task.
                    self._graph.dependencies.append((precall_id, prev_block_task_id))
                # Last task for QC block is postcall task.
                prev_block_task_id = postcall_id  # (not precall_id !)

    def _build_from_qc_task_routine_split(
        self,
        program: QoalaProgram,
        block: BasicBlock,
        pid: int,
        network_ehi: Optional[EhiNetworkInfo] = None,
    ) -> Tuple[int, int]:
        """Returns (precall_id, post_call_id)"""
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)
        req_routine = program.request_routines[instr.req_routine]
        callback = req_routine.callback
        if network_ehi is not None:
            # TODO: refactor!!
            epr_time = list(network_ehi.links.values())[0].duration
            duration = epr_time * req_routine.request.num_pairs
        else:
            duration = None

        precall_id = self.unique_id()
        precall_task = PreCallTask(precall_id, pid, block.name)
        self._graph.tasks[precall_id] = precall_task

        # rrcall will be known only at runtime, as a result of precall_task execution
        postcall_id = self.unique_id()
        postcall_task = PostCallTask(postcall_id, pid, block.name, rrcall=None)
        self._graph.tasks[postcall_id] = postcall_task

        if req_routine.callback_type == CallbackType.WAIT_ALL:
            rr_id = self.unique_id()
            rr_task = MultiPairTask(rr_id, pid, rrcall=None)
            self._graph.tasks[rr_id] = rr_task
            # RR task should come after precall task
            self._graph.dependencies.append((rr_id, precall_id))

            if callback is not None:
                cb_id = self.unique_id()
                cb_task = MultiPairCallbackTask(cb_id, pid, callback, None)
                self._graph.tasks[cb_id] = cb_task
                # callback task should come after RR task
                self._graph.dependencies.append((cb_id, rr_id))
                # postcall task should come after callback task
                self._graph.dependencies.append((postcall_id, cb_id))
            else:  # no callback
                # postcall task should come after RR task
                self._graph.dependencies.append((postcall_id, rr_id))

        else:
            assert req_routine.callback_type == CallbackType.SEQUENTIAL
            for i in range(req_routine.request.num_pairs):
                rr_pair_id = self.unique_id()
                rr_pair_task = SinglePairTask(rr_pair_id, pid, i, None)
                self._graph.tasks[rr_pair_id] = rr_pair_task
                # RR pair task should come after precall task.
                # Note: the RR pair tasks do not have precedence
                # constraints among each other.
                self._graph.dependencies.append((rr_pair_id, precall_id))
                if callback is not None:
                    pair_cb_id = self.unique_id()
                    pair_cb_task = SinglePairCallbackTask(
                        pair_cb_id, pid, callback, i, None
                    )
                    self._graph.tasks[pair_cb_id] = pair_cb_task
                    # Callback task for pair should come after corresponding
                    # RR pair task. Note: the pair callback tasks do not have
                    # precedence constraints among each other.
                    self._graph.dependencies.append((pair_cb_id, rr_pair_id))
                    # postcall task should come after callback task
                    self._graph.dependencies.append((postcall_id, pair_cb_id))
                else:  # no callback
                    # postcall task should come after RR task
                    self._graph.dependencies.append((postcall_id, rr_pair_id))

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
