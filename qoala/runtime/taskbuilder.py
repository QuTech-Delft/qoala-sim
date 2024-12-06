from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

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
from qoala.runtime.task import (
    HostEventTask,
    HostLocalTask,
    LocalRoutineTask,
    MultiPairCallbackTask,
    MultiPairTask,
    PostCallTask,
    PreCallTask,
    QoalaTask,
    SinglePairCallbackTask,
    SinglePairTask,
    TaskGraph,
    TaskInfo,
)


class TaskGraphBuilder:
    """Convenience methods for creating a task graph."""

    @classmethod
    def linear_tasks(cls, tasks: List[QoalaTask]) -> TaskGraph:
        """Create a task graph that is a 1D chain of the given tasks.
        That is, the tasks given in the list must be executed consecutively."""
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
        """Merge the given task graphs into a single task graph.
        The original task graphs are disjoint from each other in the resulting graph,
        i.e. there are no precedence constraints between tasks across original task graphs.
        A common use case for this function is when one wishes to execute multiple programs
        *concurrently*. For each program, a separate task graph is created (containing the tasks
        for that specific program including their internal precedence constraints).
        Then, these task graphs are merged into one single task graph and given to the node scheduler.
        """
        merged_tinfos = {}
        for graph in graphs:
            for tid, tinfo in graph.get_tasks().items():
                merged_tinfos[tid] = tinfo

        merged = TaskGraph(merged_tinfos)
        merged.update_successors()
        return merged

    @classmethod
    def merge_linear(cls, graphs: List[TaskGraph]) -> TaskGraph:
        """Merge the given task graphs into a single task graph, like in the `merge()` function above,
        but add precedence constraints between the final task to graph G[i] and the first task of graph G[i+1]
        (when calling that the given list of graphs [G1, G2, G3, ...]).
        A common use case for this function is when one wishes to execute multiple programs
        *sequentially*. For each program, a separate task graph is created (containing the tasks
        for that specific program including their internal precedence constraints).
        Then, these task graphs are merged into one single task graph and given to the node scheduler.
        """
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
    def __init__(self) -> None:
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

            if block.deadlines is not None:
                graph.get_tinfo(task_id).deadline = 0
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
            if block.deadlines is not None:
                # TODO: fix this hack
                graph.get_tinfo(task_id).deadline = 0
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

            if block.deadlines is not None:
                # TODO: fix this hack
                graph.get_tinfo(precall_id).deadline = 0
                graph.get_tinfo(lr_id).deadline = 0
                graph.get_tinfo(postcall_id).deadline = 0
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

            if block.deadlines is not None:
                # TODO: fix this hack
                graph.get_tinfo(precall_id).deadline = 0
                graph.get_tinfo(postcall_id).deadline = 0

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
                    if block.deadlines is not None:
                        # TODO: fix this hack
                        graph.get_tinfo(cb_id).deadline = 0
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
                    if block.deadlines is not None:
                        # TODO: fix this hack
                        graph.get_tinfo(rr_pair_id).deadline = 0
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
                        if block.deadlines is not None:
                            # TODO: fix this hack
                            graph.get_tinfo(pair_cb_id).deadline = 0
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
                    # TODO: fix this hack
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
