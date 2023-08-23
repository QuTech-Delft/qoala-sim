from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Dict, Generator

from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.hostlang import BasicBlockType, RunRequestOp, RunSubroutineOp
from qoala.runtime.message import LrCallTuple, RrCallTuple
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
)
from qoala.sim.host.hostprocessor import HostProcessor
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackprocessor import NetstackProcessor
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos.qnosprocessor import QnosProcessor
from qoala.util.logging import LogManager


class SharedSchedulerMemory:
    def __init__(self) -> None:
        # Used to share information between related tasks:
        # specifically the lrcall/rrcall tuples that are shared between
        # precall, postcall, and pair/callback tasks
        # Values are *only* written by PreCall tasks.
        self._shared_lrcalls: Dict[int, LrCallTuple] = {}
        self._shared_rrcalls: Dict[int, RrCallTuple] = {}

    def write_shared_lrcall(self, ptr: int, lrcall: LrCallTuple) -> None:
        self._shared_lrcalls[ptr] = lrcall

    def write_shared_rrcall(self, ptr: int, rrcall: RrCallTuple) -> None:
        self._shared_rrcalls[ptr] = rrcall

    def read_shared_lrcall(self, ptr: int) -> LrCallTuple:
        return self._shared_lrcalls[ptr]

    def read_shared_rrcall(self, ptr: int) -> RrCallTuple:
        return self._shared_rrcalls[ptr]


class Driver(Protocol):
    def __init__(self, name: str, memory: SharedSchedulerMemory) -> None:
        super().__init__(name=name)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({name})"
        )

        self._memory = memory

    @abstractmethod
    def handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, bool]:
        raise NotImplementedError


class CpuDriver(Driver):
    def __init__(
        self,
        node_name: str,
        memory: SharedSchedulerMemory,
        hostprocessor: HostProcessor,
        memmgr: MemoryManager,
    ) -> None:
        super().__init__(name=f"{node_name}_cpu_driver", memory=memory)

        self._hostprocessor = hostprocessor
        self._memmgr = memmgr

    def _handle_precall_lr(self, task: PreCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        # Let Host setup shared memory.
        lrcall = self._hostprocessor.prepare_lr_call(process, instr)
        # Store the lrcall object in the shared ptr, so other tasks can use it
        self._memory.write_shared_lrcall(task.shared_ptr, lrcall)

    def _handle_postcall_lr(self, task: PostCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        lrcall = self._memory.read_shared_lrcall(task.shared_ptr)
        self._hostprocessor.post_lr_call(process, instr, lrcall)

    def _handle_precall_rr(self, task: PreCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)

        # Let Host setup shared memory.
        rrcall = self._hostprocessor.prepare_rr_call(process, instr)
        # Store the lrcall object in the shared ptr, so other tasks can use it
        self._memory.write_shared_rrcall(task.shared_ptr, rrcall)

    def _handle_postcall_rr(self, task: PostCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        rrcall = self._memory.read_shared_rrcall(task.shared_ptr)
        self._hostprocessor.post_rr_call(process, instr, rrcall)

    def handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, bool]:
        if isinstance(task, HostLocalTask) or isinstance(task, HostEventTask):
            process = self._memmgr.get_process(task.pid)
            yield from self._hostprocessor.assign_block(process, task.block_name)
        elif isinstance(task, PreCallTask):
            process = self._memmgr.get_process(task.pid)
            block = process.program.get_block(task.block_name)
            if block.typ == BasicBlockType.QL:
                self._handle_precall_lr(task)
            else:
                assert block.typ == BasicBlockType.QC
                self._handle_precall_rr(task)
            # Simulate processing time of PreCallTask
            # TODO refactor
            latency = self._hostprocessor._latencies.host_instr_time
            yield from self._hostprocessor._interface.wait(latency)
        elif isinstance(task, PostCallTask):
            process = self._memmgr.get_process(task.pid)
            block = process.program.get_block(task.block_name)
            if block.typ == BasicBlockType.QL:
                self._handle_postcall_lr(task)
            else:
                assert block.typ == BasicBlockType.QC
                self._handle_postcall_rr(task)
            # Simulate processing time of PostCallTask
            # TODO refactor
            latency = self._hostprocessor._latencies.host_instr_time
            yield from self._hostprocessor._interface.wait(latency)
        else:
            raise NotImplementedError

        return True


class QpuDriver(Driver):
    def __init__(
        self,
        node_name: str,
        memory: SharedSchedulerMemory,
        hostprocessor: HostProcessor,
        qnosprocessor: QnosProcessor,
        netstackprocessor: NetstackProcessor,
        memmgr: MemoryManager,
    ) -> None:
        super().__init__(name=f"{node_name}_qpu_driver", memory=memory)

        self._hostprocessor = hostprocessor
        self._qnosprocessor = qnosprocessor
        self._netstackprocessor = netstackprocessor
        self._memmgr = memmgr

    def allocate_qubits_for_routine(
        self, process: QoalaProcess, routine_name: str
    ) -> None:
        routine = process.get_local_routine(routine_name)
        for virt_id in routine.metadata.qubit_use:
            if self._memmgr.phys_id_for(process.pid, virt_id) is None:
                self._memmgr.allocate(process.pid, virt_id)

    def free_qubits_after_routine(
        self, process: QoalaProcess, routine_name: str
    ) -> None:
        routine = process.get_local_routine(routine_name)
        for virt_id in routine.metadata.qubit_use:
            if virt_id not in routine.metadata.qubit_keep:
                self._memmgr.free(process.pid, virt_id)

    def _handle_local_routine(
        self, task: LocalRoutineTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        # The corresponding LrCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        lrcall: LrCallTuple = self._memory.read_shared_lrcall(task.shared_ptr)

        # Allocate required qubits.
        self.allocate_qubits_for_routine(process, lrcall.routine_name)
        # Execute the routine on Qnos.
        yield from self._qnosprocessor.assign_local_routine(
            process, lrcall.routine_name, lrcall.input_addr, lrcall.result_addr
        )
        # Free qubits that do not need to be kept.
        self.free_qubits_after_routine(process, lrcall.routine_name)

    def _handle_multi_pair(
        self, task: MultiPairTask
    ) -> Generator[EventExpression, None, bool]:
        process = self._memmgr.get_process(task.pid)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        rrcall: RrCallTuple = self._memory.read_shared_rrcall(task.shared_ptr)

        global_args = process.prog_instance.inputs.values
        self._netstackprocessor.instantiate_routine(process, rrcall, global_args)

        result = yield from self._netstackprocessor.handle_multi_pair(
            process, rrcall.routine_name
        )
        self._logger.info(f"Driver result: {result}")
        return result

    def _handle_multi_pair_callback(
        self, task: MultiPairCallbackTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        rrcall: RrCallTuple = self._memory.read_shared_rrcall(task.shared_ptr)

        yield from self._netstackprocessor.handle_multi_pair_callback(
            process, rrcall.routine_name, self._qnosprocessor
        )

    def _handle_single_pair(
        self, task: SinglePairTask
    ) -> Generator[EventExpression, None, bool]:
        process = self._memmgr.get_process(task.pid)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        rrcall: RrCallTuple = self._memory.read_shared_rrcall(task.shared_ptr)

        global_args = process.prog_instance.inputs.values
        self._netstackprocessor.instantiate_routine(process, rrcall, global_args)

        result = yield from self._netstackprocessor.handle_single_pair(
            process, rrcall.routine_name, task.pair_index
        )
        self._logger.info(f"Driver result: {result}")
        return result

    def _handle_single_pair_callback(
        self, task: SinglePairCallbackTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        # The corresponding PreCallTask must have executed, and it must have written
        # to the sharded scheduler memory.
        rrcall: RrCallTuple = self._memory.read_shared_rrcall(task.shared_ptr)

        yield from self._netstackprocessor.handle_single_pair_callback(
            process, rrcall.routine_name, self._qnosprocessor, task.pair_index
        )

    def handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, bool]:
        if isinstance(task, LocalRoutineTask):
            yield from self._handle_local_routine(task)
        elif isinstance(task, MultiPairTask):
            result = yield from self._handle_multi_pair(task)
            return result
        elif isinstance(task, MultiPairCallbackTask):
            yield from self._handle_multi_pair_callback(task)
        elif isinstance(task, SinglePairTask):
            result = yield from self._handle_single_pair(task)
            return result
        elif isinstance(task, SinglePairCallbackTask):
            yield from self._handle_single_pair_callback(task)
        else:
            raise NotImplementedError
        return True
