from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Dict, Generator, List, Optional

import netsquid as ns
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.hostlang import BasicBlockType, RunRequestOp, RunSubroutineOp
from qoala.runtime.message import LrCallTuple, RrCallTuple
from qoala.runtime.schedule import StaticSchedule, StaticScheduleEntry
from qoala.runtime.task import (
    BlockTask,
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
    TaskExecutionMode,
)
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.host.hostprocessor import HostProcessor
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackprocessor import NetstackProcessor
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos.qnosprocessor import QnosProcessor
from qoala.util.logging import LogManager


class Driver(Protocol):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({name})"
        )

        self._other_driver: Optional[Driver] = None
        self._task_list: List[StaticScheduleEntry] = []

        self._finished_tasks: List[QoalaTask] = []

    def set_other_driver(self, other: Driver) -> None:
        self._other_driver = other

    def upload_schedule(self, schedule: StaticSchedule) -> None:
        self._task_list.extend(schedule.entries)

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            try:
                entry = self._task_list.pop(0)
                time = entry.timestamp
                task = entry.task
                prev = entry.prev
                if time is not None:
                    now = ns.sim_time()
                    self._logger.debug(
                        f"{ns.sim_time()}: {self.name}: checking next task {task}"
                    )
                    self._logger.debug(f"scheduled for {time}")
                    self._logger.debug(f"waiting for {time - now}...")
                    yield from self.wait(time - now)
                if prev is not None:
                    assert self._other_driver is not None
                    while not all(
                        p in self._other_driver._finished_tasks for p in prev
                    ):
                        # Wait for a signal that the other driver completed a task.
                        yield self.await_signal(
                            sender=self._other_driver,
                            signal_label=SIGNAL_TASK_COMPLETED,
                        )

                self._logger.info(f"executing task {task}")
                yield from self._handle_task(task)
                self._finished_tasks.append(task)
                self.send_signal(SIGNAL_TASK_COMPLETED)
                self._logger.info(f"finished task {task}")
            except IndexError:
                break

    @abstractmethod
    def _handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, None]:
        raise NotImplementedError


class CpuDriver(Driver):
    def __init__(
        self,
        node_name: str,
        hostprocessor: HostProcessor,
        memmgr: MemoryManager,
    ) -> None:
        super().__init__(name=f"{node_name}_cpu_driver")

        self._hostprocessor = hostprocessor
        self._memmgr = memmgr

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

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def _handle_precall_lr(self, task: PreCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        # Let Host setup shared memory.
        lrcall = self._hostprocessor.prepare_lr_call(process, instr)
        # Store the lrcall object in the shared ptr, so other tasks can use it
        self.write_shared_lrcall(task.shared_ptr, lrcall)

    def _handle_postcall_lr(self, task: PostCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        lrcall = self.read_shared_lrcall(task.shared_ptr)
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
        self.write_shared_rrcall(task.shared_ptr, rrcall)

    def _handle_postcall_rr(self, task: PostCallTask) -> None:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)

        rrcall = self.read_shared_rrcall(task.shared_ptr)
        self._hostprocessor.post_rr_call(process, instr, rrcall)

    def _handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, None]:
        if isinstance(task, BlockTask):
            process = self._memmgr.get_process(task.pid)
            yield from self._hostprocessor.assign_block(process, task.block_name)
        elif isinstance(task, HostLocalTask):
            process = self._memmgr.get_process(task.pid)
            yield from self._hostprocessor.assign_block(process, task.block_name)
        elif isinstance(task, HostEventTask):
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
        elif isinstance(task, PostCallTask):
            process = self._memmgr.get_process(task.pid)
            block = process.program.get_block(task.block_name)
            if block.typ == BasicBlockType.QL:
                self._handle_postcall_lr(task)
            else:
                assert block.typ == BasicBlockType.QC
                self._handle_postcall_rr(task)
        else:
            raise NotImplementedError


class QpuDriver(Driver):
    def __init__(
        self,
        node_name: str,
        hostprocessor: HostProcessor,
        qnosprocessor: QnosProcessor,
        netstackprocessor: NetstackProcessor,
        memmgr: MemoryManager,
        tem: TaskExecutionMode = TaskExecutionMode.ROUTINE_ATOMIC,
    ) -> None:
        super().__init__(name=f"{node_name}_qpu_driver")

        self._hostprocessor = hostprocessor
        self._qnosprocessor = qnosprocessor
        self._netstackprocessor = netstackprocessor
        self._memmgr = memmgr
        self._tem = tem

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def _handle_lr(self, task: BlockTask) -> Generator[EventExpression, None, None]:
        if self._tem == TaskExecutionMode.ROUTINE_ATOMIC:
            yield from self._handle_atomic_lr(task)
        else:
            raise NotImplementedError

    def _handle_rr(self, task: BlockTask) -> Generator[EventExpression, None, None]:
        if self._tem == TaskExecutionMode.ROUTINE_ATOMIC:
            yield from self._handle_atomic_rr(task)
        else:
            raise NotImplementedError

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

    def _handle_atomic_lr(
        self, task: BlockTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        # Let Host setup shared memory.
        lrcall = self._hostprocessor.prepare_lr_call(process, instr)
        # Allocate required qubits.
        self.allocate_qubits_for_routine(process, lrcall.routine_name)
        # Execute the routine on Qnos.
        yield from self._qnosprocessor.assign_local_routine(
            process, lrcall.routine_name, lrcall.input_addr, lrcall.result_addr
        )
        # Free qubits that do not need to be kept.
        self.free_qubits_after_routine(process, lrcall.routine_name)
        # Let Host get results from shared memory.
        self._hostprocessor.post_lr_call(process, instr, lrcall)

    def _handle_atomic_rr(
        self, task: BlockTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunRequestOp)

        # Let Host setup shared memory.
        rrcall = self._hostprocessor.prepare_rr_call(process, instr)
        # TODO: refactor this. Bit of a hack to just pass the QnosProcessor around like this!
        yield from self._netstackprocessor.assign_request_routine(
            process, rrcall, self._qnosprocessor
        )
        self._hostprocessor.post_rr_call(process, instr, rrcall)

    def _handle_local_routine(
        self, task: LocalRoutineTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        assert len(block.instructions) == 1
        instr = block.instructions[0]
        assert isinstance(instr, RunSubroutineOp)

        assert self._other_driver is not None
        cpudriver: CpuDriver = self._other_driver
        lrcall: LrCallTuple = cpudriver.read_shared_lrcall(task.shared_ptr)

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
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        assert self._other_driver is not None
        cpudriver: CpuDriver = self._other_driver
        rrcall: RrCallTuple = cpudriver.read_shared_rrcall(task.shared_ptr)

        global_args = process.prog_instance.inputs.values
        self._netstackprocessor.instantiate_routine(process, rrcall, global_args)

        yield from self._netstackprocessor.handle_multi_pair(
            process, rrcall.routine_name
        )

    def _handle_multi_pair_callback(
        self, task: MultiPairCallbackTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        assert self._other_driver is not None
        cpudriver: CpuDriver = self._other_driver
        rrcall: RrCallTuple = cpudriver.read_shared_rrcall(task.shared_ptr)

        yield from self._netstackprocessor.handle_multi_pair_callback(
            process, rrcall.routine_name, self._qnosprocessor
        )

    def _handle_single_pair(
        self, task: SinglePairTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        assert self._other_driver is not None
        cpudriver: CpuDriver = self._other_driver
        rrcall: RrCallTuple = cpudriver.read_shared_rrcall(task.shared_ptr)

        global_args = process.prog_instance.inputs.values
        self._netstackprocessor.instantiate_routine(process, rrcall, global_args)

        yield from self._netstackprocessor.handle_single_pair(
            process, rrcall.routine_name, task.pair_index
        )

    def _handle_single_pair_callback(
        self, task: SinglePairCallbackTask
    ) -> Generator[EventExpression, None, None]:
        process = self._memmgr.get_process(task.pid)

        assert self._other_driver is not None
        cpudriver: CpuDriver = self._other_driver
        rrcall: RrCallTuple = cpudriver.read_shared_rrcall(task.shared_ptr)

        yield from self._netstackprocessor.handle_single_pair_callback(
            process, rrcall.routine_name, self._qnosprocessor, task.pair_index
        )

    def _handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, None]:
        if isinstance(task, BlockTask):
            if task.typ == BasicBlockType.QL:
                yield from self._handle_lr(task)
            elif task.typ == BasicBlockType.QC:
                yield from self._handle_rr(task)
            else:
                raise RuntimeError
        elif isinstance(task, LocalRoutineTask):
            yield from self._handle_local_routine(task)
        elif isinstance(task, MultiPairTask):
            yield from self._handle_multi_pair(task)
        elif isinstance(task, MultiPairCallbackTask):
            yield from self._handle_multi_pair_callback(task)
        elif isinstance(task, SinglePairTask):
            yield from self._handle_single_pair(task)
        elif isinstance(task, SinglePairCallbackTask):
            yield from self._handle_single_pair_callback(task)
        else:
            raise NotImplementedError
