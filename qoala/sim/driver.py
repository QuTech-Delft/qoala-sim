import logging
from typing import Generator, List, Optional, Tuple

import netsquid as ns
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.hostlang import RunRequestOp, RunSubroutineOp
from qoala.runtime.taskcreator import (
    CpuSchedule,
    CpuTask,
    QpuSchedule,
    QpuTask,
    RoutineType,
    TaskExecutionMode,
)
from qoala.sim.events import EVENT_WAIT
from qoala.sim.host.hostprocessor import HostProcessor
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackprocessor import NetstackProcessor
from qoala.sim.process import IqoalaProcess
from qoala.sim.qnos.qnosprocessor import QnosProcessor
from qoala.util.logging import LogManager


class CpuDriver(Protocol):
    def __init__(
        self,
        node_name: str,
        hostprocessor: HostProcessor,
        memmgr: MemoryManager,
        schedule: Optional[CpuSchedule] = None,
    ) -> None:
        super().__init__(name=f"{node_name}_cpu_driver")

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({node_name})"
        )

        self._hostprocessor = hostprocessor
        self._memmgr = memmgr

        self._task_list: List[Tuple[Optional[float], CpuTask]]

        if schedule is None:
            self._task_list = []
        else:
            self._task_list = schedule.tasks

    def upload_schedule(self, schedule: CpuSchedule) -> None:
        self._task_list.extend(schedule.tasks)

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            try:
                time, task = self._task_list.pop(0)
                if time is not None:
                    now = ns.sim_time()
                    self._logger.debug(
                        f"{ns.sim_time()}: {self.name}: checking next task {task}"
                    )
                    self._logger.debug(f"scheduled for {time}")
                    self._logger.debug(f"waiting for {time - now}...")
                    yield from self.wait(time - now)

                self._logger.debug(
                    f"{ns.sim_time()}: {self.name}: executing task {task}"
                )
                process = self._memmgr.get_process(task.pid)
                yield from self._hostprocessor.assign_block(process, task.block_name)
                self._logger.debug(
                    f"{ns.sim_time()}: {self.name}: finished task {task}"
                )
            except IndexError:
                break


class QpuDriver(Protocol):
    def __init__(
        self,
        node_name: str,
        hostprocessor: HostProcessor,
        qnosprocessor: QnosProcessor,
        netstackprocessor: NetstackProcessor,
        memmgr: MemoryManager,
        tem: TaskExecutionMode = TaskExecutionMode.ROUTINE_ATOMIC,
        schedule: Optional[QpuSchedule] = None,
    ) -> None:
        super().__init__(name=f"{node_name}_qpu_driver")

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({node_name})"
        )

        self._hostprocessor = hostprocessor
        self._qnosprocessor = qnosprocessor
        self._netstackprocessor = netstackprocessor
        self._memmgr = memmgr
        self._tem = tem

        if schedule is None:
            self._task_list: List[Tuple[Optional[float], QpuTask]] = []
        else:
            self._task_list = schedule.tasks

    def upload_schedule(self, schedule: QpuSchedule) -> None:
        self._task_list.extend(schedule.tasks)

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def _handle_lr(self, task: QpuTask) -> Generator[EventExpression, None, None]:
        if self._tem == TaskExecutionMode.ROUTINE_ATOMIC:
            yield from self._handle_atomic_lr(task)
        else:
            raise NotImplementedError

    def _handle_rr(self, task: QpuTask) -> Generator[EventExpression, None, None]:
        if self._tem == TaskExecutionMode.ROUTINE_ATOMIC:
            yield from self._handle_atomic_rr(task)
        else:
            raise NotImplementedError

    def allocate_qubits_for_routine(
        self, process: IqoalaProcess, routine_name: str
    ) -> None:
        # TODO: merge with code in scheduler.py?
        routine = process.get_local_routine(routine_name)
        for virt_id in routine.metadata.qubit_use:
            if self._memmgr.phys_id_for(process.pid, virt_id) is None:
                self._memmgr.allocate(process.pid, virt_id)

    def free_qubits_after_routine(
        self, process: IqoalaProcess, routine_name: str
    ) -> None:
        # TODO: merge with code in scheduler.py?
        routine = process.get_local_routine(routine_name)
        for virt_id in routine.metadata.qubit_use:
            if virt_id not in routine.metadata.qubit_keep:
                self._memmgr.free(process.pid, virt_id)

    def _handle_atomic_lr(
        self, task: QpuTask
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
        self, task: QpuTask
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

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            try:
                time, task = self._task_list.pop(0)
                if time is not None:
                    now = ns.sim_time()
                    self._logger.debug(
                        f"{ns.sim_time()}: {self.name}: checking next task {task}"
                    )
                    self._logger.debug(f"scheduled for {time}")
                    self._logger.debug(f"waiting for {time - now}...")
                    yield from self.wait(time - now)
                self._logger.debug(
                    f"{ns.sim_time()}: {self.name}: executing task {task}"
                )
                if task.routine_type == RoutineType.LOCAL:
                    yield from self._handle_lr(task)
                elif task.routine_type == RoutineType.REQUEST:
                    yield from self._handle_rr(task)
                else:
                    raise RuntimeError
                self._logger.debug(
                    f"{ns.sim_time()}: {self.name}: finished task {task}"
                )
            except IndexError:
                break
