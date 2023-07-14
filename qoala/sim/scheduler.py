from __future__ import annotations

import logging
import random
from inspect import isbuiltin
from typing import Dict, Generator, List, Optional, Set, Tuple

import netsquid as ns
from netqasm.lang.operand import Template
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import (
    EhiNetworkInfo,
    EhiNetworkSchedule,
    EhiNetworkTimebin,
    EhiNodeInfo,
)
from qoala.lang.hostlang import ReceiveCMsgOp
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import (
    BatchInfo,
    BatchResult,
    ProgramBatch,
    ProgramInstance,
    ProgramResult,
)
from qoala.runtime.statistics import SchedulerStatistics
from qoala.runtime.task import (
    HostEventTask,
    LocalRoutineTask,
    MultiPairTask,
    ProcessorType,
    QoalaTask,
    SinglePairTask,
    TaskGraph,
    TaskGraphBuilder,
)
from qoala.sim import eprsocket
from qoala.sim.driver import CpuDriver, Driver, QpuDriver, SharedSchedulerMemory
from qoala.sim.eprsocket import EprSocket
from qoala.sim.events import EVENT_WAIT, SIGNAL_MEMORY_FREED, SIGNAL_TASK_COMPLETED
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.host import Host
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos import Qnos
from qoala.util.logging import LogManager


class NodeScheduler(Protocol):
    def __init__(
        self,
        node_name: str,
        host: Host,
        qnos: Qnos,
        netstack: Netstack,
        memmgr: MemoryManager,
        local_ehi: EhiNodeInfo,
        network_ehi: EhiNetworkInfo,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(name=f"{node_name}_scheduler")

        self._node_name = node_name

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({node_name})"
        )

        self._host = host
        self._qnos = qnos
        self._netstack = netstack
        self._memmgr = memmgr
        self._local_ehi = local_ehi
        self._network_ehi = network_ehi

        self._prog_instance_counter: int = 0
        self._batch_counter: int = 0
        self._batches: Dict[int, ProgramBatch] = {}  # batch ID -> batch
        self._prog_results: Dict[int, ProgramResult] = {}  # program ID -> result
        self._batch_results: Dict[int, BatchResult] = {}  # batch ID -> result

        self._task_counter = 0
        self._task_graph: Optional[TaskGraph] = None
        self._prog_start_timestamps: Dict[int, float] = {}  # program ID -> start time
        self._prog_end_timestamps: Dict[int, float] = {}  # program ID -> end time

        scheduler_memory = SharedSchedulerMemory()
        netschedule = network_ehi.network_schedule

        # TODO: refactor
        node_id = self.host._comp.node_id

        cpudriver = CpuDriver(node_name, scheduler_memory, host.processor, memmgr)
        self._cpu_scheduler = CpuEdfScheduler(
            f"{node_name}_cpu",
            node_id,
            cpudriver,
            memmgr,
            host.interface,
            deterministic,
            use_deadlines,
        )

        qpudriver = QpuDriver(
            node_name,
            scheduler_memory,
            host.processor,
            qnos.processor,
            netstack.processor,
            memmgr,
        )
        self._qpu_scheduler = QpuEdfScheduler(
            f"{node_name}_qpu",
            node_id,
            qpudriver,
            memmgr,
            netschedule,
            deterministic,
            use_deadlines,
        )

        self._cpu_scheduler.set_other_scheduler(self._qpu_scheduler)
        self._qpu_scheduler.set_other_scheduler(self._cpu_scheduler)

    @property
    def host(self) -> Host:
        return self._host

    @property
    def qnos(self) -> Qnos:
        return self._qnos

    @property
    def netstack(self) -> Netstack:
        return self._netstack

    @property
    def memmgr(self) -> MemoryManager:
        return self._memmgr

    @property
    def cpu_scheduler(self) -> ProcessorScheduler:
        return self._cpu_scheduler

    @property
    def qpu_scheduler(self) -> ProcessorScheduler:
        return self._qpu_scheduler

    def submit_batch(self, batch_info: BatchInfo) -> ProgramBatch:
        prog_instances: List[ProgramInstance] = []

        for i in range(batch_info.num_iterations):
            pid = self._prog_instance_counter
            tasks = TaskGraphBuilder.from_program(
                batch_info.program,
                pid,
                self._local_ehi,
                self._network_ehi,
                first_task_id=self._task_counter,
                prog_input=batch_info.inputs[i].values,
            )
            self._task_counter += len(tasks.get_tasks())

            instance = ProgramInstance(
                pid=pid,
                program=batch_info.program,
                inputs=batch_info.inputs[i],
                unit_module=batch_info.unit_module,
                task_graph=tasks,
            )
            self._prog_instance_counter += 1
            prog_instances.append(instance)

        batch = ProgramBatch(
            batch_id=self._batch_counter, info=batch_info, instances=prog_instances
        )
        self._batches[batch.batch_id] = batch
        self._batch_counter += 1
        return batch

    def get_batches(self) -> Dict[int, ProgramBatch]:
        return self._batches

    def create_process(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> QoalaProcess:
        prog_memory = ProgramMemory(prog_instance.pid)
        meta = prog_instance.program.meta

        csockets: Dict[int, ClassicalSocket] = {}
        for i, remote_name in meta.csockets.items():
            assert remote_pid is not None
            # TODO: check for already existing classical sockets
            csockets[i] = self.host.create_csocket(
                remote_name, prog_instance.pid, remote_pid
            )

        epr_sockets: Dict[int, EprSocket] = {}
        for i, remote_name in meta.epr_sockets.items():
            assert remote_pid is not None
            remote_id = self._network_ehi.get_node_id(remote_name)
            # TODO: check for already existing epr sockets
            # TODO: fidelity
            epr_sockets[i] = EprSocket(i, remote_id, prog_instance.pid, remote_pid, 1.0)

        result = ProgramResult(values={})

        return QoalaProcess(
            prog_instance=prog_instance,
            prog_memory=prog_memory,
            csockets=csockets,
            epr_sockets=epr_sockets,
            result=result,
        )

    def create_processes_for_batches(
        self,
        remote_pids: Optional[Dict[int, List[int]]] = None,  # batch ID -> PID list
    ) -> None:
        for batch_id, batch in self._batches.items():
            for i, prog_instance in enumerate(batch.instances):
                if remote_pids is not None and batch_id in remote_pids:
                    remote_pid = remote_pids[batch_id][i]
                else:
                    remote_pid = None
                process = self.create_process(prog_instance, remote_pid)

                self.memmgr.add_process(process)
                self.initialize_process(process)

    def collect_timestamps(self, batch_id: int) -> List[Optional[Tuple[float, float]]]:
        batch = self._batches[batch_id]
        timestamps: List[Optional[Tuple[float, float]]] = []
        for prog_instance in batch.instances:
            process = self.memmgr.get_process(prog_instance.pid)
            cpu_start_end = self.cpu_scheduler.get_timestamps(process.pid)
            if cpu_start_end is None:
                timestamps.append(None)
                continue

            cpu_start, cpu_end = cpu_start_end
            qpu_start_end = self.qpu_scheduler.get_timestamps(process.pid)
            # QPU timestamps could be None (if program did not have any quantum tasks)
            if qpu_start_end is not None:
                qpu_start, qpu_end = qpu_start_end
                start = min(cpu_start, qpu_start)
                end = max(cpu_end, qpu_end)
                timestamps.append((start, end))
            else:
                timestamps.append((cpu_start, cpu_end))
        return timestamps

    def collect_batch_results(self) -> None:
        for batch_id, batch in self._batches.items():
            results: List[ProgramResult] = []
            for prog_instance in batch.instances:
                process = self.memmgr.get_process(prog_instance.pid)
                results.append(process.result)
            timestamps = self.collect_timestamps(batch_id)
            self._batch_results[batch_id] = BatchResult(batch_id, results, timestamps)

    def get_batch_results(self) -> Dict[int, BatchResult]:
        self.collect_batch_results()
        return self._batch_results

    def initialize_process(self, process: QoalaProcess) -> None:
        # Write program inputs to host memory.
        self.host.processor.initialize(process)

        # TODO: rethink how and when Requests are instantiated
        # inputs = process.prog_instance.inputs
        # for req in process.get_all_requests().values():
        #     # TODO: support for other request parameters being templates?
        #     remote_id = req.request.remote_id
        #     if isinstance(remote_id, Template):
        #         req.request.remote_id = inputs.values[remote_id.name]

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def upload_cpu_task_graph(self, graph: TaskGraph) -> None:
        self._cpu_scheduler.upload_task_graph(graph)

    def upload_qpu_task_graph(self, graph: TaskGraph) -> None:
        self._qpu_scheduler.upload_task_graph(graph)

    def upload_task_graph(self, graph: TaskGraph) -> None:
        self._task_graph = graph
        cpu_graph = graph.partial_graph(ProcessorType.CPU)
        qpu_graph = graph.partial_graph(ProcessorType.QPU)
        self._cpu_scheduler.upload_task_graph(cpu_graph)
        self._qpu_scheduler.upload_task_graph(qpu_graph)

    def start(self) -> None:
        super().start()
        self._cpu_scheduler.start()
        self._qpu_scheduler.start()

    def stop(self) -> None:
        self._qpu_scheduler.stop()
        self._cpu_scheduler.stop()
        super().stop()

    def submit_program_instance(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        process = self.create_process(prog_instance, remote_pid)
        self.memmgr.add_process(process)
        self.initialize_process(process)

    def get_tasks_to_schedule(self) -> List[TaskGraph]:
        all_tasks: List[TaskGraph] = []

        for batch in self._batches.values():
            for inst in batch.instances:
                all_tasks.append(inst.task_graph)

        return all_tasks

    def get_tasks_to_schedule_for(self, batch_id: int) -> List[TaskGraph]:
        all_tasks: List[TaskGraph] = []

        batch = self._batches[batch_id]
        for inst in batch.instances:
            all_tasks.append(inst.task_graph)

        return all_tasks

    def get_statistics(self) -> SchedulerStatistics:
        return SchedulerStatistics(
            cpu_tasks_executed=self.cpu_scheduler.get_tasks_executed(),
            qpu_tasks_executed=self.qpu_scheduler.get_tasks_executed(),
            cpu_task_starts=self.cpu_scheduler.get_task_starts(),
            qpu_task_starts=self.qpu_scheduler.get_task_starts(),
            cpu_task_ends=self.cpu_scheduler.get_task_ends(),
            qpu_task_ends=self.qpu_scheduler.get_task_ends(),
        )


class ProcessorScheduler(Protocol):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: Driver,
        memmgr: MemoryManager,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}_{driver.__class__.__name__}({name})"
        )
        self._node_id = node_id
        self._task_logger = LogManager.get_task_logger(name)
        self._driver = driver
        self._other_scheduler: Optional[ProcessorScheduler] = None
        self._memmgr = memmgr
        self._deterministic = deterministic
        self._use_deadlines = use_deadlines

        self._task_graph: Optional[TaskGraph] = None
        self._finished_tasks: List[int] = []

        self._prog_start_timestamps: Dict[int, float] = {}  # program ID -> start time
        self._prog_end_timestamps: Dict[int, float] = {}  # program ID -> end time

        self._tasks_executed: Dict[int, QoalaTask] = {}
        self._task_starts: Dict[int, float] = {}
        self._task_ends: Dict[int, float] = {}

    @property
    def driver(self) -> Driver:
        return self._driver

    def upload_task_graph(self, graph: TaskGraph) -> None:
        self._task_graph = graph

    def has_finished(self, task_id: int) -> bool:
        return task_id in self._finished_tasks

    def set_other_scheduler(self, other: ProcessorScheduler) -> None:
        self._other_scheduler = other

    def record_start_timestamp(self, pid: int, time: float) -> None:
        # Only write start time for first encountered task.
        if pid not in self._prog_start_timestamps:
            self._prog_start_timestamps[pid] = time

    def record_end_timestamp(self, pid: int, time: float) -> None:
        # Overwrite end time for every task. Automatically the last timestamp remains.
        self._prog_end_timestamps[pid] = time

    def get_timestamps(self, pid: int) -> Optional[Tuple[float, float]]:
        if pid not in self._prog_start_timestamps:
            assert pid not in self._prog_end_timestamps
            return None
        assert pid in self._prog_end_timestamps
        return self._prog_start_timestamps[pid], self._prog_end_timestamps[pid]

    def get_tasks_executed(self) -> Dict[int, QoalaTask]:
        return self._tasks_executed

    def get_task_starts(self) -> Dict[int, float]:
        return self._task_starts

    def get_task_ends(self) -> Dict[int, float]:
        return self._task_ends

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr


class SchedulerStatus:
    pass


class StatusGraphEmpty(SchedulerStatus):
    pass


class StatusEprGen(SchedulerStatus):
    def __init__(self, task_id: int) -> None:
        self._task_id = task_id

    @property
    def task_id(self) -> int:
        return self._task_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatusEprGen):
            raise NotImplementedError
        return self.task_id == other.task_id


class StatusNextTask(SchedulerStatus):
    def __init__(self, task_id: int) -> None:
        self._task_id = task_id

    @property
    def task_id(self) -> int:
        return self._task_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatusNextTask):
            raise NotImplementedError
        return self.task_id == other.task_id


class StatusBlockedOnMessage(SchedulerStatus):
    pass


class StatusBlockedOnMessageOrStartTime(SchedulerStatus):
    def __init__(self, start_time: float) -> None:
        self._start_time = start_time

    @property
    def start_time(self) -> float:
        return self._start_time


class StatusBlockedOnOtherCore(SchedulerStatus):
    pass


class StatusBlockedOnOtherCoreOrStartTime(SchedulerStatus):
    def __init__(self, start_time: float) -> None:
        self._start_time = start_time

    @property
    def start_time(self) -> float:
        return self._start_time


class StatusBlockedOnMessageOrOtherCore(SchedulerStatus):
    pass


class StatusBlockedOnMessageOrOtherCoreOrStartTime(SchedulerStatus):
    def __init__(self, start_time: float) -> None:
        self._start_time = start_time

    @property
    def start_time(self) -> float:
        return self._start_time


class StatusBlockedOnResource(SchedulerStatus):
    pass


class StatusBlockedOnResourceOrOtherCore(SchedulerStatus):
    pass


class StatusBlockedOnTimebin(SchedulerStatus):
    def __init__(self, task_id: int, delta: int) -> None:
        self._task_id = task_id
        self._delta = delta

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def delta(self) -> int:
        return self._delta


class StatusBlockedOnOtherCoreOrTimebin(SchedulerStatus):
    def __init__(self, task_id: int, delta: int) -> None:
        self._task_id = task_id
        self._delta = delta

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def delta(self) -> int:
        return self._delta


class StatusBlockedOnResourceOrTimebin(SchedulerStatus):
    def __init__(self, task_id: int, delta: int) -> None:
        self._task_id = task_id
        self._delta = delta

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def delta(self) -> int:
        return self._delta


class StatusBlockedOnOtherCoreorResourceOrTimebin(SchedulerStatus):
    def __init__(self, task_id: int, delta: int) -> None:
        self._task_id = task_id
        self._delta = delta

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def delta(self) -> int:
        return self._delta


class EdfScheduler(ProcessorScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: Driver,
        memmgr: MemoryManager,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )

    def update_external_predcessors(self) -> None:
        if self._other_scheduler is None:
            return
        assert self._task_graph is not None

        tg = self._task_graph
        for r in tg.get_roots(ignore_external=True):
            ext_preds = tg.get_tinfo(r).ext_predecessors
            new_ext_preds = {
                ext for ext in ext_preds if not self._other_scheduler.has_finished(ext)
            }
            tg.get_tinfo(r).ext_predecessors = new_ext_preds

    def handle_task(self, task_id: int) -> Generator[EventExpression, None, bool]:
        assert self._task_graph is not None
        tinfo = self._task_graph.get_tinfo(task_id)
        task = tinfo.task

        self._logger.debug(f"{ns.sim_time()}: {self.name}: checking next task {task}")

        before = ns.sim_time()

        start_time = self._task_graph.get_tinfo(task.task_id).start_time
        is_busy_task = start_time is not None
        self._logger.info(f"executing task {task}")
        if self.name == "bob_cpu" or self.name == "bob_qpu":
            if is_busy_task:
                self._task_logger.warning(
                    f"BUSY start  {task} (start time: {start_time})"
                )
            else:
                self._task_logger.warning(f"start  {task}")
        self._task_starts[task.task_id] = before
        self.record_start_timestamp(task.pid, before)

        # Execute the task
        success = yield from self._driver.handle_task(task)
        if success:
            after = ns.sim_time()

            self.record_end_timestamp(task.pid, after)
            duration = after - before
            self._task_graph.decrease_deadlines(duration)
            self._task_graph.remove_task(task_id)

            self._finished_tasks.append(task.task_id)
            self.send_signal(SIGNAL_TASK_COMPLETED)
            self._logger.info(f"finished task {task}")
            if self.name == "bob_cpu" or self.name == "bob_qpu":
                if is_busy_task:
                    self._task_logger.warning(f"BUSY finish {task}")
                else:
                    self._task_logger.warning(f"finish {task}")

            self._tasks_executed[task.task_id] = task
            self._task_ends[task.task_id] = after
        else:
            self._task_logger.info(f"task failed")


class CpuEdfScheduler(EdfScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: CpuDriver,
        memmgr: MemoryManager,
        host_interface: HostInterface,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )
        self._host_interface = host_interface

    def is_message_available(self, tid: int) -> bool:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        assert isinstance(task, HostEventTask)
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        instr = block.instructions[0]
        assert isinstance(instr, ReceiveCMsgOp)
        assert isinstance(instr.arguments[0], str)
        csck_id = process.host_mem.read(instr.arguments[0])
        csck = process.csockets[csck_id]
        remote_name = csck.remote_name
        remote_pid = csck.remote_pid
        messages = self._host_interface.get_available_messages(remote_name)
        if (task.pid, remote_pid) in messages:
            self._task_logger.debug(f"task {tid} NOT blocked")
            return True
        else:
            self._task_logger.debug(f"task {tid} blocked")
            return False

    def update_status(self) -> SchedulerStatus:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            return StatusGraphEmpty()

        # All tasks that have no predecessors, internal nor external.
        no_predecessors = tg.get_roots()

        # All tasks that have only external predecessors.
        blocked_on_other_core = tg.get_tasks_blocked_only_on_external()

        # All "receive message" tasks without predecessors (internal nor external).
        event_no_predecessors = [
            tid for tid in no_predecessors if tg.get_tinfo(tid).task.is_event_task()
        ]

        event_blocked_on_message = [
            tid for tid in event_no_predecessors if not self.is_message_available(tid)
        ]

        now = ns.sim_time()
        with_future_start: Dict[int, float] = {
            tid: tg.get_tinfo(tid).start_time
            for tid in no_predecessors
            if tg.get_tinfo(tid).start_time is not None
            and tg.get_tinfo(tid).start_time > now
        }
        wait_for_start: Optional[Tuple[int, float]] = None  # (task ID, start time)
        if len(with_future_start) > 0:
            sorted_by_start = sorted(
                with_future_start.items(), key=lambda item: item[1]
            )
            wait_for_start = sorted_by_start[0]

        ready = [
            tid
            for tid in no_predecessors
            if tid not in event_blocked_on_message and tid not in with_future_start
        ]
        for r in ready:
            if tg.get_tinfo(r).start_time is None:
                if not tg.get_tinfo(r).deadline_set:
                    if self.name in ["bob_cpu", "bob_qpu"]:
                        self._task_logger.warning(
                            f"task {r}: set deadline to now ({now})"
                        )
                    tg.get_tinfo(r).deadline = 0
                    tg.get_tinfo(r).deadline_set = True

        if len(ready) > 0:
            # self._task_logger.warning(f"ready tasks: {ready}")
            # From the readily executable tasks, choose which one to execute
            with_deadline = [t for t in ready if tg.get_tinfo(t).deadline is not None]
            if not self._use_deadlines:
                with_deadline = []

            to_return: int

            if len(with_deadline) > 0:
                # Sort them by deadline and return the one with the earliest deadline
                deadlines = {t: tg.get_tinfo(t).deadline for t in with_deadline}
                sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
                if self.name in ["bob_cpu", "bob_qpu"]:
                    self._task_logger.warning(
                        f"tasks with deadlines: {sorted_by_deadline}"
                    )
                to_return = sorted_by_deadline[0][0]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
            else:
                # No deadlines
                if self._deterministic:
                    to_return = ready[0]
                else:
                    index = random.randint(0, len(ready) - 1)
                    to_return = ready[index]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")

            return StatusNextTask(to_return)
        else:
            # No tasks ready to execute. Check what is/are the cause(s).
            if len(blocked_on_other_core) > 0:
                if len(event_blocked_on_message) > 0:
                    if wait_for_start is not None:
                        _, start = wait_for_start
                        return StatusBlockedOnMessageOrOtherCoreOrStartTime(start)
                    else:
                        return StatusBlockedOnMessageOrOtherCore()
                else:
                    if wait_for_start is not None:
                        _, start = wait_for_start
                        return StatusBlockedOnOtherCoreOrStartTime(start)
                    else:
                        return StatusBlockedOnOtherCore()
            else:
                if wait_for_start is not None:
                    _, start = wait_for_start
                    return StatusBlockedOnMessageOrStartTime(start)
                else:
                    return StatusBlockedOnMessage()

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            status = self.update_status()
            self._task_logger.debug(f"status: {status}")
            if isinstance(status, StatusGraphEmpty):
                break
            elif isinstance(status, StatusBlockedOnOtherCore):
                self._task_logger.debug("waiting for TASK_COMPLETED signal")
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._task_logger.debug("got TASK_COMPLETED signal")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnOtherCoreOrStartTime):
                now = ns.sim_time()
                delta = status.start_time - now
                self._task_logger.info(
                    f"Blocked on other core or start time (delta = {delta})"
                )
                ev_other_core = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._schedule_after(delta, EVENT_WAIT)
                ev_start_time = EventExpression(source=self, event_type=EVENT_WAIT)
                yield ev_other_core | ev_start_time
            elif isinstance(status, StatusBlockedOnMessage):
                self._task_logger.debug("blocked, waiting for message...")
                yield from self._host_interface.wait_for_any_msg()
                self._task_logger.debug("message arrived")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnMessageOrStartTime):
                now = ns.sim_time()
                delta = status.start_time - now
                self._task_logger.info(
                    f"Blocked on message or start time (delta = {delta})"
                )
                ev_msg_arrived = self._host_interface.get_evexpr_for_any_msg()

                self._schedule_after(delta, EVENT_WAIT)
                ev_start_time = EventExpression(source=self, event_type=EVENT_WAIT)
                union = ev_msg_arrived | ev_start_time

                yield union
                if len(union.first_term.triggered_events) > 0:
                    # It was "ev_msg_arrived" that triggered.
                    # Need to process this event (flushing potential other messages)
                    yield from self._host_interface.handle_msg_evexpr(union.first_term)

            elif isinstance(status, StatusBlockedOnMessageOrOtherCore):
                self._task_logger.debug("blocked, waiting for message OR other core...")

                ev_msg_arrived = self._host_interface.get_evexpr_for_any_msg()

                ev_other_core = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                union = ev_msg_arrived | ev_other_core

                yield union
                if len(union.first_term.triggered_events) > 0:
                    # It was "ev_msg_arrived" that triggered.
                    # Need to process this event (flushing potential other messages)
                    yield from self._host_interface.handle_msg_evexpr(union.first_term)
            elif isinstance(status, StatusBlockedOnMessageOrOtherCoreOrStartTime):
                now = ns.sim_time()
                delta = status.start_time - now
                self._task_logger.info(
                    f"Blocked on message or other core or start time (delta = {delta})"
                )
                ev_msg_arrived = self._host_interface.get_evexpr_for_any_msg()

                ev_other_core = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._schedule_after(delta, EVENT_WAIT)
                ev_start_time = EventExpression(source=self, event_type=EVENT_WAIT)

                union = ev_msg_arrived | ev_other_core | ev_start_time
                yield union
                if len(union.first_term.triggered_events) > 0:
                    # It was "ev_msg_arrived" that triggered.
                    # Need to process this event (flushing potential other messages)
                    yield from self._host_interface.handle_msg_evexpr(union.first_term)
            elif isinstance(status, StatusNextTask):
                yield from self.handle_task(status.task_id)
                self.update_external_predcessors()


class QpuEdfScheduler(EdfScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: QpuDriver,
        memmgr: MemoryManager,
        network_schedule: Optional[EhiNetworkSchedule] = None,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )
        self._network_schedule = network_schedule

    def timebin_for_task(self, tid: int) -> EhiNetworkTimebin:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        assert isinstance(task, SinglePairTask) or isinstance(task, MultiPairTask)
        drv_mem = self._driver._memory
        rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
        process = self._memmgr.get_process(task.pid)
        routine = process.get_request_routine(rrcall.routine_name)
        request = routine.request
        epr_sck = process.epr_sockets[request.epr_socket_id]
        return EhiNetworkTimebin(
            nodes=frozenset({self._node_id, epr_sck.remote_id}),
            pids={
                self._node_id: epr_sck.local_pid,
                epr_sck.remote_id: epr_sck.remote_pid,
            },
        )

    def are_resources_available(self, tid: int) -> bool:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        self._task_logger.debug(f"check if resources available for task {tid}")
        if isinstance(task, SinglePairTask):
            # TODO: refactor
            drv_mem = self._driver._memory
            rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            routine = process.get_request_routine(rrcall.routine_name)

            # Get virt ID which would be need to be allocated
            virt_id = routine.request.virt_ids.get_id(task.pair_index)
            # Check if virt ID is available by trying to allocate
            # (without actually allocating)
            try:
                self._memmgr.allocate(task.pid, virt_id)
                self._memmgr.free(task.pid, virt_id)
                return True
            except AllocError:
                return False
        elif isinstance(task, MultiPairTask):
            # TODO: refactor
            drv_mem = self._driver._memory
            rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            routine = process.get_request_routine(rrcall.routine_name)

            # Hack to get num_pairs (see comment in hostprocessor.py)
            prog_input = process.prog_instance.inputs.values
            if isinstance(routine.request.num_pairs, Template):
                template_name = routine.request.num_pairs.name
                num_pairs = prog_input[template_name]
            else:
                num_pairs = routine.request.num_pairs

            # Get virt IDs which would be need to be allocated
            virt_ids = [routine.request.virt_ids.get_id(i) for i in range(num_pairs)]
            # Check if virt IDs are available by trying to allocate
            # (without actually allocating)
            try:
                for virt_id in virt_ids:
                    self._memmgr.allocate(task.pid, virt_id)
                    self._memmgr.free(task.pid, virt_id)
                return True
            except AllocError:
                return False
        elif isinstance(task, LocalRoutineTask):
            drv_mem = self._driver._memory
            lrcall = drv_mem.read_shared_lrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            local_routine = process.get_local_routine(lrcall.routine_name)
            virt_ids = local_routine.metadata.qubit_use
            try:
                for virt_id in virt_ids:
                    if self._memmgr.phys_id_for(task.pid, virt_id) is not None:
                        continue  # already allocated
                    self._memmgr.allocate(task.pid, virt_id)
                    self._memmgr.free(task.pid, virt_id)
                return True
            except AllocError:
                return False
        else:
            self._logger.info(
                f"Checking if resources are available for task type {type(task)}, "
                "returning True but no actual check is implemented"
            )
            # NOTE: we assume that callback tasks never allocate any additional
            # resources so they can always return `True` here.
            return True

    def update_status(self) -> SchedulerStatus:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            return StatusGraphEmpty()

        # All tasks that have no predecessors, internal nor external.
        no_predecessors = tg.get_roots()

        # All tasks that have only external predecessors.
        blocked_on_other_core = tg.get_tasks_blocked_only_on_external()

        # All EPR (SinglePair or MultiPair) tasks that have no predecessors,
        # internal nor external.
        epr_no_predecessors = [
            tid for tid in no_predecessors if tg.get_tinfo(tid).task.is_epr_task()
        ]

        # All tasks without predecessors for which not all resources are availables.
        blocked_on_resources = [
            tid for tid in no_predecessors if not self.are_resources_available(tid)
        ]

        # All non-EPR tasks that are ready for execution.
        non_epr_ready = [
            tid
            for tid in no_predecessors
            if tid not in epr_no_predecessors and tid not in blocked_on_resources
        ]

        # All EPR tasks that have no predecessors and are not blocked on resources.
        epr_no_preds_not_blocked = [
            tid for tid in epr_no_predecessors if tid not in blocked_on_resources
        ]

        # All EPR tasks that can be immediately executed.
        epr_ready = []

        # The next EPR task (if any) that is ready to execute but needs to wait for its
        # corresponding time bin.
        epr_wait_for_bin: Optional[Tuple[int, int]] = None  # (task ID, delta)

        time_until_bin: Dict[int, int] = {}  # task ID -> time until bin

        now = ns.sim_time()
        for e in epr_no_preds_not_blocked:
            if self._network_schedule is not None:
                # Find the time until the next netschedule timebin that allows this EPR task.
                bin = self.timebin_for_task(e)
                self._task_logger.info(f"EPR ready: task {e}, bin: {bin}")
                delta = self._network_schedule.next_specific_bin(now, bin)
                time_until_bin[e] = delta
                self._task_logger.info(f"EPR ready: task {e}, delta: {delta}")
                if delta == 0:
                    epr_ready.append(e)
            else:
                # No network schedule: immediate just execute the EPR task
                epr_ready.append(e)

        epr_non_zero_delta = {
            tid: delta for tid, delta in time_until_bin.items() if delta > 0
        }
        if len(epr_non_zero_delta) > 0:
            sorted_by_delta = sorted(
                epr_non_zero_delta.items(), key=lambda item: item[1]
            )
            earliest, delta = sorted_by_delta[0]
            epr_wait_for_bin = (earliest, delta)

        self._task_logger.info(f"epr_wait_for_bin: {epr_wait_for_bin}")

        if len(epr_ready) > 0:
            self._task_logger.info(f"epr_ready: {epr_ready}")
            return StatusEprGen(epr_ready[0])
        elif len(non_epr_ready) > 0:
            with_deadline = [
                t for t in non_epr_ready if tg.get_tinfo(t).deadline is not None
            ]
            if not self._use_deadlines:
                with_deadline = []
            if len(with_deadline) > 0:
                # Sort them by deadline and return the one with the earliest deadline
                deadlines = {t: tg.get_tinfo(t).deadline for t in with_deadline}
                sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
                to_return = sorted_by_deadline[0][0]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
                return StatusNextTask(to_return)
            else:
                # No deadlines
                if self._deterministic:
                    index = 0
                else:
                    index = random.randint(0, len(non_epr_ready) - 1)
                to_return = non_epr_ready[index]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
                return StatusNextTask(to_return)
        else:
            # No tasks ready to execute. Check what is/are the cause(s).
            if len(blocked_on_other_core) > 0:
                if len(blocked_on_resources) > 0:
                    if epr_wait_for_bin is not None:
                        task_id, delta = epr_wait_for_bin
                        return StatusBlockedOnOtherCoreorResourceOrTimebin(
                            task_id, delta
                        )
                    else:
                        return StatusBlockedOnResourceOrOtherCore()
                else:
                    if epr_wait_for_bin is not None:
                        task_id, delta = epr_wait_for_bin
                        return StatusBlockedOnOtherCoreOrTimebin(task_id, delta)
                    else:
                        return StatusBlockedOnOtherCore()
            else:
                if len(blocked_on_resources) > 0:
                    if epr_wait_for_bin is not None:
                        task_id, delta = epr_wait_for_bin
                        return StatusBlockedOnResourceOrTimebin(task_id, delta)
                    else:
                        return StatusBlockedOnResource()
                else:
                    if epr_wait_for_bin is not None:
                        task_id, delta = epr_wait_for_bin
                        return StatusBlockedOnTimebin(task_id, delta)
                    else:
                        raise RuntimeError

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            status = self.update_status()
            if isinstance(status, StatusGraphEmpty):
                self._task_logger.info("graph empty")
                break
            elif isinstance(status, StatusBlockedOnOtherCore):
                self._task_logger.info("waiting for TASK_COMPLETED signal")
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._task_logger.debug("got TASK_COMPLETED signal")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnResource):
                self._task_logger.info(
                    "blocked on resource: waiting for MEMORY_FREED signal"
                )
                yield self.await_signal(
                    sender=self._memmgr,
                    signal_label=SIGNAL_MEMORY_FREED,
                )
                self._task_logger.debug("blocked on resource: got MEMORY_FREED signal")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnResourceOrOtherCore):
                self._task_logger.info(
                    "blocked on resource and other core: "
                    "waiting for MEMORY_FREED signal OR TASK_COMPLETED signal"
                )
                ev_mem_freed = self.await_signal(
                    sender=self._memmgr,
                    signal_label=SIGNAL_MEMORY_FREED,
                )
                ev_task_completed = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                yield ev_mem_freed | ev_task_completed

                self.update_external_predcessors()
            elif isinstance(status, StatusEprGen):
                self._task_logger.info(f"handling EPR task {status.task_id}")
                yield from self.handle_task(status.task_id)
            elif isinstance(status, StatusNextTask):
                self._task_logger.info(f"handling task {status.task_id}")
                yield from self.handle_task(status.task_id)
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnTimebin):
                self._task_logger.info(f"Blocked on timebin (delta = {status.delta})")
                yield from self.wait(status.delta)
            elif isinstance(status, StatusBlockedOnOtherCoreOrTimebin):
                self._task_logger.info(
                    f"Blocked on other core or timebin (delta = {status.delta})"
                )
                self._schedule_after(status.delta, EVENT_WAIT)
                ev_timebin = EventExpression(source=self, event_type=EVENT_WAIT)
                ev_task_completed = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                yield ev_timebin | ev_task_completed
                self._task_logger.info(f"unblocked")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnResourceOrTimebin):
                self._task_logger.info(
                    f"Blocked on resource or timebin (delta = {status.delta})"
                )
                self._schedule_after(status.delta, EVENT_WAIT)
                ev_timebin = EventExpression(source=self, event_type=EVENT_WAIT)
                ev_mem_freed = self.await_signal(
                    sender=self._memmgr,
                    signal_label=SIGNAL_MEMORY_FREED,
                )
                yield ev_timebin | ev_mem_freed
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnOtherCoreorResourceOrTimebin):
                self._task_logger.info(
                    f"Blocked on other core or resource or timebin (delta = {status.delta})"
                )
                ev_task_completed = self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._schedule_after(status.delta, EVENT_WAIT)
                ev_timebin = EventExpression(source=self, event_type=EVENT_WAIT)
                ev_mem_freed = self.await_signal(
                    sender=self._memmgr,
                    signal_label=SIGNAL_MEMORY_FREED,
                )
                yield ev_task_completed | ev_timebin | ev_mem_freed
                self.update_external_predcessors()
            else:
                raise RuntimeError()
