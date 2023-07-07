from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNetworkSchedule, EhiNodeInfo
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
    MultiPairTask,
    ProcessorType,
    QoalaTask,
    SinglePairTask,
    TaskGraph,
    TaskGraphBuilder,
)
from qoala.sim.driver import CpuDriver, Driver, QpuDriver, SharedSchedulerMemory
from qoala.sim.eprsocket import EprSocket
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
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

        cpudriver = CpuDriver(node_name, scheduler_memory, host.processor, memmgr)
        self._cpu_scheduler = EdfScheduler(node_name, cpudriver, netschedule)

        qpudriver = QpuDriver(
            node_name,
            scheduler_memory,
            host.processor,
            qnos.processor,
            netstack.processor,
            memmgr,
        )
        self._qpu_scheduler = EdfScheduler(node_name, qpudriver, netschedule)

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

        if remote_pid is None:
            remote_pid = 42  # TODO: allow this at all?

        csockets: Dict[int, ClassicalSocket] = {}
        for i, remote_name in meta.csockets.items():
            # TODO: check for already existing classical sockets
            csockets[i] = self.host.create_csocket(
                remote_name, prog_instance.pid, remote_pid
            )

        epr_sockets: Dict[int, EprSocket] = {}
        for i, remote_name in meta.epr_sockets.items():
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
                if remote_pids is not None:
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
        driver: Driver,
        network_schedule: Optional[EhiNetworkSchedule] = None,
    ) -> None:
        super().__init__(name=name)
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}_{driver.__class__.__name__}({name})"
        )
        self._task_logger = LogManager.get_task_logger(name)
        self._driver = driver
        self._other_scheduler: Optional[ProcessorScheduler] = None

        self._task_graph: Optional[TaskGraph] = None
        self._finished_tasks: List[int] = []

        self._prog_start_timestamps: Dict[int, float] = {}  # program ID -> start time
        self._prog_end_timestamps: Dict[int, float] = {}  # program ID -> end time

        self._tasks_executed: Dict[int, QoalaTask] = {}
        self._task_starts: Dict[int, float] = {}
        self._task_ends: Dict[int, float] = {}

        self._network_schedule = network_schedule

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


class SchedulerStatus(Enum):
    GRAPH_EMPTY = 0
    EXTERNAL_PREDECESSORS = auto()
    TASK_AVAILABLE = auto()


class EdfScheduler(ProcessorScheduler):
    def __init__(
        self,
        name: str,
        driver: Driver,
        network_schedule: Optional[EhiNetworkSchedule] = None,
    ) -> None:
        super().__init__(name=name, driver=driver, network_schedule=network_schedule)

    def next_task(self) -> Tuple[SchedulerStatus, Optional[int]]:
        if self._task_graph is None or len(self._task_graph.get_tasks()) == 0:
            return SchedulerStatus.GRAPH_EMPTY, None
        tg = self._task_graph
        # Get all tasks without predecessors
        roots = tg.get_roots(ignore_external=True)
        if len(roots) == 0:
            # External predecessor
            return SchedulerStatus.EXTERNAL_PREDECESSORS, None

        # Get all roots with deadlines
        roots_with_deadline = [r for r in roots if tg.get_tinfo(r).deadline]
        if len(roots_with_deadline) > 0:
            # Sort them by deadline and return the one with the earliest deadline
            deadlines = {r: tg.get_tinfo(r).deadline for r in roots_with_deadline}
            sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
            to_return = sorted_by_deadline[0][0]
            self._logger.debug(f"Return task {to_return}")
            return SchedulerStatus.TASK_AVAILABLE, to_return
        else:
            # No deadlines: just return the first in the list
            to_return = roots[0]
            self._logger.debug(f"Return task {to_return}")
            return SchedulerStatus.TASK_AVAILABLE, to_return

    def wait_until(self, start_time: float) -> Generator[EventExpression, None, None]:
        now = ns.sim_time()
        self._logger.debug(f"scheduled for {start_time}")
        delta = start_time - now
        if delta > 0:
            self._logger.debug(f"waiting for {delta}...")
            yield from self.wait(start_time - now)
        else:
            self._logger.warning(
                f"start time is in the past (delta = {delta}), not waiting"
            )

    def wait_for_external_tasks(
        self, ext_pred: List[int]
    ) -> Generator[EventExpression, None, None]:
        self._logger.debug("checking if external predecessors have finished...")
        assert self._other_scheduler is not None
        while not all(self._other_scheduler.has_finished(p) for p in ext_pred):
            # Wait for a signal that the other scheduler completed a task.
            self._logger.debug("waiting for predecessor task to finish...")
            yield self.await_signal(
                sender=self._other_scheduler,
                signal_label=SIGNAL_TASK_COMPLETED,
            )

    def handle_task(self, task_id: int) -> Generator[EventExpression, None, None]:
        assert self._task_graph is not None
        tinfo = self._task_graph.get_tinfo(task_id)
        task = tinfo.task
        start_time = tinfo.start_time
        ext_pred = tinfo.ext_predecessors

        self._logger.debug(f"{ns.sim_time()}: {self.name}: checking next task {task}")

        if start_time is not None:
            yield from self.wait_until(start_time)
        if len(ext_pred) > 0:
            yield from self.wait_for_external_tasks(ext_pred)
        if self._network_schedule is not None and (
            isinstance(task, SinglePairTask) or isinstance(task, MultiPairTask)
        ):
            now = ns.sim_time()
            next_timeslot = self._network_schedule.next_bin(now)
            if next_timeslot > 0:
                self._task_logger.info(
                    f"waiting until next timeslot ({next_timeslot}, (now: {now}))"
                )
                yield from self.wait_until(next_timeslot)

        before = ns.sim_time()

        self._logger.info(f"executing task {task}")
        self._task_logger.info(f"start  {task}")
        self.record_start_timestamp(task.pid, before)
        yield from self._driver.handle_task(task)
        after = ns.sim_time()
        self.record_end_timestamp(task.pid, after)
        duration = after - before
        self._task_graph.decrease_deadlines(duration)
        self._task_graph.remove_task(task_id)

        self._finished_tasks.append(task.task_id)
        self.send_signal(SIGNAL_TASK_COMPLETED)
        self._logger.info(f"finished task {task}")
        self._task_logger.info(f"finish {task}")

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            status, task_id = self.next_task()
            if status == SchedulerStatus.GRAPH_EMPTY:
                break
            elif status == SchedulerStatus.EXTERNAL_PREDECESSORS:
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
            else:
                assert task_id is not None
                yield from self.handle_task(task_id)
