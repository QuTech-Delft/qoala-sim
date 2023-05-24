from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.runtime.environment import StaticNetworkInfo
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import (
    BatchInfo,
    BatchResult,
    ProgramBatch,
    ProgramInstance,
    ProgramResult,
)
from qoala.runtime.task import (
    ProcessorType,
    TaskExecutionMode,
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
        static_network_info: StaticNetworkInfo,
        local_ehi: EhiNodeInfo,
        network_ehi: EhiNetworkInfo,
        tem: TaskExecutionMode = TaskExecutionMode.BLOCK,
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
        self._static_network_info = static_network_info
        self._local_ehi = local_ehi
        self._network_ehi = network_ehi
        self._tem = tem

        self._prog_instance_counter: int = 0
        self._batch_counter: int = 0
        self._batches: Dict[int, ProgramBatch] = {}  # batch ID -> batch
        self._prog_results: Dict[int, ProgramResult] = {}  # program ID -> result
        self._batch_results: Dict[int, BatchResult] = {}  # batch ID -> result

        self._task_counter = 0
        self._task_graph: Optional[TaskGraph] = None

        scheduler_memory = SharedSchedulerMemory()
        cpudriver = CpuDriver(node_name, scheduler_memory, host.processor, memmgr)
        self._cpu_scheduler = EdfScheduler(node_name, cpudriver)

        qpudriver = QpuDriver(
            node_name,
            scheduler_memory,
            host.processor,
            qnos.processor,
            netstack.processor,
            memmgr,
        )
        self._qpu_scheduler = EdfScheduler(node_name, qpudriver)

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

    def submit_batch(self, batch_info: BatchInfo) -> None:
        prog_instances: List[ProgramInstance] = []

        network_info = self._static_network_info

        for i in range(batch_info.num_iterations):
            pid = self._prog_instance_counter
            # TODO: allow multiple remote nodes in single program??
            remote_names = list(batch_info.program.meta.csockets.values())
            if len(remote_names) > 0:
                remote_name = list(batch_info.program.meta.csockets.values())[0]
                remote_id = network_info.get_node_id(remote_name)
            else:
                remote_id = None
            if self._tem == TaskExecutionMode.BLOCK:
                tasks = TaskGraphBuilder.from_file_block_tasks(
                    batch_info.program,
                    pid,
                    self._local_ehi,
                    self._network_ehi,
                    remote_id,
                    first_task_id=self._task_counter,
                    prog_input=batch_info.inputs[i].values,
                )
            else:
                tasks = TaskGraphBuilder.from_file(
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

    def get_batches(self) -> Dict[int, ProgramBatch]:
        return self._batches

    def create_process(self, prog_instance: ProgramInstance) -> QoalaProcess:
        prog_memory = ProgramMemory(prog_instance.pid)
        meta = prog_instance.program.meta

        csockets: Dict[int, ClassicalSocket] = {}
        for i, remote_name in meta.csockets.items():
            # TODO: check for already existing epr sockets
            csockets[i] = self.host.create_csocket(remote_name)

        epr_sockets: Dict[int, EprSocket] = {}
        for i, remote_name in meta.epr_sockets.items():
            network_info = self._static_network_info
            remote_id = network_info.get_node_id(remote_name)
            # TODO: check for already existing epr sockets
            # TODO: fidelity
            epr_sockets[i] = EprSocket(i, remote_id, 1.0)

        result = ProgramResult(values={})

        return QoalaProcess(
            prog_instance=prog_instance,
            prog_memory=prog_memory,
            csockets=csockets,
            epr_sockets=epr_sockets,
            result=result,
        )

    def create_processes_for_batches(self) -> None:
        for batch in self._batches.values():
            for prog_instance in batch.instances:
                process = self.create_process(prog_instance)

                self.memmgr.add_process(process)
                self.initialize_process(process)

    def collect_batch_results(self) -> None:
        for batch_id, batch in self._batches.items():
            results: List[ProgramResult] = []
            for prog_instance in batch.instances:
                process = self.memmgr.get_process(prog_instance.pid)
                results.append(process.result)
            self._batch_results[batch_id] = BatchResult(batch_id, results)

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

    def submit_program_instance(self, prog_instance: ProgramInstance) -> None:
        process = self.create_process(prog_instance)
        self.memmgr.add_process(process)
        self.initialize_process(process)

    def get_tasks_to_schedule(self) -> List[TaskGraph]:
        all_tasks: List[TaskGraph] = []

        for batch in self._batches.values():
            for inst in batch.instances:
                all_tasks.append(inst.task_graph)

        return all_tasks


class ProcessorScheduler(Protocol):
    def __init__(self, name: str, driver: Driver) -> None:
        super().__init__(name=name)
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}_{driver.__class__.__name__}({name})"
        )
        self._driver = driver
        self._other_scheduler: Optional[ProcessorScheduler] = None

        self._task_graph: Optional[TaskGraph] = None
        self._finished_tasks: List[int] = []

    @property
    def driver(self) -> Driver:
        return self._driver

    def upload_task_graph(self, graph: TaskGraph) -> None:
        self._task_graph = graph

    def has_finished(self, task_id: int) -> bool:
        return task_id in self._finished_tasks

    def set_other_scheduler(self, other: ProcessorScheduler) -> None:
        self._other_scheduler = other

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr


class SchedulerStatus(Enum):
    GRAPH_EMPTY = 0
    EXTERNAL_PREDECESSORS = auto()
    TASK_AVAILABLE = auto()


class EdfScheduler(ProcessorScheduler):
    def __init__(self, name: str, driver: Driver) -> None:
        super().__init__(name=name, driver=driver)

    def next_task(self) -> Tuple[SchedulerStatus, Optional[int]]:
        if self._task_graph is None or len(self._task_graph.get_tasks()) == 0:
            return SchedulerStatus.GRAPH_EMPTY, None
        tg = self._task_graph
        # Get all tasks without predecessors
        roots = tg.get_roots(ignore_external=True)
        assert len(roots) >= 1

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

    def wait_until_start_time(
        self, start_time: float
    ) -> Generator[EventExpression, None, None]:
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
            yield from self.wait_until_start_time(start_time)
        if len(ext_pred) > 0:
            yield from self.wait_for_external_tasks(ext_pred)

        before = ns.sim_time()

        self._logger.info(f"executing task {task}")
        yield from self._driver.handle_task(task)
        duration = ns.sim_time() - before
        self._task_graph.decrease_deadlines(duration)
        self._task_graph.remove_task(task_id)

        self._finished_tasks.append(task.task_id)
        self.send_signal(SIGNAL_TASK_COMPLETED)
        self._logger.info(f"finished task {task}")

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            status, task_id = self.next_task()
            if status == SchedulerStatus.GRAPH_EMPTY:
                break
            assert task_id is not None
            yield from self.handle_task(task_id)
