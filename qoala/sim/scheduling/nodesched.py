from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional, Tuple

from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.program import (
    BatchInfo,
    BatchResult,
    ProgramBatch,
    ProgramInstance,
    ProgramResult,
)
from qoala.runtime.statistics import SchedulerStatistics
from qoala.runtime.task import ProcessorType, TaskGraph
from qoala.sim.driver import CpuDriver, QpuDriver, SharedSchedulerMemory
from qoala.sim.eprsocket import EprSocket
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos import Qnos
from qoala.sim.scheduling.cpusched import (
    CpuEdfScheduler,
    CpuFcfsScheduler,
    CpuScheduler,
)
from qoala.sim.scheduling.nodeschedcomp import NodeSchedulerComponent
from qoala.sim.scheduling.nodeschedinterface import NodeSchedulerInterface
from qoala.sim.scheduling.procsched import ProcessorScheduler
from qoala.sim.scheduling.qpusched import QpuScheduler
from qoala.util.logging import LogManager


class NodeScheduler(Protocol):
    """Scheduler of tasks on a node.

    The NodeScheduler has a single task graph, containing tasks to be executed.
    These tasks may be for different programs and program instances, allowing
    concurrent execution of programs and program instances.
    """

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
        fcfs: bool = False,
        prio_epr: bool = False,
    ) -> None:
        super().__init__(name=f"{node_name}_scheduler")

        self._node_name = node_name
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({node_name})"
        )
        self._task_logger = LogManager.get_task_logger(f"{node_name}_NodeScheduler")

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

        self._const_batch: Optional[ProgramBatch] = None

        scheduler_memory = SharedSchedulerMemory()
        netschedule = network_ehi.network_schedule

        # TODO: refactor
        node_id = self.host._comp.node_id

        cpudriver = CpuDriver(node_name, scheduler_memory, host.processor, memmgr)
        cpu_sched_typ = CpuFcfsScheduler if fcfs else CpuEdfScheduler
        self._cpu_scheduler: CpuScheduler = cpu_sched_typ(  # type: ignore
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
            qnos.processor,
            netstack.processor,
            memmgr,
        )
        self._qpu_scheduler = QpuScheduler(
            f"{node_name}_qpu",
            node_id,
            qpudriver,
            memmgr,
            netschedule,
            deterministic,
            use_deadlines,
            prio_epr,
        )

        self._comp = NodeSchedulerComponent(
            f"{node_name}_scheduler",
            self._cpu_scheduler,
            self._qpu_scheduler,
            internal_sched_latency=local_ehi.latencies.internal_sched_latency,
        )
        self._interface = NodeSchedulerInterface(self._comp)

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

            instance = ProgramInstance(
                pid=pid,
                program=batch_info.program,
                inputs=batch_info.inputs[i],
                unit_module=batch_info.unit_module,
            )
            self._prog_instance_counter += 1
            prog_instances.append(instance)

        batch = ProgramBatch(
            batch_id=self._batch_counter, info=batch_info, instances=prog_instances
        )
        self._batches[batch.batch_id] = batch
        self._batch_counter += 1
        return batch

    def submit_const_batch(self, batch_info: BatchInfo) -> ProgramBatch:
        prog_instances: List[ProgramInstance] = []

        for i in range(batch_info.num_iterations):
            pid = self._prog_instance_counter

            instance = ProgramInstance(
                pid=pid,
                program=batch_info.program,
                inputs=batch_info.inputs[i],
                unit_module=batch_info.unit_module,
            )
            self._prog_instance_counter += 1
            prog_instances.append(instance)

        batch = ProgramBatch(
            batch_id=self._batch_counter, info=batch_info, instances=prog_instances
        )
        self._const_batch = batch
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
        linear: bool = False,
    ) -> None:
        raise NotImplementedError()

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

    def get_all_non_const_pids(self) -> List[int]:
        pids = self.memmgr.get_all_program_ids()
        if self._const_batch is None:
            return pids
        else:
            const_pids = [inst.pid for inst in self._const_batch.instances]
            return [pid for pid in pids if pid not in const_pids]

    def is_from_const_batch(self, pid: int) -> bool:
        if self._const_batch is None:
            return False
        # else:
        const_pids = [inst.pid for inst in self._const_batch.instances]
        return pid in const_pids

    def initialize_process(self, process: QoalaProcess) -> None:
        # Write program inputs to host memory.
        self.host.processor.initialize(process)

        inputs = process.prog_instance.inputs
        for req in process.prog_instance.program.request_routines.values():
            req.instantiate(inputs.values)

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def start(self) -> None:
        super().start()
        self._interface.start()

    def stop(self) -> None:
        super().stop()
        self._interface.stop()

    def upload_task_graph(self, graph: TaskGraph) -> None:
        """
        Assigns tasks in the given task graph to the CPU and QPU schedulers.

        :param graph: The task graph to upload.
        :return: None
        """
        self._task_graph = graph
        cpu_graph = graph.partial_graph(ProcessorType.CPU)
        qpu_graph = graph.partial_graph(ProcessorType.QPU)
        self._cpu_scheduler.upload_task_graph(cpu_graph)
        self._qpu_scheduler.upload_task_graph(qpu_graph)

    def submit_program_instance(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        raise NotImplementedError()

    def get_statistics(self) -> SchedulerStatistics:
        return SchedulerStatistics(
            cpu_tasks_executed=self.cpu_scheduler.get_tasks_executed(),
            qpu_tasks_executed=self.qpu_scheduler.get_tasks_executed(),
            cpu_task_starts=self.cpu_scheduler.get_task_starts(),
            qpu_task_starts=self.qpu_scheduler.get_task_starts(),
            cpu_task_ends=self.cpu_scheduler.get_task_ends(),
            qpu_task_ends=self.qpu_scheduler.get_task_ends(),
        )
