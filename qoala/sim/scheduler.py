from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import netsquid as ns
from netqasm.lang.operand import Template
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.lang.ehi import (
    EhiNetworkInfo,
    EhiNetworkSchedule,
    EhiNetworkTimebin,
    EhiNodeInfo,
)
from qoala.lang.hostlang import BasicBlockType, ReceiveCMsgOp
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
from qoala.runtime.task import (
    HostEventTask,
    LocalRoutineTask,
    MultiPairTask,
    ProcessorType,
    QoalaTask,
    SinglePairTask,
    TaskGraph,
    TaskGraphFromBlockBuilder,
    TaskInfo,
)
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


class NodeSchedulerComponent(Component):
    """
    NetSquid component representing for a node scheduler.
    It is used to send messages from the node scheduler to processor schedulers.

    :param name: Name of the component
    :param cpu_scheduler: CPU scheduler that node scheduler will send messages to.
    :param qpu_scheduler: QPU scheduler that node scheduler will send messages to.

    """

    def __init__(
        self,
        name,
        cpu_scheduler: ProcessorScheduler,
        qpu_scheduler: ProcessorScheduler,
        internal_sched_latency: float = 0.0,
    ):
        super().__init__(name=name)
        self.add_ports(["cpu_scheduler_out"])
        self.add_ports(["qpu_scheduler_out"])

        node_sched_to_cpu = ClassicalChannel(
            "node_scheduler_to_cpu_scheduler", delay=internal_sched_latency
        )
        self.cpu_scheduler_out_port.connect(node_sched_to_cpu.ports["send"])
        node_sched_to_cpu.ports["recv"].connect(cpu_scheduler.node_scheduler_in_port)
        node_sched_to_qpu = ClassicalChannel(
            "node_scheduler_to_qpu_scheduler", delay=internal_sched_latency
        )
        self.qpu_scheduler_out_port.connect(node_sched_to_qpu.ports["send"])
        node_sched_to_qpu.ports["recv"].connect(qpu_scheduler.node_scheduler_in_port)

    @property
    def cpu_scheduler_out_port(self) -> Port:
        """
        Port used to send messages to the CPU scheduler.
        """
        return self.ports["cpu_scheduler_out"]

    @property
    def qpu_scheduler_out_port(self) -> Port:
        """
        Port used to send messages to the QPU scheduler.
        """
        return self.ports["qpu_scheduler_out"]

    def send_cpu_scheduler_message(self, msg: Message) -> None:
        """
        Send a message to the CPU scheduler.
        :param msg: Message to send.
        :return: None
        """
        self.cpu_scheduler_out_port.tx_output(msg)

    def send_qpu_scheduler_message(self, msg: Message) -> None:
        """
        Send a message to the QPU scheduler.
        :param msg: Message to send.
        :return: None
        """
        self.qpu_scheduler_out_port.tx_output(msg)


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
        is_predictable: bool = False,
    ) -> None:
        super().__init__(name=f"{node_name}_scheduler")

        self._node_name = node_name
        self.add_signal(SIGNAL_TASK_COMPLETED)

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

        self._current_block_index: Dict[int, int] = {}  # program ID -> block index
        self._task_from_block_builder = TaskGraphFromBlockBuilder()
        self._prog_instance_dependency: Dict[
            int, int
        ] = {}  # program ID -> dependent program ID

        self._last_cpu_task_pid = -1
        self._last_qpu_task_pid = -1

        scheduler_memory = SharedSchedulerMemory()
        netschedule = network_ehi.network_schedule

        self._is_predictable = is_predictable

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

        self._comp = NodeSchedulerComponent(
            f"{node_name}_scheduler",
            self._cpu_scheduler,
            self._qpu_scheduler,
            internal_sched_latency=local_ehi.latencies.internal_sched_latency,
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
        prev_prog_instance_id = -1
        for batch_id, batch in self._batches.items():
            for i, prog_instance in enumerate(batch.instances):
                if remote_pids is not None:
                    remote_pid = remote_pids[batch_id][i]
                else:
                    remote_pid = None
                process = self.create_process(prog_instance, remote_pid)

                self.memmgr.add_process(process)
                self.initialize_process(process)
                self._current_block_index[prog_instance.pid] = 0
                if linear:
                    self._prog_instance_dependency[
                        prog_instance.pid
                    ] = prev_prog_instance_id
                    prev_prog_instance_id = prog_instance.pid
                else:
                    self._prog_instance_dependency[prog_instance.pid] = -1

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

        inputs = process.prog_instance.inputs
        for req in process.prog_instance.program.request_routines.values():
            req.instantiate(inputs.values)

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def start(self) -> None:
        # Processor schedulers start first to ensure that they will start running tasks after they receive the first
        # message from the node scheduler.
        self._cpu_scheduler.start()
        self._qpu_scheduler.start()
        if not self._is_predictable:
            super().start()
            self.schedule_all()

    def stop(self) -> None:
        self._qpu_scheduler.stop()
        self._cpu_scheduler.stop()
        super().stop()

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            if (
                self._last_cpu_task_pid != -1
                and self.is_program_instance_finished(self._last_cpu_task_pid)
            ) or (
                self._last_qpu_task_pid != -1
                and self.is_program_instance_finished(self._last_qpu_task_pid)
            ):
                self.schedule_all()
            ev_expr = self.await_signal(self._cpu_scheduler, SIGNAL_TASK_COMPLETED)
            ev_expr = ev_expr | self.await_signal(
                self._qpu_scheduler, SIGNAL_TASK_COMPLETED
            )
            yield ev_expr

            now = ns.sim_time()

            # Gets the pid of the last finished task at the current time,
            # if there is no task that is finished at the current time, it returns -1
            self._last_cpu_task_pid = self.cpu_scheduler.get_last_finished_task_pid_at(
                now
            )
            # If there is a task that is finished at the current time, assign the next
            if self._last_cpu_task_pid != -1:
                self.schedule_next_for(self._last_cpu_task_pid)

            self._last_qpu_task_pid = self.qpu_scheduler.get_last_finished_task_pid_at(
                now
            )
            # If there is a task that is finished at the current time, assign the next, but we do not need to assign
            # again if it is the same pid as the one that came from CPU
            if (
                self._last_qpu_task_pid != -1
                and self._last_qpu_task_pid != self._last_cpu_task_pid
            ):
                self.schedule_next_for(self._last_qpu_task_pid)

    def schedule_next_for(self, pid: int) -> None:
        """
        Schedule the tasks of the next block for program instance with given pid
        by assigning respective tasks to CPU and QPU schedulers and send a message
        to schedulers for informing them about the newly assigned tasks.

        :param pid: program instance id
        :return: None
        """
        new_cpu_tasks, new_qpu_tasks = self.find_next_tasks_for(pid)

        # If there are new tasks, send a message to schedulers
        # Note that find_next_tasks_for() returns None if there are no new tasks for that processor
        if new_cpu_tasks:
            self._cpu_scheduler.add_tasks(new_cpu_tasks)
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
        if new_qpu_tasks:
            self._qpu_scheduler.add_tasks(new_qpu_tasks)
            self._comp.send_qpu_scheduler_message(Message(-1, -1, "New Task"))

    def schedule_all(self) -> None:
        """
        Schedules the tasks of the next block for each available program instance in the memory manager
        by assigning respective tasks to CPU and QPU schedulers and sends a message
        to schedulers for informing them about the newly assigned tasks.

        This method is responsible for scheduling tasks for each available program instance in the memory manager.
        A program instance is considered available if it meets two conditions:
        1. It is not finished.
        2. It does not have any dependencies on an unfinished program instance
        (it can have such dependencies if the batch of program instances are submitted to run linearly).

        :return: None
        """
        all_new_cpu_tasks: Dict[int, TaskInfo] = {}
        all_new_qpu_tasks: Dict[int, TaskInfo] = {}

        for pid in self.memmgr.get_all_program_ids():
            # If there is a dependency, check if it is finished
            dependency_pid = self._prog_instance_dependency[pid]
            if dependency_pid != -1:
                dep_cur_index = self._current_block_index[dependency_pid]
                dep_block_length = len(
                    self.memmgr.get_process(dependency_pid).prog_instance.program.blocks
                )
                if dep_cur_index < dep_block_length:
                    continue

            # Note that find_next_tasks_for() returns None if there are no new tasks for that processor
            new_cpu_tasks, new_qpu_tasks = self.find_next_tasks_for(pid)
            if new_cpu_tasks:
                all_new_cpu_tasks.update(new_cpu_tasks)
            if new_qpu_tasks:
                all_new_qpu_tasks.update(new_qpu_tasks)

        # If there are new tasks, send a message to schedulers
        if len(all_new_cpu_tasks) > 0:
            self._cpu_scheduler.add_tasks(all_new_cpu_tasks)
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
        if len(all_new_qpu_tasks) > 0:
            self._qpu_scheduler.add_tasks(all_new_qpu_tasks)
            self._comp.send_qpu_scheduler_message(Message(-1, -1, "New Task"))

    def find_next_tasks_for(
        self, pid: int
    ) -> Tuple[Optional[Dict[int, TaskInfo]], Optional[Dict[int, TaskInfo]]]:
        """
        Finds the tasks of the next block for program instance with given pid,
        and returns them as CPU tasks and QPU tasks separately.

        :param pid: The program instance ID for which to find the next tasks.
        :return: A 2-tuple containing the new CPU tasks and new QPU tasks. If no tasks are
             found for the given PID, both elements of the tuple will be set to None.
        """
        new_cpu_tasks: Dict[int, TaskInfo] = {}
        new_qpu_tasks: Dict[int, TaskInfo] = {}

        if (
            pid in self.host.interface.program_instance_jumps
            and self.host.interface.program_instance_jumps[pid] != -1
        ):
            self._current_block_index[pid] = self.host.interface.program_instance_jumps[
                pid
            ]
            self.host.interface.program_instance_jumps[pid] = -1

        current_block_index = self._current_block_index[pid]
        prog_instance = self.memmgr.get_process(pid).prog_instance
        blocks = prog_instance.program.blocks

        is_program_finished = current_block_index >= len(blocks)
        # If program is finished or CPU scheduler has a task for this pid, do not schedule
        # Note that for all block types, there will be tasks for CPU scheduler
        if is_program_finished or self.cpu_scheduler.task_exists_for_pid(pid):
            return None, None

        block = prog_instance.program.blocks[current_block_index]

        # If it is CL or CC it will only have tasks in CPU but others will have tasks in both
        if block.typ in {
            BasicBlockType.CL,
            BasicBlockType.CC,
        }:
            graph = self._task_from_block_builder.build(
                prog_instance, current_block_index, self._network_ehi
            )
            new_cpu_tasks.update(graph.get_tasks())

            self._current_block_index[pid] += 1
            # If scheduler does not have any task send a message to wake it up
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
            return new_cpu_tasks, None
        elif not self.qpu_scheduler.task_exists_for_pid(pid):
            # Note that we know that cpu does not have any tasks for this pid
            graph = self._task_from_block_builder.build(
                prog_instance, current_block_index, self._network_ehi
            )
            new_cpu_tasks.update(graph.partial_graph(ProcessorType.CPU).get_tasks())
            new_qpu_tasks.update(graph.partial_graph(ProcessorType.QPU).get_tasks())

            self._current_block_index[pid] += 1

        return new_cpu_tasks, new_qpu_tasks

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

    def is_program_instance_finished(self, pid: int) -> bool:
        return self._current_block_index[pid] >= len(
            self.memmgr.get_process(pid).prog_instance.program.blocks
        )

    def submit_program_instance(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        process = self.create_process(prog_instance, remote_pid)
        self.memmgr.add_process(process)
        self.initialize_process(process)
        self._current_block_index[prog_instance.pid] = 0
        self._prog_instance_dependency[prog_instance.pid] = -1

    def get_statistics(self) -> SchedulerStatistics:
        return SchedulerStatistics(
            cpu_tasks_executed=self.cpu_scheduler.get_tasks_executed(),
            qpu_tasks_executed=self.qpu_scheduler.get_tasks_executed(),
            cpu_task_starts=self.cpu_scheduler.get_task_starts(),
            qpu_task_starts=self.qpu_scheduler.get_task_starts(),
            cpu_task_ends=self.cpu_scheduler.get_task_ends(),
            qpu_task_ends=self.qpu_scheduler.get_task_ends(),
        )


class ProcessorSchedulerComponent(Component):
    """
    NetSquid component representing for the ProcessorScheduler.
    It is used to receive messages from the node scheduler.

    :param name: Name of the component
    """

    def __init__(self, name):
        super().__init__(name=name)
        self.add_ports(["node_scheduler_in"])

    @property
    def node_scheduler_in_port(self) -> Port:
        """
        Port that the node scheduler uses to send messages to this component.
        """
        return self.ports["node_scheduler_in"]


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

        self._task_graph: TaskGraph = TaskGraph()
        self._finished_tasks: List[int] = []

        self._prog_start_timestamps: Dict[int, float] = {}  # program ID -> start time
        self._prog_end_timestamps: Dict[int, float] = {}  # program ID -> end time

        self._tasks_executed: Dict[int, QoalaTask] = {}
        self._task_starts: Dict[int, float] = {}
        self._task_ends: Dict[int, float] = {}
        self.last_finished_task_pid: Tuple[int, int] = (-1, -1)  # (pid, end_time)

        self._comp = ProcessorSchedulerComponent(name + "_comp")

    @property
    def node_scheduler_in_port(self) -> Port:
        return self._comp.node_scheduler_in_port

    @property
    def driver(self) -> Driver:
        return self._driver

    def upload_task_graph(self, graph: TaskGraph) -> None:
        """
        Sets the given task graph as the current task graph.

        :param graph: The task graph to upload.
        :return: None
        """
        self._task_graph = graph

    # Gets the pid of the last finished task at the current time,
    # if there is no task that is finished at the current time, it returns -1
    def get_last_finished_task_pid_at(self, time: float) -> int:
        """
        Finds the pid of the last finished task at the given time and returns it,
        if there is no task that is finished at the current time, it returns -1

        :param time: The time to check for finished tasks.
        :return: The pid of the last finished task at the given time if such task exists, -1 otherwise.
        """
        if self.last_finished_task_pid[1] == time:
            return self.last_finished_task_pid[0]
        else:
            return -1

    def task_exists_for_pid(self, pid: int) -> bool:
        """
        Checks the current task graph for the existence of a task with the given pid. Returns True if such task exists,
        False otherwise.

        :param pid: The pid to check for.
        :return: True if a task with the given pid exists, False otherwise.
        """
        return self._task_graph.task_exists_for_pid(pid)

    def add_tasks(self, tasks: Dict[int, TaskInfo]) -> None:
        """
        Adds the given tasks to the current task graph.

        :param tasks: The tasks to add.
        :return: None
        """
        self._task_graph.get_tasks().update(tasks)

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


class Status(Enum):
    GRAPH_EMPTY = auto()
    EPR_GEN = auto()
    NEXT_TASK = auto()
    WAITING_OTHER_CORE = auto()
    WAITING_MSG = auto()
    WAITING_START_TIME = auto()
    WAITING_RESOURCES = auto()
    WAITING_TIME_BIN = auto()


@dataclass
class SchedulerStatus:
    status: Set[Status]
    params: Dict[str, Any]


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
        self._status: SchedulerStatus = SchedulerStatus(status=set(), params={})

    @property
    def status(self) -> SchedulerStatus:
        return self._status

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

    def handle_task(self, task_id: int) -> Generator[EventExpression, None, None]:
        assert self._task_graph is not None
        tinfo = self._task_graph.get_tinfo(task_id)
        task = tinfo.task

        self._logger.debug(f"{ns.sim_time()}: {self.name}: checking next task {task}")

        before = ns.sim_time()

        self._logger.info(f"executing task {task}")
        self._task_logger.info(f"start  {task}")
        self._task_starts[task.task_id] = before
        self.record_start_timestamp(task.pid, before)

        # Execute the task
        success = yield from self._driver.handle_task(task)
        if success:
            after = ns.sim_time()

            self.record_end_timestamp(task.pid, after)
            self.last_finished_task_pid = (task.pid, after)
            duration = after - before
            self._task_graph.decrease_deadlines(duration)
            self._task_graph.remove_task(task_id)

            self._finished_tasks.append(task.task_id)
            self.send_signal(SIGNAL_TASK_COMPLETED)
            self._logger.info(f"finished task {task}")
            self._task_logger.info(f"finish {task}")

            self._tasks_executed[task.task_id] = task
            self._task_ends[task.task_id] = after
        else:
            self._task_logger.info("task failed")


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
        assert isinstance(instr.arguments[0], hostlang.IqoalaSingleton)
        csck_id = process.host_mem.read(instr.arguments[0].name)
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

    def update_status(self) -> None:

        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            self._status = SchedulerStatus(status={Status.GRAPH_EMPTY}, params={})
            return

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
            tid: tg.get_tinfo(tid).start_time  # type: ignore
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
        self._task_logger.info(f"wait_for_start: {wait_for_start}")

        ready = [
            tid
            for tid in no_predecessors
            if tid not in event_blocked_on_message and tid not in with_future_start
        ]

        if len(ready) > 0:
            # From the readily executable tasks, choose which one to execute
            with_deadline = [t for t in ready if tg.get_tinfo(t).deadline is not None]
            if not self._use_deadlines:
                with_deadline = []

            to_return: int

            if len(with_deadline) > 0:
                # Sort them by deadline and return the one with the earliest deadline
                deadlines = {t: tg.get_tinfo(t).deadline for t in with_deadline}
                sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
                self._task_logger.info(f"tasks with deadlines: {sorted_by_deadline}")
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
            self._status = SchedulerStatus(
                status={Status.NEXT_TASK}, params={"task_id": to_return}
            )
            return
        else:
            if len(blocked_on_other_core) > 0:
                self._logger.debug("Waiting other core")
                self._task_logger.debug("Waiting other core")
                self._status.status.add(Status.WAITING_OTHER_CORE)
            if len(event_blocked_on_message) > 0:
                self._logger.debug("Waiting message")
                self._task_logger.debug("Waiting message")
                self._status.status.add(Status.WAITING_MSG)
            if wait_for_start is not None:
                _, start = wait_for_start
                self._logger.debug("Waiting Start Time")
                self._task_logger.debug("Waiting Start Time")
                self._status.status.add(Status.WAITING_START_TIME)
                self._status.params["start_time"] = start

            if len(self.status.status) == 0:
                raise RuntimeError

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            self._status = SchedulerStatus(status=set(), params={})
            self.update_external_predcessors()
            self.update_status()
            self._task_logger.debug(f"status: {self.status.status}")
            if Status.NEXT_TASK in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            else:
                ev_expr = self.await_port_input(self.node_scheduler_in_port)
                if Status.WAITING_OTHER_CORE in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._other_scheduler,
                        signal_label=SIGNAL_TASK_COMPLETED,
                    )
                if Status.WAITING_START_TIME in self.status.status:
                    start_time = self.status.params["start_time"]
                    now = ns.sim_time()
                    delta = start_time - now
                    self._schedule_after(delta, EVENT_WAIT)
                    ev_start_time = EventExpression(source=self, event_type=EVENT_WAIT)
                    ev_expr = ev_expr | ev_start_time

                if Status.WAITING_MSG in self.status.status:
                    ev_msg_arrived = self._host_interface.get_evexpr_for_any_msg()

                    ev_expr = ev_msg_arrived | ev_expr
                    yield ev_expr
                    if len(ev_expr.first_term.triggered_events) > 0:
                        # It was "ev_msg_arrived" that triggered.
                        # Need to process this event (flushing potential other messages)
                        yield from self._host_interface.handle_msg_evexpr(
                            ev_expr.first_term
                        )
                else:
                    yield ev_expr


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

    def update_status(self) -> None:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            self._status = SchedulerStatus(status={Status.GRAPH_EMPTY}, params={})
            return

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
            self._status = SchedulerStatus(
                status={Status.EPR_GEN}, params={"task_id": epr_ready[0]}
            )
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
                self._status = SchedulerStatus(
                    status={Status.NEXT_TASK}, params={"task_id": to_return}
                )
            else:
                # No deadlines
                if self._deterministic:
                    index = 0
                else:
                    index = random.randint(0, len(non_epr_ready) - 1)
                to_return = non_epr_ready[index]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
                self._status = SchedulerStatus(
                    status={Status.NEXT_TASK}, params={"task_id": to_return}
                )
        else:
            if len(blocked_on_other_core) > 0:
                self._logger.debug("Waiting other core")
                self._task_logger.debug("Waiting other core")
                self._status.status.add(Status.WAITING_OTHER_CORE)
            if len(blocked_on_resources) > 0:
                self._logger.debug("Waiting resources")
                self._task_logger.debug("Waiting resources")
                self._status.status.add(Status.WAITING_RESOURCES)
            if epr_wait_for_bin is not None:
                self._logger.debug("Waiting time bin")
                self._task_logger.debug("Waiting time bin")
                task_id, delta = epr_wait_for_bin
                self._status.status.add(Status.WAITING_TIME_BIN)
                self._status.params["delta"] = delta

            if len(self.status.status) == 0:
                raise RuntimeError

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            self._status = SchedulerStatus(status=set(), params={})
            self.update_external_predcessors()
            self.update_status()
            self._task_logger.debug(f"status: {self.status.status}")
            if Status.EPR_GEN in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            elif Status.NEXT_TASK in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            else:
                ev_expr = self.await_port_input(self.node_scheduler_in_port)
                if Status.WAITING_OTHER_CORE in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._other_scheduler,
                        signal_label=SIGNAL_TASK_COMPLETED,
                    )
                if Status.WAITING_RESOURCES in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._memmgr,
                        signal_label=SIGNAL_MEMORY_FREED,
                    )
                if Status.WAITING_TIME_BIN in self.status.status:
                    delta = self.status.params["delta"]
                    self._schedule_after(delta, EVENT_WAIT)
                    ev_timebin = EventExpression(source=self, event_type=EVENT_WAIT)
                    ev_expr = ev_expr | ev_timebin
                yield ev_expr
