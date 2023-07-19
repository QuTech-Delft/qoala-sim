from __future__ import annotations

import logging
import random
from typing import Dict, Generator, List, Optional, Set, Tuple

import netsquid as ns
from netqasm.lang.operand import Template
from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNetworkSchedule, EhiNodeInfo
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
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.host import Host
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos import Qnos
from qoala.util.logging import LogManager


class NodeSchedulerComponent(Component):
    def __init__(
        self,
        name,
        cpu_scheduler: ProcessorScheduler,
        qpu_scheduler: ProcessorScheduler,
    ):
        super().__init__(name=name)
        self.add_ports(["cpu_scheduler_out", "cpu_scheduler_in"])
        self.add_ports(["qpu_scheduler_out", "qpu_scheduler_in"])
        self.add_ports(["host_out"])

        self.ports["cpu_scheduler_out"].connect(cpu_scheduler.node_scheduler_out_port)
        self.ports["cpu_scheduler_in"].connect(cpu_scheduler.node_scheduler_in_port)
        self.ports["qpu_scheduler_out"].connect(qpu_scheduler.node_scheduler_out_port)
        self.ports["qpu_scheduler_in"].connect(qpu_scheduler.node_scheduler_in_port)

    @property
    def cpu_scheduler_out_port(self) -> Port:
        return self.ports["cpu_scheduler_out"]

    @property
    def cpu_scheduler_in_port(self) -> Port:
        return self.ports["cpu_scheduler_in"]

    @property
    def qpu_scheduler_out_port(self) -> Port:
        return self.ports["qpu_scheduler_out"]

    @property
    def qpu_scheduler_in_port(self) -> Port:
        return self.ports["qpu_scheduler_in"]

    @property
    def host_out_port(self) -> Port:
        return self.ports["host_out"]

    def send_cpu_scheduler_message(self, msg: Message) -> None:
        self.cpu_scheduler_out_port.tx_output(msg)

    def send_qpu_scheduler_message(self, msg: Message) -> None:
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

        self._current_block_index: Dict[int, int] = {}  # program ID -> block index
        self._task_from_block_builder = TaskGraphFromBlockBuilder()
        self._prog_instance_dependency: Dict[
            int, int
        ] = {}  # program ID -> dependent program ID

        scheduler_memory = SharedSchedulerMemory()
        netschedule = network_ehi.network_schedule

        cpudriver = CpuDriver(node_name, scheduler_memory, host.processor, memmgr)
        self._cpu_scheduler = CpuEdfScheduler(
            f"{node_name}_cpu",
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

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self.schedule()
            ev_expr = self.await_signal(self._cpu_scheduler, SIGNAL_TASK_COMPLETED)
            ev_expr = ev_expr | self.await_signal(
                self._qpu_scheduler, SIGNAL_TASK_COMPLETED
            )
            yield ev_expr

    def schedule(self) -> None:
        new_cpu_tasks: Dict[int, TaskInfo] = {}
        new_qpu_tasks: Dict[int, TaskInfo] = {}

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

            # If there is a jump, jump to that block
            if (
                pid in self.host.interface.program_instance_jumps
                and self.host.interface.program_instance_jumps[pid] != -1
            ):
                self._current_block_index[
                    pid
                ] = self.host.interface.program_instance_jumps[pid]
                self.host.interface.program_instance_jumps[pid] = -1

            current_block_index = self._current_block_index[pid]
            prog_instance = self.memmgr.get_process(pid).prog_instance
            blocks = prog_instance.program.blocks

            # If it is not in range or cpu is not done with tasks with this pid
            # We cannot add any more tasks because all block types have cpu tasks
            if len(
                blocks
            ) <= current_block_index or self.cpu_scheduler.is_task_for_pid_exist(pid):
                continue

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
            elif not self.qpu_scheduler.is_task_for_pid_exist(pid):
                # Note that we know that cpu does not have any tasks for this pid
                graph = self._task_from_block_builder.build(
                    prog_instance, current_block_index, self._network_ehi
                )
                new_cpu_tasks.update(graph.partial_graph(ProcessorType.CPU).get_tasks())
                new_qpu_tasks.update(graph.partial_graph(ProcessorType.QPU).get_tasks())

                self._current_block_index[pid] += 1

            # If scheduler does not have any task send a message to wake it up
            if len(self.cpu_scheduler._task_graph.get_tasks()) == 0:
                self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
            if len(self.qpu_scheduler._task_graph.get_tasks()) == 0:
                self._comp.send_qpu_scheduler_message(Message(-1, -1, "New Task"))
            self._cpu_scheduler.add_tasks(new_cpu_tasks)
            self._qpu_scheduler.add_tasks(new_qpu_tasks)

    def submit_program_instance_new(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        process = self.create_process(prog_instance, remote_pid)
        self.memmgr.add_process(process)
        self.initialize_process(process)
        self._current_block_index[prog_instance.pid] = 0
        self._prog_instance_dependency[prog_instance.pid] = -1

    def submit_program_instance(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        process = self.create_process(prog_instance, remote_pid)
        self.memmgr.add_process(process)
        self.initialize_process(process)

    def get_tasks_to_schedule(self) -> List[TaskGraph]:
        all_tasks: List[TaskGraph] = []

        # for batch in self._batches.values():
        #     for inst in batch.instances:
        #         all_tasks.append(inst.task_graph)

        return all_tasks

    def get_tasks_to_schedule_for(self, batch_id: int) -> List[TaskGraph]:
        all_tasks: List[TaskGraph] = []

        # batch = self._batches[batch_id]
        # for inst in batch.instances:
        #     all_tasks.append(inst.task_graph)

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


class ProcessorSchedulerComponent(Component):
    def __init__(self, name):
        super().__init__(name=name)
        self.add_ports(["node_scheduler_out", "node_scheduler_in"])

    @property
    def node_scheduler_out_port(self) -> Port:
        return self.ports["node_scheduler_out"]

    @property
    def node_scheduler_in_port(self) -> Port:
        return self.ports["node_scheduler_in"]


class ProcessorScheduler(Protocol):
    def __init__(
        self,
        name: str,
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

        self._comp = ProcessorSchedulerComponent(name + "_comp")

    @property
    def node_scheduler_out_port(self) -> Port:
        return self._comp.node_scheduler_out_port

    @property
    def node_scheduler_in_port(self) -> Port:
        return self._comp.node_scheduler_in_port

    @property
    def driver(self) -> Driver:
        return self._driver

    def is_task_for_pid_exist(self, pid: int) -> bool:
        return self._task_graph.is_task_for_pid_exist(pid)

    def add_tasks(self, tasks: Dict[int, TaskInfo]) -> None:
        self._task_graph.get_tasks().update(tasks)

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
    def __init__(self, task_ids: List[int]) -> None:
        self._task_ids = task_ids

    @property
    def task_ids(self) -> List[int]:
        return self._task_ids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatusEprGen):
            raise NotImplementedError
        return self.task_ids == other.task_ids


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


class StatusBlockedOnOtherCore(SchedulerStatus):
    pass


class StatusBlockedOnResource(SchedulerStatus):
    pass


class EdfScheduler(ProcessorScheduler):
    def __init__(
        self,
        name: str,
        driver: Driver,
        memmgr: MemoryManager,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )

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
        self, ext_pred: Set[int]
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
        start_time = tinfo.start_time
        ext_pred = tinfo.ext_predecessors

        self._logger.debug(f"{ns.sim_time()}: {self.name}: checking next task {task}")

        if start_time is not None:
            yield from self.wait_until(start_time)
        if len(ext_pred) > 0:
            yield from self.wait_for_external_tasks(ext_pred)
        yield from self.wait_for_timeslot(task)

        before = ns.sim_time()

        self._logger.info(f"executing task {task}")
        self._task_logger.info(f"start  {task}")
        self._task_starts[task.task_id] = before
        self.record_start_timestamp(task.pid, before)

        # Execute the task
        print(f"executing task {task}", self.name)
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

        if self.name == "bob_qpu" and isinstance(task, LocalRoutineTask):
            self._task_logger.warning(f"finish {task}")

        self._tasks_executed[task.task_id] = task
        self._task_ends[task.task_id] = after

    def wait_for_timeslot(self, task) -> Generator[EventExpression, None, None]:
        raise NotImplementedError


class CpuEdfScheduler(EdfScheduler):
    def __init__(
        self,
        name: str,
        driver: CpuDriver,
        memmgr: MemoryManager,
        host_interface: HostInterface,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
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

        # Get all "receive message" tasks without predecessors
        event_roots = tg.get_event_roots(ignore_external=False)
        self._task_logger.debug(f"event roots: {event_roots}")

        blocked_event_roots: List[int] = []
        blocked_on_message = False
        for tid in event_roots:
            if not self.is_message_available(tid):
                blocked_event_roots.append(tid)
                blocked_on_message = True

        # if at least one "receive message" task is blocked on actually receiving
        # the message, `blocked_on_message` is True

        roots = tg.get_roots(ignore_external=False)

        # filter out blocked event roots
        roots = [r for r in roots if r not in blocked_event_roots]
        self._task_logger.debug(f"roots: {roots}")
        self._task_logger.debug(f"blocked event roots: {blocked_event_roots}")

        # `roots` now contains all tasks that can be immediately executed

        if len(roots) == 0:
            # No tasks exist that can be immediately executed; check the reason for this.

            if blocked_on_message:
                # There is at least one task blocked on receiving a message.
                self._task_logger.debug("blocked on message")
                return StatusBlockedOnMessage()
            else:
                internal_roots = tg.get_roots(ignore_external=True)
                self._task_logger.debug(f"all intenral roots: {internal_roots}")
                internal_roots = [
                    r for r in internal_roots if r not in blocked_event_roots
                ]
                self._task_logger.debug(f"non-blocked intenral roots: {internal_roots}")
                self._task_logger.debug("only roots with external dependencies")

                # otherwise the graph must be empty (already checked above)
                assert len(internal_roots) > 0

                return StatusBlockedOnOtherCore()

                # TODO: handle case where we are blocked on both (message or other core)
                # and we don't know which one will unblock us first
                # (now we assume it's always receiving a message)

        with_ext_preds = [
            tid for tid in roots if len(tg.get_tinfo(tid).ext_predecessors) > 0
        ]
        self._task_logger.debug(f"with ext preds: {with_ext_preds}")

        if blocked_on_message:
            self._task_logger.warning(
                "blocked on message but there is a local task to execute"
            )

        # From the readily executable tasks, choose which one to execute
        with_deadline = [t for t in roots if tg.get_tinfo(t).deadline is not None]
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
                to_return = roots[0]
            else:
                index = random.randint(0, len(roots) - 1)
                to_return = roots[index]
            self._logger.debug(f"Return task {to_return}")
            self._task_logger.debug(f"Return task {to_return}")

        return StatusNextTask(to_return)

    def wait_for_timeslot(self, task) -> Generator[EventExpression, None, None]:
        return None
        yield

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            status = self.update_status()
            self._task_logger.debug(f"status: {status}")
            if isinstance(status, StatusGraphEmpty):
                yield self.await_port_input(self.node_scheduler_out_port)
            elif isinstance(status, StatusBlockedOnOtherCore):
                self._task_logger.debug("waiting for TASK_COMPLETED signal")
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._task_logger.debug("got TASK_COMPLETED signal")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnMessage):
                self._task_logger.debug("blocked, waitingn for message...")
                yield from self._host_interface.wait_for_any_msg()
                self._task_logger.debug("message arrived")
                self.update_external_predcessors()
            elif isinstance(status, StatusNextTask):
                yield from self.handle_task(status.task_id)
                self.update_external_predcessors()


class QpuEdfScheduler(EdfScheduler):
    def __init__(
        self,
        name: str,
        driver: QpuDriver,
        memmgr: MemoryManager,
        network_schedule: Optional[EhiNetworkSchedule] = None,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )
        self._network_schedule = network_schedule

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
        raise RuntimeError

    def update_status(self) -> SchedulerStatus:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            return StatusGraphEmpty()

        epr_roots = tg.get_epr_roots(ignore_external=False)
        self._task_logger.debug(f"epr roots: {epr_roots}")
        blocked_epr_roots: List[int] = []
        available_epr_roots: List[int] = []
        blocked_on_resource = False
        for er in epr_roots:
            if self.are_resources_available(er):
                available_epr_roots.append(er)
            else:
                blocked_epr_roots.append(er)
                blocked_on_resource = True
        self._task_logger.debug(f"available epr roots: {available_epr_roots}")
        if len(available_epr_roots) > 0:
            return StatusEprGen(available_epr_roots)

        roots = tg.get_roots(ignore_external=False)
        # Filter out blocked tasks
        roots = [r for r in roots if r not in blocked_epr_roots]
        self._task_logger.debug(f"roots: {roots}")

        if len(roots) == 0:
            internal_roots = tg.get_roots(ignore_external=True)
            self._task_logger.debug(f"internal roots: {internal_roots}")
            if len(internal_roots) == 0:
                assert blocked_on_resource
                self._task_logger.debug("blocked on resource")
                return StatusBlockedOnResource()
            else:
                self._task_logger.debug("only roots with external dependencies")
                return StatusBlockedOnOtherCore()

        blocked_roots: List[int] = []
        blocked_on_resource = False
        for r in roots:
            task = tg.get_tinfo(r).task
            # NOTE: Callback Tasks are not checked since they are assumed to never
            # allocate extra qubits !
            if isinstance(task, LocalRoutineTask):
                self._task_logger.debug("checking resources")
                if not self.are_resources_available(task.task_id):
                    self._task_logger.debug("blocked on resource")
                    blocked_roots.append(task.task_id)
                    blocked_on_resource = True

        roots = [r for r in roots if r not in blocked_roots]
        if len(roots) == 0:
            assert blocked_on_resource is True
            return StatusBlockedOnResource()

        # `roots` contains the tasks we can choose from
        with_deadline = [t for t in roots if tg.get_tinfo(t).deadline is not None]
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
                index = random.randint(0, len(roots) - 1)
            to_return = roots[index]
            self._logger.debug(f"Return task {to_return}")
            self._task_logger.debug(f"Return task {to_return}")
            return StatusNextTask(to_return)

    def wait_for_timeslot(self, task) -> Generator[EventExpression, None, None]:
        if self._network_schedule is not None and (
            isinstance(task, SinglePairTask) or isinstance(task, MultiPairTask)
        ):
            now = ns.sim_time()
            next_timeslot = self._network_schedule.next_bin(now)
            if next_timeslot - now > 0:
                self._task_logger.debug(
                    f"waiting until next timeslot ({next_timeslot}, (now: {now}))"
                )
                yield from self.wait_until(next_timeslot)

    def handle_epr_gen(
        self, task_ids: List[int]
    ) -> Generator[EventExpression, None, None]:
        assert self._task_graph is not None
        tg = self._task_graph

        tasks_to_send: Dict[int, QoalaTask] = {}  # PID -> task
        # NOTE: we only allow one task per PID !!
        for tid in task_ids:
            pid = tg.get_tinfo(tid).task.pid
            tasks_to_send[pid] = tg.get_tinfo(tid).task

        self._task_logger.info(f"start  {task_ids}")

        before = ns.sim_time()

        to_send_str = ", ".join(
            [f"(pid={p}, tid={t.task_id})" for p, t in tasks_to_send.items()]
        )

        # NOTE: we only allow all tasks to be of SinglePairTask for now!
        self._task_logger.debug(f"trying EPR tasks {to_send_str}")
        if all(isinstance(task, SinglePairTask) for task in tasks_to_send.values()):
            pid = yield from self.driver.handle_single_pair_group(
                tasks_to_send.values()
            )
        elif len(tasks_to_send) == 1:
            pid = list(tasks_to_send.keys())[0]
            task = tasks_to_send[pid]
            yield from self._driver.handle_task(task)
        else:
            raise RuntimeError(
                "Scheduling multiple MultiPairTasks is not supported at the moment"
            )
        self._task_logger.debug(f"EPR pair delivered for PID {pid}")

        finished_task = tasks_to_send[pid]

        # Only now we know which task has finished, and we can record its start time.
        self._task_starts[finished_task.task_id] = before
        self.record_start_timestamp(finished_task.pid, before)

        after = ns.sim_time()

        self.record_end_timestamp(finished_task.pid, after)
        duration = after - before
        self._task_graph.decrease_deadlines(duration)
        self._task_logger.debug(f"removing task with tid {finished_task.task_id}")
        self._task_graph.remove_task(finished_task.task_id)
        self._finished_tasks.append(finished_task.task_id)
        self.send_signal(SIGNAL_TASK_COMPLETED)
        self._logger.info(f"finished task {finished_task}")
        self._task_logger.info(f"finish epr task {finished_task}")

        if self.name == "bob_qpu":
            self._task_logger.warning(f"finish epr task {finished_task}")

        self._tasks_executed[finished_task.task_id] = finished_task
        self._task_ends[finished_task.task_id] = after

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            status = self.update_status()
            if isinstance(status, StatusGraphEmpty):
                yield self.await_port_input(self.node_scheduler_out_port)
            elif isinstance(status, StatusBlockedOnOtherCore):
                self._task_logger.debug("waiting for TASK_COMPLETED signal")
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._task_logger.debug("got TASK_COMPLETED signal")
                self.update_external_predcessors()
            elif isinstance(status, StatusBlockedOnResource):
                # raise RuntimeError(
                #     "Blocked on resource: handling this has not been implemented yet"
                # )
                # TODO: ACTUALLY WAIT FOR A SIGNAL SAYING A RESOURCE HAS BEEN FREED!
                # Now we just assume that finishing an external task also always
                # resolves the resource problem, but this is not necessarily the case.
                self._task_logger.debug(
                    "blocked on resource: waiting for TASK_COMPLETED signal"
                )
                yield self.await_signal(
                    sender=self._other_scheduler,
                    signal_label=SIGNAL_TASK_COMPLETED,
                )
                self._task_logger.debug(
                    "blocked on resource: got TASK_COMPLETED signal"
                )
                self.update_external_predcessors()
            elif isinstance(status, StatusEprGen):
                yield from self.handle_epr_gen(status.task_ids)
                self.update_external_predcessors()
            elif isinstance(status, StatusNextTask):
                yield from self.handle_task(status.task_id)
                self.update_external_predcessors()
