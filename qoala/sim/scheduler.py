from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional

from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.runtime.environment import LocalEnvironment
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import (
    BatchInfo,
    BatchResult,
    ProgramBatch,
    ProgramInstance,
    ProgramResult,
)
from qoala.runtime.schedule import TaskSchedule
from qoala.runtime.task import BlockTask, TaskCreator, TaskExecutionMode
from qoala.sim.driver import CpuDriver, QpuDriver
from qoala.sim.eprsocket import EprSocket
from qoala.sim.events import EVENT_WAIT
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.process import QoalaProcess
from qoala.sim.qnos import Qnos
from qoala.util.logging import LogManager


class Scheduler(Protocol):
    def __init__(
        self,
        node_name: str,
        host: Host,
        qnos: Qnos,
        netstack: Netstack,
        memmgr: MemoryManager,
        local_env: LocalEnvironment,
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
        self._local_env = local_env
        self._local_ehi = local_ehi
        self._network_ehi = network_ehi

        self._prog_instance_counter: int = 0
        self._batch_counter: int = 0
        self._batches: Dict[int, ProgramBatch] = {}  # batch ID -> batch
        self._prog_results: Dict[int, ProgramResult] = {}  # program ID -> result
        self._batch_results: Dict[int, BatchResult] = {}  # batch ID -> result

        self._block_schedule: Optional[TaskSchedule] = None

        self._cpudriver = CpuDriver(node_name, host.processor, memmgr)
        self._qpudriver = QpuDriver(
            node_name, host.processor, qnos.processor, netstack.processor, memmgr
        )

        self._cpudriver.set_other_driver(self._qpudriver)
        self._qpudriver.set_other_driver(self._cpudriver)

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
    def cpudriver(self) -> CpuDriver:
        return self._cpudriver

    @property
    def qpudriver(self) -> QpuDriver:
        return self._qpudriver

    @property
    def block_schedule(self) -> TaskSchedule:
        assert self._block_schedule is not None
        return self._block_schedule

    def submit_batch(self, batch_info: BatchInfo) -> None:
        prog_instances: List[ProgramInstance] = []

        network_info = self._local_env.get_network_info()

        for i in range(batch_info.num_iterations):
            pid = self._prog_instance_counter
            task_creator = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC)
            # TODO: allow multiple remote nodes in single program??
            remote_names = list(batch_info.program.meta.csockets.values())
            if len(remote_names) > 0:
                remote_name = list(batch_info.program.meta.csockets.values())[0]
                remote_id = network_info.get_node_id(remote_name)
            else:
                remote_id = None
            block_tasks = task_creator.from_program(
                batch_info.program, pid, self._local_ehi, self._network_ehi, remote_id
            )

            instance = ProgramInstance(
                pid=pid,
                program=batch_info.program,
                inputs=batch_info.inputs[i],
                unit_module=batch_info.unit_module,
                block_tasks=block_tasks,
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
            network_info = self._local_env.get_network_info()
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

    def upload_cpu_schedule(self, schedule: TaskSchedule) -> None:
        self._cpudriver.upload_schedule(schedule)

    def upload_qpu_schedule(self, schedule: TaskSchedule) -> None:
        self._qpudriver.upload_schedule(schedule)

    def upload_schedule(self, schedule: TaskSchedule) -> None:
        self._block_schedule = schedule
        self._cpudriver.upload_schedule(schedule.cpu_schedule)
        self._qpudriver.upload_schedule(schedule.qpu_schedule)

    def start(self) -> None:
        super().start()
        self._cpudriver.start()
        self._qpudriver.start()

    def stop(self) -> None:
        self._qpudriver.stop()
        self._cpudriver.stop()
        super().stop()

    def submit_program_instance(self, prog_instance: ProgramInstance) -> None:
        process = self.create_process(prog_instance)
        self.memmgr.add_process(process)
        self.initialize_process(process)

    def get_tasks_to_schedule(self) -> List[BlockTask]:
        all_tasks: List[BlockTask] = []

        for batch in self._batches.values():
            for inst in batch.instances:
                all_tasks.extend(inst.block_tasks)

        return all_tasks
