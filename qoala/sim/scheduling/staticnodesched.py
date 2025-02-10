from __future__ import annotations

from typing import Dict, Generator, List, Optional

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.qnos import Qnos
from qoala.sim.scheduling.nodesched import NodeScheduler


class StaticNodeScheduler(NodeScheduler):
    """
    The static node scheduler assumes that the control-flow of all programs being
    executed is known beforehand, and that this control-flow is already encoded in the
    task graph that is uploaded to it (using `upload_task_graph()`).
    That is, when predictable programs are to be executed, the user must create a
    "full" task graph (meaning, a graph containing all tasks needed for full execution
    of the programs), and upload it before starting the scheduler.
    The scheduler then simply executes all tasks until the task graph is empty.
    It will hence never itself add new tasks to the task graph at runtime.
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
        super().__init__(
            node_name=node_name,
            host=host,
            qnos=qnos,
            netstack=netstack,
            memmgr=memmgr,
            local_ehi=local_ehi,
            network_ehi=network_ehi,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
            fcfs=fcfs,
            prio_epr=prio_epr,
        )

    def create_processes_for_batches(
        self,
        remote_pids: Optional[Dict[int, List[int]]] = None,  # batch ID -> PID list
        linear: bool = False,
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

        if self._const_batch is not None:
            for i, prog_instance in enumerate(self._const_batch.instances):
                process = self.create_process(prog_instance)
                self.memmgr.add_process(process)
                self.initialize_process(process)

    def start(self) -> None:
        # Processor schedulers start first to ensure that they will start running tasks after they receive the first
        # message from the node scheduler.
        self._cpu_scheduler.start()
        self._qpu_scheduler.start()
        super().start()

    def stop(self) -> None:
        self._qpu_scheduler.stop()
        self._cpu_scheduler.stop()
        super().stop()

    def run(self) -> Generator[EventExpression, None, None]:
        # static node scheduler doesn't do anything at runtime
        self._logger.debug("static node scheduler")

        # To make mypy happy
        return  # type: ignore
