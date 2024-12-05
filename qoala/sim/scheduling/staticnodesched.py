from __future__ import annotations

from typing import Generator

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.qnos import Qnos
from qoala.sim.scheduling.nodesched import NodeScheduler


class StaticNodeScheduler(NodeScheduler):
    """TODO docstrings"""

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
            is_predictable=True,
        )

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
