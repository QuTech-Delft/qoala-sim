from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from qlink_interface.interface import (
    ReqCreateBase,
    ResCreateAndKeep,
    ResMeasureDirectly,
)

from pydynaa import EventExpression
from qoala.runtime.environment import GlobalEnvironment, LocalEnvironment
from qoala.runtime.message import Message
from qoala.sim.common import ComponentProtocol, PortListener
from qoala.sim.egpmgr import EgpManager
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.events import EVENT_WAIT
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.qdevice import QDevice
from qoala.sim.signals import (
    SIGNAL_ENTD_NSTK_MSG,
    SIGNAL_MEMORY_FREED,
    SIGNAL_NSTK_ENTD_MSG,
    SIGNAL_PEER_NSTK_MSG,
    SIGNAL_PROC_NSTK_MSG,
)


class EntDistInterface(ComponentProtocol):
    def __init__(
        self,
        comp: EntDistComponent,
        global_env: GlobalEnvironment,
    ) -> None:
        super().__init__(name=f"{comp.name}_protocol", comp=comp)
        self._comp = comp
        self._global_env = global_env

        for node in self._global_env.get_all_node_names():
            self.add_listener(
                f"node_{node}",
                PortListener(
                    self._comp.node_in_port(node), f"{SIGNAL_NSTK_ENTD_MSG}_{node}"
                ),
            )

    def send_node_msg(self, node: str, msg: Message) -> None:
        self._comp.node_out_port(node).tx_output(msg)

    def receive_node_msg(self, node: str) -> Generator[EventExpression, None, Message]:
        return (
            yield from self._receive_msg(
                f"node_{node}", f"{SIGNAL_NSTK_ENTD_MSG}_{node}"
            )
        )

    def await_result_create_keep(
        self, remote_id: int
    ) -> Generator[EventExpression, None, ResCreateAndKeep]:
        egp = self._egpmgr.get_egp(remote_id)
        yield self.await_signal(
            sender=egp,
            signal_label=ResCreateAndKeep.__name__,
        )
        result: ResCreateAndKeep = egp.get_signal_result(
            ResCreateAndKeep.__name__, receiver=self
        )
        return result
