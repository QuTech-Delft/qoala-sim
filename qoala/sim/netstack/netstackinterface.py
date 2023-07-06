from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.componentprot import ComponentProtocol, PortListener
from qoala.sim.events import (
    EVENT_WAIT,
    SIGNAL_ENTD_NSTK_MSG,
    SIGNAL_HOST_NSTK_MSG,
    SIGNAL_MEMORY_FREED,
    SIGNAL_NSTK_NSTK_MSG,
    SIGNAL_QNOS_NSTK_MSG,
)
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.qdevice import QDevice


@dataclass
class NetstackLatencies:
    netstack_peer_latency: float = 0  # processing time for messages from remote node

    @classmethod
    def all_zero(cls) -> NetstackLatencies:
        # NOTE: can also just use NetstackLatencies() which will default all values to 0
        # However, using this classmethod makes this behavior more explicit and clear.
        return NetstackLatencies(0)


class NetstackInterface(ComponentProtocol):
    """NetSquid protocol representing the QNodeOS network stack."""

    def __init__(
        self,
        comp: NetstackComponent,
        ehi_network: EhiNetworkInfo,
        qdevice: QDevice,
        memmgr: MemoryManager,
    ) -> None:
        """Network stack protocol constructor. Typically created indirectly through
        constructing a `Qnos` instance.

        :param comp: NetSquid component representing the network stack
        :param qnos: `Qnos` protocol that owns this protocol
        """
        super().__init__(name=f"{comp.name}_protocol", comp=comp)
        self._comp = comp
        self._qdevice = qdevice
        self._ehi_network = ehi_network
        self._memmgr = memmgr

        self.add_listener(
            "host",
            PortListener(self._comp.host_in_port, SIGNAL_HOST_NSTK_MSG),
        )

        self.add_listener(
            "qnos",
            PortListener(self._comp.qnos_in_port, SIGNAL_QNOS_NSTK_MSG),
        )

        self.add_listener(
            "qnos_mem",
            PortListener(self._comp.qnos_mem_in_port, SIGNAL_MEMORY_FREED),
        )

        self.add_listener(
            "entdist",
            PortListener(self._comp.entdist_in_port, SIGNAL_ENTD_NSTK_MSG),
        )

        for peer in self._ehi_network.nodes.values():
            if peer == self._comp.node_name:
                continue
            self.add_listener(
                f"peer_{peer}",
                PortListener(
                    self._comp.peer_in_port(peer), f"{SIGNAL_NSTK_NSTK_MSG}_{peer}"
                ),
            )

    def send_entdist_msg(self, msg: Message) -> None:
        """Send a message to the Entdist."""
        self._comp.entdist_out_port.tx_output(msg)

    def receive_entdist_msg(self) -> Generator[EventExpression, None, Message]:
        """Receive a message from the Entdist. Block until there is at least one
        message."""
        yield from self._wait_for_msg("entdist", SIGNAL_ENTD_NSTK_MSG)
        return self._pop_any_msg("entdist")

    def send_peer_msg(self, peer: str, msg: Message) -> None:
        """Send a message to the network stack of the other node."""

        self._comp.peer_out_port(peer).tx_output(msg)

    def receive_peer_msg(self, peer: str) -> Generator[EventExpression, None, Message]:
        """Receive a message from the network stack of the other node. Block until
        there is at least one message."""

        yield from self._wait_for_msg(f"peer_{peer}", f"{SIGNAL_NSTK_NSTK_MSG}_{peer}")
        return self._pop_any_msg(f"peer_{peer}")

    @property
    def qdevice(self) -> QDevice:
        return self._qdevice

    @property
    def node_id(self) -> int:
        return self._qdevice.node.ID  # type: ignore

    @property
    def memmgr(self) -> MemoryManager:
        return self._memmgr

    def remote_id_to_peer_name(self, remote_id: int) -> str:
        return self._ehi_network.nodes[remote_id]

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr
