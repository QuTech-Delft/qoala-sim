from __future__ import annotations

from typing import Generator, List

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.componentprot import ComponentProtocol, PortListener
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.events import EVENT_WAIT, SIGNAL_NSTK_ENTD_MSG


class EntDistInterface(ComponentProtocol):
    def __init__(
        self,
        comp: EntDistComponent,
        ehi_network: EhiNetworkInfo,
    ) -> None:
        super().__init__(name=f"{comp.name}_protocol", comp=comp)
        self._comp = comp
        self._ehi_network = ehi_network

        self._all_node_names: List[str] = self._ehi_network.get_all_node_names()

        for node in self._all_node_names:
            self.add_listener(
                f"node_{node}",
                PortListener(
                    self._comp.node_in_port(node), f"{SIGNAL_NSTK_ENTD_MSG}_{node}"
                ),
            )

    def remote_id_to_peer_name(self, remote_id: int) -> str:
        return self._ehi_network.nodes[remote_id]

    def send_node_msg(self, node: str, msg: Message) -> None:
        self._comp.node_out_port(node).tx_output(msg)

    def receive_node_msg(self, node: str) -> Generator[EventExpression, None, Message]:
        yield from self._wait_for_msg(f"node_{node}", f"{SIGNAL_NSTK_ENTD_MSG}_{node}")
        return self._pop_any_msg(f"node_{node}")

    def wait_for_any_msg(self) -> Generator[EventExpression, None, None]:
        yield from self._wait_for_msg_any_source(
            [f"node_{node}" for node in self._all_node_names],
            [f"{SIGNAL_NSTK_ENTD_MSG}_{node}" for node in self._all_node_names],
        )

    def receive_msg(self) -> Generator[EventExpression, None, Message]:
        yield from self._wait_for_msg_any_source(
            [f"node_{node}" for node in self._all_node_names],
            [f"{SIGNAL_NSTK_ENTD_MSG}_{node}" for node in self._all_node_names],
        )
        return self._pop_any_msg_any_source(
            [f"node_{node}" for node in self._all_node_names]
        )

    def pop_all_messages(self) -> List[Message]:
        return self._pop_all_messages([f"node_{node}" for node in self._all_node_names])

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr
