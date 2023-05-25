from __future__ import annotations

from typing import Dict

from netsquid.components.component import Component, Port
from netsquid.nodes import Node

from qoala.lang.ehi import EhiNetworkInfo


class HostComponent(Component):
    """NetSquid component representing a Host.

    Subcomponent of a ProcNodeComponent.

    This is a static container for Host-related components and ports. Behavior
    of a Host is modeled in the `Host` class, which is a subclass of `Protocol`.
    """

    def __init__(self, node: Node, ehi_network: EhiNetworkInfo) -> None:
        super().__init__(f"{node.name}_host")
        self._node = node

        self._peer_in_ports: Dict[str, str] = {}  # peer name -> port name
        self._peer_out_ports: Dict[str, str] = {}  # peer name -> port name

        for other_node in ehi_network.nodes.values():
            if other_node == node.name:
                continue
            port_in_name = f"peer_{other_node}_in"
            port_out_name = f"peer_{other_node}_out"
            self._peer_in_ports[other_node] = port_in_name
            self._peer_out_ports[other_node] = port_out_name

        self.add_ports(self._peer_in_ports.values())
        self.add_ports(self._peer_out_ports.values())

        self.add_ports(["qnos_in", "qnos_out"])
        self.add_ports(["nstk_in", "nstk_out"])

    @property
    def node_name(self) -> str:
        return self._node.name  # type: ignore

    @property
    def node_id(self) -> int:
        return self._node.ID  # type: ignore

    @property
    def qnos_in_port(self) -> Port:
        return self.ports["qnos_in"]

    @property
    def qnos_out_port(self) -> Port:
        return self.ports["qnos_out"]

    @property
    def netstack_in_port(self) -> Port:
        return self.ports["nstk_in"]

    @property
    def netstack_out_port(self) -> Port:
        return self.ports["nstk_out"]

    def peer_in_port(self, name: str) -> Port:
        port_name = self._peer_in_ports[name]
        return self.ports[port_name]

    def peer_out_port(self, name: str) -> Port:
        port_name = self._peer_out_ports[name]
        return self.ports[port_name]
