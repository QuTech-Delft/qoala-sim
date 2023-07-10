from __future__ import annotations

from typing import Dict

from netsquid.components.component import Component, Port
from netsquid.nodes import Node

from qoala.lang.ehi import EhiNetworkInfo


class NetstackComponent(Component):
    """NetSquid component representing the network stack in QNodeOS.

    Subcomponent of a QnosComponent.

    Has communications ports with
     - the processor component of this QNodeOS
     - the netstack component of the remote node
        NOTE: at this moment only a single other node is supported in the network

    This is a static container for network-stack-related components and ports.
    Behavior of a QNodeOS network stack is modeled in the `NetProcNode` class,
    which is a subclass of `Protocol`.
    """

    def __init__(self, node: Node, ehi_network: EhiNetworkInfo) -> None:
        super().__init__(f"{node.name}_netstack")
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

        self.add_ports(["host_out", "host_in"])
        self.add_ports(["qnos_out", "qnos_in"])

        # Separate channel for "memory freed" signals
        self.add_ports(["qnos_mem_out", "qnos_mem_in"])

        self.add_ports(["entdist_out", "entdist_in"])

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
    def host_in_port(self) -> Port:
        return self.ports["host_in"]

    @property
    def host_out_port(self) -> Port:
        return self.ports["host_out"]

    @property
    def qnos_mem_in_port(self) -> Port:
        return self.ports["qnos_mem_in"]

    @property
    def qnos_mem_out_port(self) -> Port:
        return self.ports["qnos_mem_out"]

    @property
    def entdist_in_port(self) -> Port:
        return self.ports["entdist_in"]

    @property
    def entdist_out_port(self) -> Port:
        return self.ports["entdist_out"]

    def peer_in_port(self, name: str) -> Port:
        port_name = self._peer_in_ports[name]
        return self.ports[port_name]

    def peer_out_port(self, name: str) -> Port:
        port_name = self._peer_out_ports[name]
        return self.ports[port_name]

    @property
    def node(self) -> Node:
        return self._node
