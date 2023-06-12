from __future__ import annotations

from typing import Dict, Optional

from netsquid.components import QuantumProcessor
from netsquid.components.component import Port
from netsquid.nodes import Node

from qoala.lang.ehi import EhiNetworkInfo
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.netstack import NetstackComponent
from qoala.sim.qnos import QnosComponent


class ProcNodeComponent(Node):
    """NetSquid component representing a quantum network node containing a software
    stack consisting of Host, QNodeOS and QDevice.

    This component has three subcomponents:
        - a QnosComponent
        - a HostComponent

    Has communications ports between
     - the Host component on this node and the Host components on other nodes
     - the QNodeOS component on this node and the QNodeOS components on other nodes

    This is a static container for components and ports.
    Behavior of the node is modeled in the `ProcNode` class, which is a subclass
    of `Protocol`.

    This class is a subclass of the NetSquid `Node` class and can hence be used as
    a standard NetSquid node.
    """

    def __init__(
        self,
        name: str,
        qprocessor: QuantumProcessor,
        ehi_network: EhiNetworkInfo,
        node_id: Optional[int] = None,
    ) -> None:
        """ProcNodeComponent constructor. Typically created indirectly through
        constructing a `ProcNode`."""
        super().__init__(name, ID=node_id)
        self.qmemory = qprocessor

        qnos_comp = QnosComponent(self)
        self.add_subcomponent(qnos_comp, "qnos")

        host_comp = HostComponent(self, ehi_network)
        self.add_subcomponent(host_comp, "host")

        netstack_comp = NetstackComponent(self, ehi_network)
        self.add_subcomponent(netstack_comp, "netstack")

        self.host_comp.ports["qnos_out"].connect(self.qnos_comp.ports["host_in"])
        self.host_comp.ports["qnos_in"].connect(self.qnos_comp.ports["host_out"])

        # Ports for communicating with other nodes
        self._netstack_peer_in_ports: Dict[str, str] = {}  # peer name -> port name
        self._netstack_peer_out_ports: Dict[str, str] = {}  # peer name -> port name
        self._host_peer_in_ports: Dict[str, str] = {}  # peer name -> port name
        self._host_peer_out_ports: Dict[str, str] = {}  # peer name -> port name

        # Ports for communicating with the EntDist
        self.add_ports(["entdist_out", "entdist_in"])
        self.netstack_comp.entdist_out_port.forward_output(self.entdist_out_port)
        self.entdist_in_port.forward_input(self.netstack_comp.entdist_in_port)

        for other_node in ehi_network.nodes.values():
            if other_node == self.name:
                continue

            netstack_port_in_name = f"netstack_peer_{other_node}_in"
            netstack_port_out_name = f"netstack_peer_{other_node}_out"
            self._netstack_peer_in_ports[other_node] = netstack_port_in_name
            self._netstack_peer_out_ports[other_node] = netstack_port_out_name

            host_port_in_name = f"host_peer_{other_node}_in"
            host_port_out_name = f"host_peer_{other_node}_out"
            self._host_peer_in_ports[other_node] = host_port_in_name
            self._host_peer_out_ports[other_node] = host_port_out_name

        self.add_ports(self._netstack_peer_in_ports.values())
        self.add_ports(self._netstack_peer_out_ports.values())
        self.add_ports(self._host_peer_in_ports.values())
        self.add_ports(self._host_peer_out_ports.values())

        for other_node in ehi_network.nodes.values():
            if other_node == self.name:
                continue
            self.netstack_comp.peer_out_port(other_node).forward_output(
                self.netstack_peer_out_port(other_node)
            )
            self.netstack_peer_in_port(other_node).forward_input(
                self.netstack_comp.peer_in_port(other_node)
            )
            self.host_comp.peer_out_port(other_node).forward_output(
                self.host_peer_out_port(other_node)
            )
            self.host_peer_in_port(other_node).forward_input(
                self.host_comp.peer_in_port(other_node)
            )

    @property
    def node_name(self) -> str:
        return self.name  # type: ignore

    @property
    def node_id(self) -> int:
        return self.ID  # type: ignore

    @property
    def host_comp(self) -> HostComponent:
        return self.subcomponents["host"]  # type: ignore

    @property
    def qnos_comp(self) -> QnosComponent:
        return self.subcomponents["qnos"]  # type: ignore

    @property
    def netstack_comp(self) -> NetstackComponent:
        return self.subcomponents["netstack"]  # type: ignore

    @property
    def qprocessor(self) -> QuantumProcessor:
        return self.qmemory

    def host_peer_in_port(self, name: str) -> Port:
        port_name = self._host_peer_in_ports[name]
        return self.ports[port_name]

    def host_peer_out_port(self, name: str) -> Port:
        port_name = self._host_peer_out_ports[name]
        return self.ports[port_name]

    def netstack_peer_in_port(self, name: str) -> Port:
        port_name = self._netstack_peer_in_ports[name]
        return self.ports[port_name]

    def netstack_peer_out_port(self, name: str) -> Port:
        port_name = self._netstack_peer_out_ports[name]
        return self.ports[port_name]

    @property
    def entdist_in_port(self) -> Port:
        return self.ports["entdist_in"]

    @property
    def entdist_out_port(self) -> Port:
        return self.ports["entdist_out"]
