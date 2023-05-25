from __future__ import annotations

from typing import Dict

from netsquid.components.component import Component, Port

from qoala.lang.ehi import EhiNetworkInfo


class EntDistComponent(Component):
    def __init__(self, ehi_network: EhiNetworkInfo) -> None:
        super().__init__("global_entanglement_distributor")

        self._node_in_ports: Dict[str, str] = {}  # node name -> port name
        self._node_out_ports: Dict[str, str] = {}  # node name -> port name

        for node_name in ehi_network.nodes.values():
            port_in_name = f"node_{node_name}_in"
            port_out_name = f"node_{node_name}_out"
            self._node_in_ports[node_name] = port_in_name
            self._node_out_ports[node_name] = port_out_name

        self.add_ports(self._node_in_ports.values())
        self.add_ports(self._node_out_ports.values())

    def node_in_port(self, name: str) -> Port:
        port_name = self._node_in_ports[name]
        return self.ports[port_name]

    def node_out_port(self, name: str) -> Port:
        port_name = self._node_out_ports[name]
        return self.ports[port_name]
