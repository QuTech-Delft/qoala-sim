from __future__ import annotations

from typing import Dict

from netsquid.components.component import Component, Port
from netsquid.nodes import Node

from qoala.runtime.environment import GlobalEnvironment


class EntDistComponent(Component):
    def __init__(self, global_env: GlobalEnvironment) -> None:
        super().__init__(f"global_entanglement_distributor")

        self._node_in_ports: Dict[str, str] = {}  # node name -> port name
        self._node_out_ports: Dict[str, str] = {}  # node name -> port name

        for node in global_env.get_nodes().values():
            port_in_name = f"node_{node.name}_in"
            port_out_name = f"node_{node.name}_out"
            self._node_in_ports[node.name] = port_in_name
            self._node_out_ports[node.name] = port_out_name

        self.add_ports(self._node_in_ports.values())
        self.add_ports(self._node_out_ports.values())

    def node_in_port(self, name: str) -> Port:
        port_name = self._node_in_ports[name]
        return self.ports[port_name]

    def node_out_port(self, name: str) -> Port:
        port_name = self._node_out_ports[name]
        return self.ports[port_name]
