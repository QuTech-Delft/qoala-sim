from __future__ import annotations

from typing import Generator

import netsquid as ns
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.runtime.environment import (
    GlobalEnvironment,
    GlobalNodeInfo,
    LocalEnvironment,
)
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.netstack.netstackinterface import NetstackInterface


def create_entdistcomp(num_nodes: int) -> EntDistComponent:
    env = GlobalEnvironment()

    for id in range(num_nodes):
        node_info = GlobalNodeInfo(f"node_{id}", id)
        env.add_node(id, node_info)

    return EntDistComponent(env)


def test_one_node():
    comp = create_entdistcomp(num_nodes=1)

    # should have 2 ndoe ports
    assert len(comp.ports) == 2
    assert "node_node_0_in" in comp.ports
    assert "node_node_0_out" in comp.ports

    # Test properties
    assert comp.node_in_port("node_0") == comp.ports["node_node_0_in"]
    assert comp.node_out_port("node_0") == comp.ports["node_node_0_out"]


def test_many_nodes():
    comp = create_entdistcomp(num_nodes=5)

    # should have 5 * 2 node ports
    assert len(comp.ports) == 10

    for i in range(5):
        assert f"node_node_{i}_in" in comp.ports
        assert f"node_node_{i}_out" in comp.ports
        # Test properties
        assert comp.node_in_port(f"node_{i}") == comp.ports[f"node_node_{i}_in"]
        assert comp.node_out_port(f"node_{i}") == comp.ports[f"node_node_{i}_out"]


if __name__ == "__main__":
    test_one_node()
    test_many_nodes()
