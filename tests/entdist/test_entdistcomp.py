from __future__ import annotations

from qoala.lang.ehi import EhiNetworkInfo
from qoala.sim.entdist.entdistcomp import EntDistComponent


def create_entdistcomp(num_nodes: int) -> EntDistComponent:
    nodes = {id: f"node_{id}" for id in range(num_nodes)}
    ehi_network = EhiNetworkInfo.only_nodes(nodes)

    return EntDistComponent(ehi_network)


def test_one_node():
    comp = create_entdistcomp(num_nodes=1)

    # should have 2 node ports
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
