from __future__ import annotations

from qoala.runtime.environment import NetworkInfo
from qoala.sim.procnodecomp import ProcNodeComponent


def create_procnodecomp(num_other_nodes: int) -> ProcNodeComponent:
    nodes = {id: f"node_{id}" for id in range(1, num_other_nodes + 1)}
    nodes[0] = "alice"
    env = NetworkInfo.with_nodes(nodes)

    return ProcNodeComponent(name="alice", qprocessor=None, network_info=env)


def test_no_other_nodes():
    comp = create_procnodecomp(num_other_nodes=0)

    # should not have 2 entdist ports
    assert len(comp.ports) == 2
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports


def test_one_other_node():
    comp = create_procnodecomp(num_other_nodes=1)

    # should have 2 host peer ports + 2 netstack peer ports + 2 entdist ports
    assert len(comp.ports) == 6
    assert "host_peer_node_1_in" in comp.ports
    assert "host_peer_node_1_out" in comp.ports
    assert "netstack_peer_node_1_in" in comp.ports
    assert "netstack_peer_node_1_out" in comp.ports
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports

    # Test properties
    assert comp.host_peer_in_port("node_1") == comp.ports["host_peer_node_1_in"]
    assert comp.host_peer_out_port("node_1") == comp.ports["host_peer_node_1_out"]
    assert comp.netstack_peer_in_port("node_1") == comp.ports["netstack_peer_node_1_in"]
    assert (
        comp.netstack_peer_out_port("node_1") == comp.ports["netstack_peer_node_1_out"]
    )


def test_many_other_nodes():
    comp = create_procnodecomp(num_other_nodes=5)

    # should 5 * 4 peer ports + 2 entdist ports
    assert len(comp.ports) == 22

    for i in range(1, 6):
        assert f"host_peer_node_{i}_in" in comp.ports
        assert f"host_peer_node_{i}_out" in comp.ports
        assert f"netstack_peer_node_{i}_in" in comp.ports
        assert f"netstack_peer_node_{i}_out" in comp.ports
        # Test properties
        assert (
            comp.host_peer_in_port(f"node_{i}") == comp.ports[f"host_peer_node_{i}_in"]
        )
        assert (
            comp.host_peer_out_port(f"node_{i}")
            == comp.ports[f"host_peer_node_{i}_out"]
        )
        assert (
            comp.netstack_peer_in_port(f"node_{i}")
            == comp.ports[f"netstack_peer_node_{i}_in"]
        )
        assert (
            comp.netstack_peer_out_port(f"node_{i}")
            == comp.ports[f"netstack_peer_node_{i}_out"]
        )


if __name__ == "__main__":
    test_no_other_nodes()
    test_one_other_node()
    test_many_other_nodes()
