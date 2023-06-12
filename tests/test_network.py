import pytest

from qoala.runtime.config import (
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.sim.build import build_network_from_config


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def test_perfect_links():
    server_id = 0
    client_id = 1

    num_qubits = 1
    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )
    network = build_network_from_config(network_cfg)

    server = network.nodes["server"]
    link_info = server.network_ehi.get_link(server_id, client_id)
    assert link_info.duration == 1000
    assert link_info.fidelity == 1.0


def test_depolarise_links():
    server_id = 0
    client_id = 1

    num_qubits = 1
    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    link_fidelity = 0.8
    link_cfg = LinkConfig.simple_depolarise_config(
        fidelity=link_fidelity, state_delay=1000
    )
    link_between_cfg = LinkBetweenNodesConfig(
        node_id1=server_id, node_id2=client_id, link_config=link_cfg
    )
    network_cfg = ProcNodeNetworkConfig(
        nodes=[server_node_cfg, client_node_cfg], links=[link_between_cfg]
    )
    network = build_network_from_config(network_cfg)

    server = network.nodes["server"]
    link_info = server.network_ehi.get_link(server_id, client_id)
    assert link_info.duration == 1000
    assert link_info.fidelity == pytest.approx(0.8)


if __name__ == "__main__":
    test_perfect_links()
    test_depolarise_links()
