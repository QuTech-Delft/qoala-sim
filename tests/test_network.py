from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkInfo
from qoala.sim.build import build_network


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


def test1():
    server_id = 0
    client_id = 1
    network_info = NetworkInfo.with_nodes({server_id: "server", client_id: "client"})

    num_qubits = 1
    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )
    network = build_network(network_cfg, network_info)


if __name__ == "__main__":
    test1()
