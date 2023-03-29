import pytest
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel
from netsquid.components.qprocessor import MissingInstructionError, QuantumProcessor

from qoala.runtime.config import (
    GenericQDeviceConfig,
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
    NVQDeviceConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkEhi
from qoala.runtime.lhi import LhiGateInfo, LhiQubitInfo, LhiTopology, LhiTopologyBuilder
from qoala.sim.build import (
    build_generic_qprocessor,
    build_network,
    build_nv_qprocessor,
    build_procnode,
    build_qprocessor_from_topology,
)


def uniform_topology(num_qubits: int) -> LhiTopology:
    qubit_info = LhiQubitInfo(
        is_communication=True,
        error_model=T1T2NoiseModel,
        error_model_kwargs={"T1": 1e6, "T2": 1e6},
    )
    single_gate_infos = [
        LhiGateInfo(
            instruction=instr,
            duration=5e3,
            error_model=DepolarNoiseModel,
            error_model_kwargs={
                "depolar_rate": 0.2,
                "time_independent": True,
            },
        )
        for instr in [INSTR_X, INSTR_Y, INSTR_Z]
    ]
    two_gate_infos = [
        LhiGateInfo(
            instruction=INSTR_CNOT,
            duration=2e4,
            error_model=DepolarNoiseModel,
            error_model_kwargs={
                "depolar_rate": 0.2,
                "time_independent": True,
            },
        )
    ]
    return LhiTopologyBuilder.fully_uniform(
        num_qubits=num_qubits,
        qubit_info=qubit_info,
        single_gate_infos=single_gate_infos,
        two_gate_infos=two_gate_infos,
    )


def test_build_from_topology():
    num_qubits = 3
    topology = uniform_topology(num_qubits)
    proc: QuantumProcessor = build_qprocessor_from_topology("proc", topology)
    assert proc.num_positions == num_qubits

    for i in range(num_qubits):
        assert (
            proc.get_instruction_duration(INSTR_X, [i])
            == topology.find_single_gate(i, INSTR_X).duration
        )
        with pytest.raises(MissingInstructionError):
            proc.get_instruction_duration(INSTR_ROT_X, [i])

    assert (
        proc.get_instruction_duration(INSTR_CNOT, [0, 1])
        == topology.find_multi_gate([0, 1], INSTR_CNOT).duration
    )


def test_build_perfect_topology():
    num_qubits = 3
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[INSTR_X, INSTR_Y],
        single_duration=5e3,
        two_instructions=[INSTR_CNOT],
        two_duration=100e3,
    )
    proc: QuantumProcessor = build_qprocessor_from_topology("proc", topology)
    assert proc.num_positions == num_qubits

    for i in range(num_qubits):
        assert (
            proc.get_instruction_duration(INSTR_X, [i])
            == topology.find_single_gate(i, INSTR_X).duration
        )
        assert proc.get_instruction_duration(INSTR_X, [i]) == 5e3

        with pytest.raises(MissingInstructionError):
            proc.get_instruction_duration(INSTR_ROT_X, [i])

    assert (
        proc.get_instruction_duration(INSTR_CNOT, [0, 1])
        == topology.find_multi_gate([0, 1], INSTR_CNOT).duration
    )
    assert proc.get_instruction_duration(INSTR_CNOT, [0, 1]) == 100e3


def test_build_generic_perfect():
    num_qubits = 2
    cfg = GenericQDeviceConfig.perfect_config(num_qubits)
    proc: QuantumProcessor = build_generic_qprocessor(name="alice", cfg=cfg)
    assert proc.num_positions == num_qubits

    for i in range(num_qubits):
        assert proc.get_instruction_duration(INSTR_INIT, [i]) == cfg.init_time
        assert proc.get_instruction_duration(INSTR_MEASURE, [i]) == cfg.measure_time
        assert proc.get_instruction_duration(INSTR_X, [i]) == cfg.single_qubit_gate_time
        assert (
            proc.get_instruction_duration(INSTR_ROT_X, [i])
            == cfg.single_qubit_gate_time
        )

    assert proc.get_instruction_duration(INSTR_CNOT, [0, 1]) == cfg.two_qubit_gate_time

    # TODO: check topology??!


def test_build_nv_perfect():
    num_qubits = 2
    cfg = NVQDeviceConfig.perfect_config(num_qubits)
    proc: QuantumProcessor = build_nv_qprocessor(name="alice", cfg=cfg)
    assert proc.num_positions == num_qubits

    assert proc.get_instruction_duration(INSTR_INIT, [0]) == cfg.electron_init
    assert proc.get_instruction_duration(INSTR_MEASURE, [0]) == cfg.measure
    assert proc.get_instruction_duration(INSTR_ROT_X, [0]) == cfg.electron_rot_x

    for i in range(1, num_qubits):
        assert proc.get_instruction_duration(INSTR_INIT, [i]) == cfg.carbon_init
        assert proc.get_instruction_duration(INSTR_ROT_X, [i]) == cfg.carbon_rot_x
        with pytest.raises(MissingInstructionError):
            assert proc.get_instruction_duration(INSTR_MEASURE, [i])

    with pytest.raises(MissingInstructionError):
        proc.get_instruction_duration(INSTR_CNOT, [0, 1])
        proc.get_instruction_duration(INSTR_CXDIR, [1, 0])

    assert proc.get_instruction_duration(INSTR_CXDIR, [0, 1]) == cfg.ec_controlled_dir_x


def test_build_procnode():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    latencies = LatenciesConfig(
        host_qnos_latency=3,
        host_instr_time=17,
        qnos_instr_time=20,
        host_peer_latency=5,
        netstack_peer_latency=9,
    )
    cfg = ProcNodeConfig(
        node_name="the_node", node_id=42, topology=top_cfg, latencies=latencies
    )
    network_ehi = NetworkEhi()
    network_ehi.add_node(42, "the_node")
    network_ehi.add_node(43, "other_node")

    procnode = build_procnode(cfg, network_ehi)

    assert procnode.node.name == "the_node"
    procnode.host_comp.peer_in_port("other_node")  # should not raise error
    procnode.netstack_comp.peer_in_port("other_node")  # should not raise error
    assert procnode.qnos.processor._latencies.qnos_instr_time == 20
    assert procnode.host.processor._latencies.host_instr_time == 17
    assert procnode.host.processor._latencies.host_peer_latency == 5

    expected_topology = LhiTopologyBuilder.from_config(top_cfg)
    expected_qprocessor = build_qprocessor_from_topology("the_node", expected_topology)
    actual_qprocessor = procnode.qdevice.qprocessor

    assert expected_qprocessor.name == actual_qprocessor.name
    assert expected_qprocessor.num_positions == actual_qprocessor.num_positions

    assert expected_qprocessor.get_instruction_duration(
        INSTR_X, [0]
    ) == actual_qprocessor.get_instruction_duration(INSTR_X, [0])

    assert expected_topology == procnode.qdevice.topology


def test_build_network():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    cfg_alice = ProcNodeConfig(
        node_name="alice", node_id=42, topology=top_cfg, latencies=LatenciesConfig()
    )
    cfg_bob = ProcNodeConfig(
        node_name="bob", node_id=43, topology=top_cfg, latencies=LatenciesConfig()
    )
    network_ehi = NetworkEhi()
    network_ehi.add_node(42, "alice")
    network_ehi.add_node(43, "bob")

    link_cfg = LinkConfig.perfect_config(state_delay=1000)
    link_ab = LinkBetweenNodesConfig(node_id1=42, node_id2=43, link_config=link_cfg)
    cfg = ProcNodeNetworkConfig(nodes=[cfg_alice, cfg_bob], links=[link_ab])
    network = build_network(cfg, network_ehi)

    assert len(network.nodes) == 2
    assert "alice" in network.nodes
    assert "bob" in network.nodes
    assert network.entdist is not None

    alice = network.nodes["alice"]
    bob = network.nodes["bob"]
    entdist = network.entdist

    assert entdist.get_sampler(42, 43).delay == 1000

    # NOTE: alice.host_comp.peer_in_port("bob") does not have a 'connected_port'.
    # Rather, messages are forwarded from alice.node.host_peer_in_port("bob").

    alice_host_in = alice.node.host_peer_in_port("bob")
    alice_host_out = alice.node.host_peer_out_port("bob")
    bob_host_in = bob.node.host_peer_in_port("alice")
    bob_host_out = bob.node.host_peer_out_port("alice")
    assert alice_host_in.connected_port == bob_host_out
    assert alice_host_out.connected_port == bob_host_in

    alice_ent_in = alice.node.entdist_in_port
    alice_ent_out = alice.node.entdist_out_port
    bob_ent_in = bob.node.entdist_in_port
    bob_ent_out = bob.node.entdist_out_port
    assert alice_ent_in.connected_port == entdist.comp.node_out_port("alice")
    assert alice_ent_out.connected_port == entdist.comp.node_in_port("alice")
    assert bob_ent_in.connected_port == entdist.comp.node_out_port("bob")
    assert bob_ent_out.connected_port == entdist.comp.node_in_port("bob")


def test_build_network_perfect_links():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    cfg_alice = ProcNodeConfig(
        node_name="alice", node_id=42, topology=top_cfg, latencies=LatenciesConfig()
    )
    cfg_bob = ProcNodeConfig(
        node_name="bob", node_id=43, topology=top_cfg, latencies=LatenciesConfig()
    )
    network_ehi = NetworkEhi()
    network_ehi.add_node(42, "alice")
    network_ehi.add_node(43, "bob")

    cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[cfg_alice, cfg_bob], link_duration=500
    )
    network = build_network(cfg, network_ehi)

    assert len(network.nodes) == 2
    assert "alice" in network.nodes
    assert "bob" in network.nodes
    assert network.entdist is not None

    entdist = network.entdist
    assert entdist.get_sampler(42, 43).delay == 500


if __name__ == "__main__":
    test_build_from_topology()
    test_build_perfect_topology()
    test_build_generic_perfect()
    test_build_nv_perfect()
    test_build_procnode()
    test_build_network()
    test_build_network_perfect_links()
