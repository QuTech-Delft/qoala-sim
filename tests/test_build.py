import pytest
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel
from netsquid.components.qprocessor import MissingInstructionError, QuantumProcessor

from qoala.lang.ehi import EhiLinkInfo, EhiNetworkInfo
from qoala.runtime.config import (
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.instructions import INSTR_MEASURE_ALL, INSTR_ROT_X_ALL
from qoala.runtime.lhi import (
    LhiGateInfo,
    LhiLatencies,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiQubitInfo,
    LhiTopology,
    LhiTopologyBuilder,
)
from qoala.runtime.ntf import GenericNtf
from qoala.sim.build import (
    build_network_from_config,
    build_network_from_lhi,
    build_procnode_from_config,
    build_procnode_from_lhi,
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

    assert proc.name == "proc"

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


def test_build_from_topology_with_all_qubit_gates():
    num_qubits = 3
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[INSTR_INIT, INSTR_MEASURE, INSTR_ROT_Z],
        single_duration=5e3,
        two_instructions=[],
        two_duration=100e3,
        all_qubit_instructions=[INSTR_ROT_X_ALL, INSTR_MEASURE_ALL],
        all_qubit_duration=50e3,
    )
    proc: QuantumProcessor = build_qprocessor_from_topology("proc", topology)
    assert proc.num_positions == num_qubits

    for i in range(num_qubits):
        assert (
            proc.get_instruction_duration(INSTR_ROT_Z, [i])
            == topology.find_single_gate(i, INSTR_ROT_Z).duration
        )
        with pytest.raises(MissingInstructionError):
            proc.get_instruction_duration(INSTR_X, [i])

    assert proc.get_instruction_duration(INSTR_ROT_X_ALL, [0, 1, 2]) == 50e3
    assert proc.get_instruction_duration(INSTR_MEASURE_ALL, [0, 1, 2]) == 50e3


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

    assert proc.name == "proc"

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


def test_build_nv_perfect():
    num_qubits = 2
    parms = NvParams()
    parms.comm_init_duration = 500
    parms.comm_meas_duration = 1200
    parms.comm_gate_duration = 100
    parms.mem_init_duration = 35000
    parms.mem_meas_duration = 80000
    parms.mem_gate_duration = 25000
    parms.two_gate_duration = 100_000
    cfg = TopologyConfig.from_nv_params(num_qubits, parms)
    topology = LhiTopologyBuilder.from_config(cfg)
    proc: QuantumProcessor = build_qprocessor_from_topology("alice", topology)
    assert proc.num_positions == num_qubits

    assert proc.get_instruction_duration(INSTR_INIT, [0]) == parms.comm_init_duration
    assert proc.get_instruction_duration(INSTR_MEASURE, [0]) == parms.comm_meas_duration
    assert proc.get_instruction_duration(INSTR_ROT_X, [0]) == parms.comm_gate_duration

    for i in range(1, num_qubits):
        assert proc.get_instruction_duration(INSTR_INIT, [i]) == parms.mem_init_duration
        assert (
            proc.get_instruction_duration(INSTR_ROT_X, [i]) == parms.mem_gate_duration
        )
        assert (
            proc.get_instruction_duration(INSTR_MEASURE, [i]) == parms.mem_meas_duration
        )

    with pytest.raises(MissingInstructionError):
        proc.get_instruction_duration(INSTR_CNOT, [0, 1])
        proc.get_instruction_duration(INSTR_CXDIR, [1, 0])

    assert proc.get_instruction_duration(INSTR_CXDIR, [0, 1]) == parms.two_gate_duration


def test_build_procnode_from_config():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    latencies = LatenciesConfig(
        host_instr_time=17,
        qnos_instr_time=20,
        host_peer_latency=5,
    )
    cfg = ProcNodeConfig(
        node_name="the_node",
        node_id=42,
        topology=top_cfg,
        latencies=latencies,
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    nodes = {42: "the_node", 43: "other_node"}
    network_ehi = EhiNetworkInfo.perfect_fully_connected(nodes, duration=1000)

    procnode = build_procnode_from_config(cfg, network_ehi)

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

    assert procnode.network_ehi.get_link(42, 43) == EhiLinkInfo(1000, 1.0)


def test_build_network_from_config():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    cfg_alice = ProcNodeConfig(
        node_name="alice",
        node_id=42,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg_bob = ProcNodeConfig(
        node_name="bob",
        node_id=43,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )

    link_cfg = LinkConfig.perfect_config(state_delay=1000)
    link_ab = LinkBetweenNodesConfig(node_id1=42, node_id2=43, link_config=link_cfg)
    cfg = ProcNodeNetworkConfig(nodes=[cfg_alice, cfg_bob], links=[link_ab])
    network = build_network_from_config(cfg)

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

    alice_host_in_chan = alice_host_in.connected_port.component
    alice_host_in_remote = alice_host_in_chan.ports["send"].connected_port
    assert alice_host_in_remote == bob_host_out

    alice_host_out_chan = alice_host_out.connected_port.component
    alice_host_out_remote = alice_host_out_chan.ports["recv"].connected_port
    assert alice_host_out_remote == bob_host_in

    alice_ent_in = alice.node.entdist_in_port
    alice_ent_out = alice.node.entdist_out_port
    bob_ent_in = bob.node.entdist_in_port
    bob_ent_out = bob.node.entdist_out_port

    alice_ent_in_chan = alice_ent_in.connected_port.component
    alice_ent_in_remote = alice_ent_in_chan.ports["send"].connected_port
    assert alice_ent_in_remote == entdist.comp.node_out_port("alice")

    alice_ent_out_chan = alice_ent_out.connected_port.component
    alice_ent_out_remote = alice_ent_out_chan.ports["recv"].connected_port
    assert alice_ent_out_remote == entdist.comp.node_in_port("alice")

    bob_ent_in_chan = bob_ent_in.connected_port.component
    bob_ent_in_remote = bob_ent_in_chan.ports["send"].connected_port
    assert bob_ent_in_remote == entdist.comp.node_out_port("bob")

    bob_ent_out_chan = bob_ent_out.connected_port.component
    bob_ent_out_remote = bob_ent_out_chan.ports["recv"].connected_port
    assert bob_ent_out_remote == entdist.comp.node_in_port("bob")


def test_build_network_perfect_links():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=2)
    cfg_alice = ProcNodeConfig(
        node_name="alice",
        node_id=42,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg_bob = ProcNodeConfig(
        node_name="bob",
        node_id=43,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[cfg_alice, cfg_bob], link_duration=500
    )
    network = build_network_from_config(cfg)

    assert len(network.nodes) == 2
    assert "alice" in network.nodes
    assert "bob" in network.nodes
    assert network.entdist is not None

    entdist = network.entdist
    assert entdist.get_sampler(42, 43).delay == 500


def test_build_network_from_lhi():
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=3)
    latencies = LhiLatencies(
        host_instr_time=500,
        qnos_instr_time=1000,
        host_peer_latency=20_000,
    )
    alice_lhi = LhiProcNodeInfo(
        name="alice",
        id=42,
        topology=topology,
        latencies=latencies,
    )
    bob_lhi = LhiProcNodeInfo(
        name="bob",
        id=43,
        topology=topology,
        latencies=latencies,
    )
    nodes = {42: "alice", 43: "bob"}

    network_lhi = LhiNetworkInfo.perfect_fully_connected(nodes, 100_000)
    ntfs = [GenericNtf(), GenericNtf()]
    network = build_network_from_lhi([alice_lhi, bob_lhi], ntfs, network_lhi)

    assert len(network.nodes) == 2
    assert "alice" in network.nodes
    assert "bob" in network.nodes
    assert network.entdist is not None

    alice = network.nodes["alice"]
    entdist = network.entdist

    assert entdist.get_sampler(42, 43).delay == 100_000

    assert alice.local_ehi.latencies.host_instr_time == 500
    assert alice.local_ehi.latencies.qnos_instr_time == 1000
    assert alice.local_ehi.latencies.host_peer_latency == 20_000


def test_build_procnode_from_lhi():
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=3)
    latencies = LhiLatencies(
        host_instr_time=500,
        qnos_instr_time=1000,
        host_peer_latency=20_000,
    )
    nodes = {42: "alice", 43: "bob"}

    network_lhi = LhiNetworkInfo.perfect_fully_connected(nodes, 100_000)

    alice_procnode = build_procnode_from_lhi(
        name="alice",
        id=42,
        topology=topology,
        latencies=latencies,
        network_lhi=network_lhi,
        ntf=GenericNtf(),
    )

    assert alice_procnode.qdevice.qprocessor.name == "alice_processor"
    assert alice_procnode.qdevice.qprocessor.num_positions == 3

    assert alice_procnode.network_ehi.get_link(42, 43).duration == 100_000

    assert alice_procnode.local_ehi.latencies.host_instr_time == 500
    assert alice_procnode.local_ehi.latencies.qnos_instr_time == 1000
    assert alice_procnode.local_ehi.latencies.host_peer_latency == 20_000

    assert alice_procnode.node.netstack_peer_in_port("bob") is not None
    assert alice_procnode.node.netstack_peer_out_port("bob") is not None


if __name__ == "__main__":
    test_build_from_topology()
    test_build_perfect_topology()
    test_build_nv_perfect()
    test_build_procnode_from_config()
    test_build_network_from_config()
    test_build_network_perfect_links()
    test_build_network_from_lhi()
    test_build_procnode_from_lhi()
