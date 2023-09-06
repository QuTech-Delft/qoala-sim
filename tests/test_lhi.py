import os

import pytest
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_CZ,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel
from netsquid_magic.state_delivery_sampler import (
    DepolariseWithFailureStateSamplerFactory,
    PerfectStateSamplerFactory,
)

from qoala.lang.common import MultiQubit
from qoala.runtime.config import LatenciesConfig, LinkConfig, NvParams, TopologyConfig
from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiGateInfo,
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiQubitInfo,
    LhiTopology,
    LhiTopologyBuilder,
)
from qoala.util.math import fidelity_to_prob_max_mixed


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


def test_topology():
    comm_qubit_info = LhiQubitInfo(
        is_communication=True,
        error_model=T1T2NoiseModel,
        error_model_kwargs={"T1": 1e3, "T2": 2e3},
    )
    mem_qubit_info = LhiQubitInfo(
        is_communication=False,
        error_model=T1T2NoiseModel,
        error_model_kwargs={"T1": 1e3, "T2": 2e3},
    )
    gate_x_info = LhiGateInfo(
        instruction=INSTR_X,
        duration=1e4,
        error_model=DepolarNoiseModel,
        error_model_kwargs={
            "depolar_rate": 0.2,
            "time_independent": True,
        },
    )
    gate_y_info = LhiGateInfo(
        instruction=INSTR_Y,
        duration=1e4,
        error_model=DepolarNoiseModel,
        error_model_kwargs={
            "depolar_rate": 0.2,
            "time_independent": True,
        },
    )
    cnot_gate_info = LhiGateInfo(
        instruction=INSTR_CNOT,
        duration=1e4,
        error_model=DepolarNoiseModel,
        error_model_kwargs={
            "depolar_rate": 0.2,
            "time_independent": True,
        },
    )
    all_rot_x_gate_info = LhiGateInfo(
        instruction=INSTR_ROT_X_ALL,
        duration=1e4,
        error_model=DepolarNoiseModel,
        error_model_kwargs={
            "depolar_rate": 0.2,
            "time_independent": True,
        },
    )

    qubit_infos = {0: comm_qubit_info, 1: mem_qubit_info}
    single_gate_infos = {0: [gate_x_info, gate_y_info], 1: [gate_x_info]}
    multi_gate_infos = {MultiQubit([0, 1]): [cnot_gate_info]}
    all_qubit_gate_infos = [all_rot_x_gate_info]
    topology = LhiTopology(
        qubit_infos=qubit_infos,
        single_gate_infos=single_gate_infos,
        multi_gate_infos=multi_gate_infos,
        all_qubit_gate_infos=all_qubit_gate_infos,
    )

    assert topology.qubit_infos[0].is_communication
    assert not topology.qubit_infos[1].is_communication

    q0_gates = [info.instruction for info in topology.single_gate_infos[0]]
    assert q0_gates == [INSTR_X, INSTR_Y]

    q1_gates = [info.instruction for info in topology.single_gate_infos[1]]
    assert q1_gates == [INSTR_X]

    q01_gates = [
        info.instruction for info in topology.multi_gate_infos[MultiQubit([0, 1])]
    ]
    assert q01_gates == [INSTR_CNOT]

    assert [INSTR_ROT_X_ALL] == [
        info.instruction for info in topology.all_qubit_gate_infos
    ]
    assert all_rot_x_gate_info == topology.find_all_qubit_gate(INSTR_ROT_X_ALL)


def test_topology_from_config():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_4.yaml"))
    topology = LhiTopologyBuilder.from_config(cfg)

    assert topology.qubit_infos[0].is_communication
    assert not topology.qubit_infos[1].is_communication

    q0_gates = [info.instruction for info in topology.single_gate_infos[0]]
    assert q0_gates == [INSTR_X, INSTR_Y]

    q1_gates = [info.instruction for info in topology.single_gate_infos[1]]
    assert q1_gates == [INSTR_X]

    q01_gates = [
        info.instruction for info in topology.multi_gate_infos[MultiQubit([0, 1])]
    ]
    assert q01_gates == [INSTR_CNOT]

    assert [INSTR_ROT_X_ALL] == [
        info.instruction for info in topology.all_qubit_gate_infos
    ]


def test_topology_from_config_2():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_6.yaml"))
    topology = LhiTopologyBuilder.from_config(cfg)

    assert topology.qubit_infos[0].is_communication
    for i in [1, 2, 3]:
        assert not topology.qubit_infos[i].is_communication

    q0_gates = [info.instruction for info in topology.single_gate_infos[0]]
    assert q0_gates == [INSTR_X, INSTR_Y, INSTR_Z]

    for i in [1, 2, 3]:
        qi_gates = [info.instruction for info in topology.single_gate_infos[i]]
        assert qi_gates == [INSTR_X, INSTR_Y]

    for i in [1, 2, 3]:
        q0i_gates = [
            info.instruction for info in topology.multi_gate_infos[MultiQubit([0, i])]
        ]
        assert q0i_gates == [INSTR_CNOT]


def test_topology_from_nv_config():
    params = NvParams()
    params.comm_gate_duration = 500
    params.comm_gate_fidelity = 0.85
    cfg = TopologyConfig.from_nv_params(2, params)
    topology = LhiTopologyBuilder.from_config(cfg)

    assert topology.qubit_infos[0].is_communication
    assert not topology.qubit_infos[1].is_communication

    q0_gates = [info.instruction for info in topology.single_gate_infos[0]]
    assert all(
        instr in q0_gates
        for instr in [INSTR_ROT_X, INSTR_ROT_Y, INSTR_INIT, INSTR_MEASURE]
    )

    q1_gates = [info.instruction for info in topology.single_gate_infos[1]]
    assert all(
        instr in q1_gates
        for instr in [INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_INIT]
    )

    q01_gates = [
        info.instruction for info in topology.multi_gate_infos[MultiQubit([0, 1])]
    ]
    assert all(instr in q01_gates for instr in [INSTR_CXDIR, INSTR_CYDIR])

    expected_duration = params.comm_gate_duration
    expected_depolar_rate = fidelity_to_prob_max_mixed(1, params.comm_gate_fidelity)
    assert topology.find_single_gate(0, INSTR_ROT_X).duration == expected_duration
    assert topology.find_single_gate(0, INSTR_ROT_X).error_model_kwargs == {
        "depolar_rate": expected_depolar_rate,
        "time_independent": True,
    }


def test_find_gates():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_6.yaml"))
    topology = LhiTopologyBuilder.from_config(cfg)

    for i in range(4):
        assert topology.find_single_gate(i, INSTR_X) is not None
        assert topology.find_single_gate(i, INSTR_Y) is not None
        if i == 0:
            assert topology.find_single_gate(i, INSTR_Z) is not None
        else:
            assert topology.find_single_gate(i, INSTR_Z) is None
    assert topology.find_single_gate(4, INSTR_X) is None

    for i in range(1, 4):
        assert topology.find_multi_gate([0, i], INSTR_CNOT) is not None
        assert topology.find_multi_gate([i, 0], INSTR_CNOT) is None

    assert topology.find_multi_gate([0, 0], INSTR_CNOT) is None


def test_perfect_qubit():
    qubit_info = LhiTopologyBuilder.perfect_qubit(is_communication=True)
    assert qubit_info.is_communication
    assert qubit_info.error_model == T1T2NoiseModel
    assert qubit_info.error_model_kwargs == {"T1": 0, "T2": 0}


def test_t1t2_qubit():
    qubit_info = LhiTopologyBuilder.t1t2_qubit(is_communication=True, t1=1e3, t2=1e4)
    assert qubit_info.is_communication
    assert qubit_info.error_model == T1T2NoiseModel
    assert qubit_info.error_model_kwargs == {"T1": 1e3, "T2": 1e4}


def test_perfect_gates():
    duration = 5e3
    instructions = [INSTR_X, INSTR_Y, INSTR_Z]
    gate_infos = LhiTopologyBuilder.perfect_gates(duration, instructions)

    assert gate_infos == [
        LhiGateInfo(
            instruction=instr,
            duration=duration,
            error_model=DepolarNoiseModel,
            error_model_kwargs={"depolar_rate": 0},
        )
        for instr in instructions
    ]


def test_depolar_gates():
    duration = 5e3
    instructions = [INSTR_X, INSTR_Y, INSTR_Z]
    gate_infos = LhiTopologyBuilder.depolar_gates(
        duration, instructions, depolar_rate=0.4
    )

    assert gate_infos == [
        LhiGateInfo(
            instruction=instr,
            duration=duration,
            error_model=DepolarNoiseModel,
            error_model_kwargs={"depolar_rate": 0.4},
        )
        for instr in instructions
    ]


def test_perfect_uniform():
    num_qubits = 3
    single_qubit_instructions = [INSTR_X, INSTR_Y, INSTR_Z]
    single_gate_duration = 5e3
    two_qubit_instructions = [INSTR_CNOT, INSTR_CZ]
    two_gate_duration = 2e5
    all_qubit_instructions = [INSTR_ROT_X_ALL, INSTR_ROT_Y_ALL, INSTR_ROT_Z_ALL]
    all_qubit_duration = 1e5
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits,
        single_qubit_instructions,
        single_gate_duration,
        two_qubit_instructions,
        two_gate_duration,
        all_qubit_instructions,
        all_qubit_duration,
    )

    assert topology.qubit_infos == {
        i: LhiQubitInfo(
            is_communication=True,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": 0, "T2": 0},
        )
        for i in range(num_qubits)
    }

    assert topology.single_gate_infos == {
        i: [
            LhiGateInfo(
                instruction=instr,
                duration=single_gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": 0},
            )
            for instr in single_qubit_instructions
        ]
        for i in range(num_qubits)
    }

    assert topology.multi_gate_infos == {
        MultiQubit([i, j]): [
            LhiGateInfo(
                instruction=instr,
                duration=two_gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": 0},
            )
            for instr in two_qubit_instructions
        ]
        for i in range(num_qubits)
        for j in range(num_qubits)
        if i != j
    }

    assert topology.all_qubit_gate_infos == [
        LhiGateInfo(
            instruction=instr,
            duration=all_qubit_duration,
            error_model=DepolarNoiseModel,
            error_model_kwargs={"depolar_rate": 0},
        )
        for instr in all_qubit_instructions
    ]


def test_build_fully_uniform():
    qubit_info = LhiQubitInfo(
        is_communication=True,
        error_model=T1T2NoiseModel,
        error_model_kwargs={"T1": 1e6, "T2": 2e6},
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

    all_qubit_gate_infos = [
        LhiGateInfo(
            instruction=INSTR_ROT_X_ALL,
            duration=1e4,
            error_model=DepolarNoiseModel,
            error_model_kwargs={
                "depolar_rate": 0.2,
                "time_independent": True,
            },
        )
    ]

    topology = LhiTopologyBuilder.fully_uniform(
        num_qubits=3,
        qubit_info=qubit_info,
        single_gate_infos=single_gate_infos,
        two_gate_infos=two_gate_infos,
        all_qubit_gate_infos=all_qubit_gate_infos,
    )

    assert len(topology.qubit_infos) == 3
    for i in range(3):
        assert topology.qubit_infos[i].is_communication
        single_gates = [info.instruction for info in topology.single_gate_infos[i]]
        assert INSTR_X in single_gates
        assert INSTR_Y in single_gates
        assert INSTR_Z in single_gates

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            multi = MultiQubit([i, j])
            gates = [info.instruction for info in topology.multi_gate_infos[multi]]
            assert INSTR_CNOT in gates

    assert [INSTR_ROT_X_ALL] == [
        info.instruction for info in topology.all_qubit_gate_infos
    ]


def test_perfect_star():
    num_qubits = 3
    comm_instructions = [INSTR_X, INSTR_Y, INSTR_Z]
    comm_duration = 5e3
    mem_instructions = [INSTR_X, INSTR_Y]
    mem_duration = 1e4
    two_instructions = [INSTR_CNOT, INSTR_CZ]
    two_duration = 2e5
    topology = LhiTopologyBuilder.perfect_star(
        num_qubits,
        comm_instructions,
        comm_duration,
        mem_instructions,
        mem_duration,
        two_instructions,
        two_duration,
    )

    assert topology.qubit_infos[0] == LhiTopologyBuilder.perfect_qubit(
        is_communication=True
    )

    for i in range(1, num_qubits):
        assert topology.qubit_infos[i] == LhiTopologyBuilder.perfect_qubit(
            is_communication=False
        )

    assert topology.single_gate_infos[0] == LhiTopologyBuilder.perfect_gates(
        comm_duration, [INSTR_X, INSTR_Y, INSTR_Z]
    )
    for i in range(1, num_qubits):
        assert topology.single_gate_infos[i] == LhiTopologyBuilder.perfect_gates(
            mem_duration, [INSTR_X, INSTR_Y]
        )

    for i in range(1, num_qubits):
        assert topology.multi_gate_infos[
            MultiQubit([0, i])
        ] == LhiTopologyBuilder.perfect_gates(two_duration, [INSTR_CNOT, INSTR_CZ])


def test_generic_t1t2_star():
    num_qubits = 3
    comm_t1 = 1e8
    comm_t2 = 2e8
    mem_t1 = 1e9
    mem_t2 = 2e9
    comm_instructions = [INSTR_X, INSTR_Y, INSTR_Z]
    comm_duration = 5e3
    comm_instr_depolar_rate = 0.2
    mem_instructions = [INSTR_X, INSTR_Y]
    mem_duration = 1e4
    mem_instr_depolar_rate = 0.3
    two_instructions = [INSTR_CNOT, INSTR_CZ]
    two_duration = 2e5
    two_instr_depolar_rate = 0.4
    topology = LhiTopologyBuilder.generic_t1t2_star(
        num_qubits,
        comm_t1,
        comm_t2,
        mem_t1,
        mem_t2,
        comm_instructions,
        comm_duration,
        comm_instr_depolar_rate,
        mem_instructions,
        mem_duration,
        mem_instr_depolar_rate,
        two_instructions,
        two_duration,
        two_instr_depolar_rate,
    )

    assert topology.qubit_infos[0] == LhiTopologyBuilder.t1t2_qubit(
        is_communication=True, t1=comm_t1, t2=comm_t2
    )

    for i in range(1, num_qubits):
        assert topology.qubit_infos[i] == LhiTopologyBuilder.t1t2_qubit(
            is_communication=False, t1=mem_t1, t2=mem_t2
        )

    assert topology.single_gate_infos[0] == LhiTopologyBuilder.depolar_gates(
        comm_duration, [INSTR_X, INSTR_Y, INSTR_Z], comm_instr_depolar_rate
    )
    for i in range(1, num_qubits):
        assert topology.single_gate_infos[i] == LhiTopologyBuilder.depolar_gates(
            mem_duration, [INSTR_X, INSTR_Y], mem_instr_depolar_rate
        )

    for i in range(1, num_qubits):
        assert topology.multi_gate_infos[
            MultiQubit([0, i])
        ] == LhiTopologyBuilder.depolar_gates(
            two_duration, [INSTR_CNOT, INSTR_CZ], two_instr_depolar_rate
        )


def test_trapped_ion_default_perfect_gates():
    num_qubits = 3

    topology = LhiTopologyBuilder.trapped_ion_default_perfect_gates(num_qubits)

    single_qubit_instructions = [
        INSTR_INIT,
        INSTR_ROT_Z,
        INSTR_MEASURE,
        INSTR_MEASURE_INSTANT,
    ]

    assert topology.qubit_infos == {
        i: LhiQubitInfo(
            is_communication=True,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": 0, "T2": 0},
        )
        for i in range(num_qubits)
    }

    assert topology.single_gate_infos == {
        i: [
            LhiGateInfo(
                instruction=instr,
                duration=5e3,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": 0},
            )
            for instr in single_qubit_instructions
        ]
        for i in range(num_qubits)
    }

    assert topology.multi_gate_infos == {
        MultiQubit([i, j]): []
        for i in range(num_qubits)
        for j in range(num_qubits)
        if i != j
    }

    all_qubit_instructions = [
        INSTR_INIT,
        INSTR_MEASURE_ALL,
        INSTR_ROT_X_ALL,
        INSTR_ROT_Y_ALL,
        INSTR_ROT_Z_ALL,
        INSTR_BICHROMATIC,
    ]
    assert topology.all_qubit_gate_infos == [
        LhiGateInfo(
            instruction=instr,
            duration=5e3,
            error_model=DepolarNoiseModel,
            error_model_kwargs={"depolar_rate": 0},
        )
        for instr in all_qubit_instructions
    ]


def test_latencies():
    latencies = LhiLatencies()
    assert latencies.host_instr_time == 0
    assert latencies.qnos_instr_time == 0
    assert latencies.host_peer_latency == 0

    latencies2 = LhiLatencies.all_zero()
    assert latencies2.host_instr_time == 0
    assert latencies2.qnos_instr_time == 0
    assert latencies2.host_peer_latency == 0

    latencies3 = LhiLatencies(host_instr_time=1, qnos_instr_time=2, host_peer_latency=3)
    assert latencies3.host_instr_time == 1
    assert latencies3.qnos_instr_time == 2
    assert latencies3.host_peer_latency == 3


def test_latencies_from_config():
    cfg = LatenciesConfig(
        host_instr_time=2,
        qnos_instr_time=3,
        host_peer_latency=4,
    )
    latencies = LhiLatencies.from_config(cfg)

    assert latencies.host_instr_time == 2
    assert latencies.qnos_instr_time == 3
    assert latencies.host_peer_latency == 4


def test_link():
    link = LhiLinkInfo.perfect(5)
    assert link.state_delay == 5
    assert link.sampler_factory == PerfectStateSamplerFactory
    assert link.sampler_kwargs == {"cycle_time": 0}

    link2 = LhiLinkInfo.depolarise(
        cycle_time=10, prob_max_mixed=0.3, prob_success=0.1, state_delay=5
    )
    assert link2.state_delay == 5
    assert link2.sampler_factory == DepolariseWithFailureStateSamplerFactory
    assert link2.sampler_kwargs == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


def test_link_from_config():
    cfg = LinkConfig.from_file(relative_path("configs/link_cfg_1.yaml"))
    link = LhiLinkInfo.from_config(cfg)

    assert link.state_delay == 750
    assert link.sampler_factory == PerfectStateSamplerFactory
    assert link.sampler_kwargs == {"cycle_time": 25}


def test_link_from_config_2():
    cfg = LinkConfig.from_file(relative_path("configs/link_cfg_2.yaml"))
    link = LhiLinkInfo.from_config(cfg)

    assert link.state_delay == 750
    assert link.sampler_factory == DepolariseWithFailureStateSamplerFactory
    assert link.sampler_kwargs == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


def test_network_lhi():
    nodes = {0: "node0", 1: "node1", 2: "node2"}
    network_lhi = LhiNetworkInfo.perfect_fully_connected(nodes=nodes, duration=1000)

    assert network_lhi.get_link(0, 1) == LhiLinkInfo.perfect(duration=1000)
    assert network_lhi.get_link(0, 2) == LhiLinkInfo.perfect(duration=1000)
    assert network_lhi.get_link(1, 2) == LhiLinkInfo.perfect(duration=1000)


def test_network_lhi2():
    nodes = {0: "node0", 1: "node1", 2: "node2"}
    link_info = LhiLinkInfo.perfect(duration=10)
    network = LhiNetworkInfo.fully_connected(nodes=nodes, info=link_info)

    assert network.get_link(0, 1) == link_info
    assert network.get_link(0, 2) == link_info
    assert network.get_link(1, 2) == link_info


def test_network_lhi3():
    nodes = {0: "node0", 1: "node1", 2: "node2"}
    network = LhiNetworkInfo(nodes=nodes, links={})

    with pytest.raises(ValueError):
        network.get_link(0, 1)

    link_info = LhiLinkInfo.perfect(duration=10)
    network.add_link(0, 1, link_info)

    assert network.get_link(0, 1) == link_info

    assert network.get_link(1, 0) == network.get_link(0, 1)

    # Already existing link (order does not matter)
    with pytest.raises(ValueError):
        network.add_link(1, 0, link_info)

    # Cannot create link with itself
    with pytest.raises(ValueError):
        network.add_link(0, 0, link_info)

    # Cannot create link with non-existing node
    with pytest.raises(ValueError):
        network.add_link(0, 4, link_info)


if __name__ == "__main__":
    test_topology()
    test_topology_from_config()
    test_topology_from_config_2()
    test_topology_from_nv_config()
    test_find_gates()
    test_perfect_qubit()
    test_t1t2_qubit()
    test_perfect_gates()
    test_depolar_gates()
    test_perfect_uniform()
    test_build_fully_uniform()
    test_perfect_star()
    test_generic_t1t2_star()
    test_trapped_ion_default_perfect_gates()
    test_latencies()
    test_latencies_from_config()
    test_link()
    test_link_from_config()
    test_link_from_config_2()
    test_network_lhi()
    test_network_lhi2()
    test_network_lhi3()
