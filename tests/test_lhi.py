import os

from netsquid.components.instructions import INSTR_CNOT, INSTR_X, INSTR_Y, INSTR_Z
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel

from qoala.runtime.config import TopologyConfig
from qoala.runtime.lhi import (
    LhiGateInfo,
    LhiQubitInfo,
    LhiTopology,
    LhiTopologyBuilder,
    MultiQubit,
)


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

    qubit_infos = {0: comm_qubit_info, 1: mem_qubit_info}
    single_gate_infos = {0: [gate_x_info, gate_y_info], 1: [gate_x_info]}
    multi_gate_infos = {MultiQubit([0, 1]): [cnot_gate_info]}
    topology = LhiTopology(
        qubit_infos=qubit_infos,
        single_gate_infos=single_gate_infos,
        multi_gate_infos=multi_gate_infos,
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
    topology = LhiTopologyBuilder.fully_uniform(
        num_qubits=3,
        qubit_info=qubit_info,
        single_gate_infos=single_gate_infos,
        two_gate_infos=two_gate_infos,
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


if __name__ == "__main__":
    test_topology()
    test_topology_from_config()
    test_topology_from_config_2()
    test_find_gates()
    test_build_fully_uniform()
