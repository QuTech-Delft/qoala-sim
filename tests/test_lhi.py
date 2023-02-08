import os

from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CZ,
    INSTR_INIT,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)

from qoala.runtime.config import (
    DepolariseLinkConfig,
    GateConfig,
    GateDepolariseConfig,
    MultiGateConfig,
    QubitConfig,
    QubitIdConfig,
    QubitT1T2Config,
    SingleGateConfig,
    TopologyConfig,
)
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


if __name__ == "__main__":
    test_topology()
    test_topology_from_config()
    test_topology_from_config_2()
