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
    QubitT1T2Config,
    TopologyConfig,
)


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


def test_qubit_t1t2_config():
    cfg = QubitT1T2Config(is_communication=True, T1=1e6, T2=3e6)

    assert cfg.is_communication
    assert cfg.T1 == 1e6
    assert cfg.T2 == 3e6


def test_qubit_t1t2_config_file():
    cfg = QubitT1T2Config.from_file(relative_path("configs/qubit_cfg_1.yaml"))

    assert cfg.is_communication
    assert cfg.T1 == 1e6
    assert cfg.T2 == 3e6


def test_gate_depolarise_config():
    cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)

    assert cfg.duration == 4e3
    assert cfg.depolarise_prob == 0.2


def test_gate_depolarise_config_file():
    cfg = GateDepolariseConfig.from_file(relative_path("configs/gate_cfg_1.yaml"))

    assert cfg.duration == 4e3
    assert cfg.depolarise_prob == 0.2
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == DepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_gate_config():
    noise_cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    cfg = GateConfig(
        name="INSTR_X", noise_config_cls="GateDepolariseConfig", noise_config=noise_cfg
    )

    assert cfg.name == "INSTR_X"
    assert cfg.noise_config.to_duration() == 4e3
    assert cfg.noise_config.to_error_model() == DepolarNoiseModel
    assert cfg.noise_config.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_gate_config_file():
    cfg = GateConfig.from_file(relative_path("configs/gate_cfg_2.yaml"))

    assert cfg.name == "INSTR_X"
    assert cfg.noise_config.to_duration() == 4e3
    assert cfg.noise_config.to_error_model() == DepolarNoiseModel
    assert cfg.noise_config.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config():
    qubit_cfg = QubitT1T2Config(is_communication=True, T1=1e6, T2=3e6)
    gate_noise_cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    gate_cfg = GateConfig(
        name="INSTR_X",
        noise_config_cls="GateDepolariseConfig",
        noise_config=gate_noise_cfg,
    )

    cfg = TopologyConfig(
        qubits={0: qubit_cfg}, single_gates={0: [gate_cfg]}, multi_gates={}
    )

    assert cfg.qubits[0].is_communication
    assert cfg.single_gates[0][0].to_instruction() == INSTR_X


def test_topology_config_file():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_1.yaml"))

    assert cfg.qubits[0].is_communication
    assert cfg.single_gates[0][0].to_instruction() == INSTR_X
    assert cfg.single_gates[0][0].to_duration() == 4e3


if __name__ == "__main__":
    # test_qubit_t1t2_config()
    # test_qubit_t1t2_config_file()
    # test_gate_depolarise_config()
    # test_gate_depolarise_config_file()
    # test_gate_config()
    # test_gate_config_file()
    # test_topology_config()
    test_topology_config_file()
