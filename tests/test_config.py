import os

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
        name="INSTR_X", noise_config_typ=GateDepolariseConfig, noise_config=noise_cfg
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
    assert cfg.gate_noise_config.to_duration() == 4e3
    assert cfg.gate_noise_config.to_error_model() == DepolarNoiseModel
    assert cfg.gate_noise_config.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


if __name__ == "__main__":
    test_qubit_t1t2_config()
    test_qubit_t1t2_config_file()
    test_gate_depolarise_config()
    test_gate_depolarise_config_file()
    test_gate_config()
    test_gate_config_file()
