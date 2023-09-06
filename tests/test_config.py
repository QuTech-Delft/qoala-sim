from __future__ import annotations

import os
from typing import Any, Dict, Type

from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_H,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.instructions import Instruction as NetSquidInstruction
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)
from netsquid_magic.state_delivery_sampler import (
    DepolariseWithFailureStateSamplerFactory,
    PerfectStateSamplerFactory,
)

from qoala.lang.common import MultiQubit
from qoala.runtime.config import (
    AllQubitGateConfig,
    BaseModel,
    DepolariseSamplerConfig,
    GateConfig,
    GateConfigRegistry,
    GateDepolariseConfig,
    GateNoiseConfigInterface,
    InstrConfigRegistry,
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
    MultiGateConfig,
    NtfConfig,
    NvParams,
    PerfectSamplerConfig,
    ProcNodeConfig,
    QubitConfig,
    QubitConfigRegistry,
    QubitIdConfig,
    QubitNoiseConfigInterface,
    QubitT1T2Config,
    SingleGateConfig,
    TopologyConfig,
    TopologyConfigBuilder,
)
from qoala.runtime.instructions import INSTR_ROT_X_ALL
from qoala.runtime.ntf import GenericNtf, NvNtf, TrappedIonNtf
from qoala.util.math import fidelity_to_prob_max_mixed


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


def test_qubit_t1t2_config():
    cfg = QubitT1T2Config(T1=1e6, T2=3e6)

    assert cfg.T1 == 1e6
    assert cfg.T2 == 3e6


def test_qubit_t1t2_config_file():
    cfg = QubitT1T2Config.from_file(relative_path("configs/qubit_cfg_1.yaml"))

    assert cfg.T1 == 1e6
    assert cfg.T2 == 3e6


def test_qubit_config():
    noise_cfg = QubitT1T2Config(T1=1e6, T2=3e6)
    cfg = QubitConfig(
        is_communication=True,
        noise_config_cls="QubitT1T2Config",
        noise_config=noise_cfg,
    )

    assert cfg.is_communication
    assert cfg.to_is_communication()
    assert cfg.to_error_model() == T1T2NoiseModel
    assert cfg.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}


def test_qubit_config_perfect():
    for comm in [True, False]:
        cfg = QubitConfig.perfect_config(is_communication=comm)
        assert cfg.is_communication == comm
        assert cfg.to_is_communication() == comm
        assert cfg.to_error_model() == T1T2NoiseModel
        assert cfg.to_error_model_kwargs() == {"T1": 0, "T2": 0}


def test_qubit_config_file():
    cfg = QubitConfig.from_file(relative_path("configs/qubit_cfg_2.yaml"))

    assert cfg.is_communication
    assert cfg.to_is_communication()
    assert cfg.to_error_model() == T1T2NoiseModel
    assert cfg.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}


def test_gate_simple_depolarise_config():
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
    assert cfg.to_instruction() == INSTR_X
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == DepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_gate_config_perfect():
    cfg = GateConfig.perfect_config(name="INSTR_X", duration=4e3)

    assert cfg.name == "INSTR_X"
    assert cfg.to_instruction() == INSTR_X
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == DepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }


def test_gate_config_file():
    cfg = GateConfig.from_file(relative_path("configs/gate_cfg_2.yaml"))

    assert cfg.name == "INSTR_X"
    assert cfg.to_instruction() == INSTR_X
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == DepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_ntf_config():
    ntf_config = NtfConfig(ntf_interface_cls="GenericNtf", ntf_interface=GenericNtf)
    assert ntf_config.to_ntf_interface() == GenericNtf


def test_ntf_config_file():
    cfg_1 = NtfConfig.from_file(relative_path("configs/ntf_1.yaml"))
    assert cfg_1.to_ntf_interface() == GenericNtf

    cfg_2 = NtfConfig.from_file(relative_path("configs/ntf_2.yaml"))
    assert cfg_2.to_ntf_interface() == NvNtf

    cfg_2 = NtfConfig.from_file(relative_path("configs/ntf_3.yaml"))
    assert cfg_2.to_ntf_interface() == TrappedIonNtf


def test_topology_config():
    qubit_noise_cfg = QubitT1T2Config(T1=1e6, T2=3e6)
    qubit_cfg = QubitConfig(
        is_communication=True,
        noise_config_cls="QubitT1T2Config",
        noise_config=qubit_noise_cfg,
    )
    gate_noise_cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    gate_cfg = GateConfig(
        name="INSTR_X",
        noise_config_cls="GateDepolariseConfig",
        noise_config=gate_noise_cfg,
    )

    cfg = TopologyConfig(
        qubits=[QubitIdConfig(qubit_id=0, qubit_config=qubit_cfg)],
        single_gates=[SingleGateConfig(qubit_id=0, gate_configs=[gate_cfg])],
        multi_gates=[],
    )

    assert cfg.qubits[0].qubit_id == 0
    assert cfg.qubits[0].qubit_config.to_is_communication()
    assert cfg.qubits[0].qubit_config.to_error_model() == T1T2NoiseModel
    assert cfg.qubits[0].qubit_config.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert cfg.single_gates[0].qubit_id == 0
    assert cfg.single_gates[0].gate_configs[0].to_instruction() == INSTR_X
    assert cfg.single_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.single_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    assert cfg.get_qubit_configs()[0].to_is_communication()
    assert cfg.get_qubit_configs()[0].to_error_model() == T1T2NoiseModel
    assert cfg.get_qubit_configs()[0].to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert cfg.get_single_gate_configs()[0][0].to_instruction() == INSTR_X
    assert cfg.get_single_gate_configs()[0][0].to_duration() == 4e3
    assert cfg.get_single_gate_configs()[0][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[0][0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_perfect_uniform():
    cfg = TopologyConfig.perfect_config_uniform(
        num_qubits=1,
        single_instructions=["INSTR_X"],
        single_duration=4e3,
        two_instructions=[],
        two_duration=0,
    )

    assert cfg.qubits[0].qubit_id == 0
    assert cfg.qubits[0].qubit_config.to_is_communication()
    assert cfg.qubits[0].qubit_config.to_error_model() == T1T2NoiseModel
    assert cfg.qubits[0].qubit_config.to_error_model_kwargs() == {"T1": 0, "T2": 0}
    assert cfg.single_gates[0].qubit_id == 0
    assert cfg.single_gates[0].gate_configs[0].to_instruction() == INSTR_X
    assert cfg.single_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.single_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }

    # check interface
    assert cfg.get_qubit_configs()[0].to_is_communication()
    assert cfg.get_qubit_configs()[0].to_error_model() == T1T2NoiseModel
    assert cfg.get_qubit_configs()[0].to_error_model_kwargs() == {"T1": 0, "T2": 0}
    assert cfg.get_single_gate_configs()[0][0].to_instruction() == INSTR_X
    assert cfg.get_single_gate_configs()[0][0].to_duration() == 4e3
    assert cfg.get_single_gate_configs()[0][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[0][0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }


def test_topology_config_file():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_1.yaml"))

    assert cfg.qubits[0].qubit_id == 0
    assert cfg.qubits[0].qubit_config.to_is_communication()
    assert cfg.qubits[0].qubit_config.to_error_model() == T1T2NoiseModel
    assert cfg.qubits[0].qubit_config.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert cfg.single_gates[0].qubit_id == 0
    assert cfg.single_gates[0].gate_configs[0].to_instruction() == INSTR_X
    assert cfg.single_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.single_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    assert cfg.get_qubit_configs()[0].to_is_communication()
    assert cfg.get_qubit_configs()[0].to_error_model() == T1T2NoiseModel
    assert cfg.get_qubit_configs()[0].to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert cfg.get_single_gate_configs()[0][0].to_instruction() == INSTR_X
    assert cfg.get_single_gate_configs()[0][0].to_duration() == 4e3
    assert cfg.get_single_gate_configs()[0][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[0][0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_file_2():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_2.yaml"))

    assert cfg.qubits[0].qubit_id == 0
    assert cfg.qubits[0].qubit_config.to_is_communication()
    assert cfg.qubits[0].qubit_config.to_error_model() == T1T2NoiseModel
    assert cfg.qubits[0].qubit_config.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert cfg.qubits[1].qubit_id == 1
    assert not cfg.qubits[1].qubit_config.to_is_communication()
    assert cfg.qubits[1].qubit_config.to_error_model() == T1T2NoiseModel
    assert cfg.qubits[1].qubit_config.to_error_model_kwargs() == {"T1": 2e6, "T2": 4e6}
    assert cfg.single_gates[0].qubit_id == 0
    assert cfg.single_gates[0].gate_configs[0].to_instruction() == INSTR_X
    assert cfg.single_gates[0].gate_configs[0].to_duration() == 2e3
    assert cfg.single_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }
    assert cfg.single_gates[0].gate_configs[1].to_instruction() == INSTR_Y
    assert cfg.single_gates[0].gate_configs[1].to_duration() == 4e3
    assert cfg.single_gates[0].gate_configs[1].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[1].to_error_model_kwargs() == {
        "depolar_rate": 0.4,
        "time_independent": True,
    }
    assert cfg.single_gates[1].qubit_id == 1
    assert cfg.single_gates[1].gate_configs[0].to_instruction() == INSTR_Z
    assert cfg.single_gates[1].gate_configs[0].to_duration() == 6e3
    assert cfg.single_gates[1].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[1].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.6,
        "time_independent": True,
    }

    # check interface
    assert cfg.get_qubit_configs()[0].to_is_communication()
    assert cfg.get_qubit_configs()[0].to_error_model() == T1T2NoiseModel
    assert cfg.get_qubit_configs()[0].to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6}
    assert not cfg.get_qubit_configs()[1].to_is_communication()
    assert cfg.get_qubit_configs()[1].to_error_model() == T1T2NoiseModel
    assert cfg.get_qubit_configs()[1].to_error_model_kwargs() == {"T1": 2e6, "T2": 4e6}
    assert cfg.get_single_gate_configs()[0][0].to_instruction() == INSTR_X
    assert cfg.get_single_gate_configs()[0][0].to_duration() == 2e3
    assert cfg.get_single_gate_configs()[0][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[0][0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }
    assert cfg.get_single_gate_configs()[0][1].to_instruction() == INSTR_Y
    assert cfg.get_single_gate_configs()[0][1].to_duration() == 4e3
    assert cfg.get_single_gate_configs()[0][1].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[0][1].to_error_model_kwargs() == {
        "depolar_rate": 0.4,
        "time_independent": True,
    }
    assert cfg.get_single_gate_configs()[1][0].to_instruction() == INSTR_Z
    assert cfg.get_single_gate_configs()[1][0].to_duration() == 6e3
    assert cfg.get_single_gate_configs()[1][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_single_gate_configs()[1][0].to_error_model_kwargs() == {
        "depolar_rate": 0.6,
        "time_independent": True,
    }


def test_topology_config_multi_gate():
    qubit_noise_cfg = QubitT1T2Config(T1=1e6, T2=3e6)
    qubit_cfg = QubitConfig(
        is_communication=True,
        noise_config_cls="QubitT1T2Config",
        noise_config=qubit_noise_cfg,
    )
    gate_noise_cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    gate_cfg = GateConfig(
        name="INSTR_CNOT",
        noise_config_cls="GateDepolariseConfig",
        noise_config=gate_noise_cfg,
    )

    cfg = TopologyConfig(
        qubits=[
            QubitIdConfig(qubit_id=0, qubit_config=qubit_cfg),
            QubitIdConfig(qubit_id=1, qubit_config=qubit_cfg),
        ],
        single_gates=[],
        multi_gates=[MultiGateConfig(qubit_ids=[0, 1], gate_configs=[gate_cfg])],
    )

    for i in [0, 1]:
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.multi_gates[0].qubit_ids == [0, 1]
    assert cfg.multi_gates[0].gate_configs[0].to_instruction() == INSTR_CNOT
    assert cfg.multi_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.multi_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.multi_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    for i in [0, 1]:
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }
    q01 = MultiQubit([0, 1])
    assert cfg.get_multi_gate_configs()[q01][0].to_instruction() == INSTR_CNOT
    assert cfg.get_multi_gate_configs()[q01][0].to_duration() == 4e3
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_multi_gate_perfect_uniform():
    cfg = TopologyConfig.perfect_config_uniform(
        num_qubits=2,
        single_instructions=[],
        single_duration=0,
        two_instructions=["INSTR_CNOT"],
        two_duration=4e3,
    )

    for i in [0, 1]:
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 0,
            "T2": 0,
        }

    assert cfg.multi_gates[0].qubit_ids == [0, 1]
    assert cfg.multi_gates[0].gate_configs[0].to_instruction() == INSTR_CNOT
    assert cfg.multi_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.multi_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.multi_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }

    # check interface
    for i in [0, 1]:
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 0,
            "T2": 0,
        }
    q01 = MultiQubit([0, 1])
    assert cfg.get_multi_gate_configs()[q01][0].to_instruction() == INSTR_CNOT
    assert cfg.get_multi_gate_configs()[q01][0].to_duration() == 4e3
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }


def test_topology_config_multi_gate_perfect_star():
    cfg = TopologyConfig.perfect_config_star(
        num_qubits=3,
        comm_instructions=["INSTR_X"],
        comm_duration=5e3,
        mem_instructions=["INSTR_Y", "INSTR_Z"],
        mem_duration=10e3,
        two_instructions=["INSTR_CNOT"],
        two_duration=200e3,
    )

    for i, comm in zip([0, 1, 2], [True, False, False]):
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication() == comm
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 0,
            "T2": 0,
        }

    assert cfg.single_gates[0].qubit_id == 0
    assert len(cfg.single_gates[0].gate_configs) == 1
    assert cfg.single_gates[0].gate_configs[0].to_instruction() == INSTR_X
    assert cfg.single_gates[0].gate_configs[0].to_duration() == 5e3
    assert cfg.single_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.single_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }

    for i in [1, 2]:
        assert cfg.single_gates[i].qubit_id == i
        assert len(cfg.single_gates[i].gate_configs) == 2
        assert cfg.single_gates[i].gate_configs[0].to_instruction() == INSTR_Y
        assert cfg.single_gates[i].gate_configs[1].to_instruction() == INSTR_Z
        for j in [0, 1]:
            assert cfg.single_gates[i].gate_configs[j].to_duration() == 10e3
            assert (
                cfg.single_gates[i].gate_configs[j].to_error_model()
                == DepolarNoiseModel
            )
            assert cfg.single_gates[i].gate_configs[j].to_error_model_kwargs() == {
                "depolar_rate": 0,
                "time_independent": True,
            }

    assert len(cfg.multi_gates) == 2
    for i in [1, 2]:
        assert cfg.multi_gates[i - 1].qubit_ids == [0, i]
        assert len(cfg.multi_gates[i - 1].gate_configs) == 1
        assert len(cfg.multi_gates[i - 1].gate_configs) == 1
        assert cfg.multi_gates[i - 1].gate_configs[0].to_instruction() == INSTR_CNOT
        assert cfg.multi_gates[i - 1].gate_configs[0].to_duration() == 200e3
        assert (
            cfg.multi_gates[i - 1].gate_configs[0].to_error_model() == DepolarNoiseModel
        )
        assert cfg.multi_gates[i - 1].gate_configs[0].to_error_model_kwargs() == {
            "depolar_rate": 0,
            "time_independent": True,
        }


def test_topology_config_file_multi_gate():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_3.yaml"))

    for i in [0, 1]:
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.multi_gates[0].qubit_ids == [0, 1]
    assert cfg.multi_gates[0].gate_configs[0].to_instruction() == INSTR_CNOT
    assert cfg.multi_gates[0].gate_configs[0].to_duration() == 4e3
    assert cfg.multi_gates[0].gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.multi_gates[0].gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    for i in [0, 1]:
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }
    q01 = MultiQubit([0, 1])
    assert cfg.get_multi_gate_configs()[q01][0].to_instruction() == INSTR_CNOT
    assert cfg.get_multi_gate_configs()[q01][0].to_duration() == 4e3
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model() == DepolarNoiseModel
    assert cfg.get_multi_gate_configs()[q01][0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_all_qubit_gate():
    qubit_noise_cfg = QubitT1T2Config(T1=1e6, T2=3e6)
    qubit_cfg = QubitConfig(
        is_communication=True,
        noise_config_cls="QubitT1T2Config",
        noise_config=qubit_noise_cfg,
    )
    gate_noise_cfg = GateDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    gate_cfg = GateConfig(
        name="INSTR_ROT_X_ALL",
        noise_config_cls="GateDepolariseConfig",
        noise_config=gate_noise_cfg,
    )

    cfg = TopologyConfig(
        qubits=[
            QubitIdConfig(qubit_id=0, qubit_config=qubit_cfg),
            QubitIdConfig(qubit_id=1, qubit_config=qubit_cfg),
        ],
        single_gates=[],
        multi_gates=[],
        all_qubit_gates=AllQubitGateConfig(gate_configs=[gate_cfg]),
    )

    for i in [0, 1]:
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.all_qubit_gates.gate_configs[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.all_qubit_gates.gate_configs[0].to_duration() == 4e3
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    for i in [0, 1]:
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.get_all_qubit_gate_configs()[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.get_all_qubit_gate_configs()[0].to_duration() == 4e3
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model() == DepolarNoiseModel
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_all_qubit_gates_perfect_uniform():
    cfg = TopologyConfig.perfect_config_uniform(
        num_qubits=3,
        single_instructions=[],
        single_duration=0,
        two_instructions=[],
        two_duration=0,
        all_qubit_gate_instructions=["INSTR_ROT_X_ALL"],
        all_qubit_gate_duration=4e3,
    )

    for i in range(3):
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 0,
            "T2": 0,
        }

    assert cfg.all_qubit_gates.gate_configs[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.all_qubit_gates.gate_configs[0].to_duration() == 4e3
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }

    # check interface
    for i in range(3):
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 0,
            "T2": 0,
        }

    assert cfg.get_all_qubit_gate_configs()[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.get_all_qubit_gate_configs()[0].to_duration() == 4e3
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model() == DepolarNoiseModel
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model_kwargs() == {
        "depolar_rate": 0,
        "time_independent": True,
    }


def test_topology_config_file_all_qubit_gates():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_7.yaml"))

    for i in [0, 1]:
        assert cfg.qubits[i].qubit_id == i
        assert cfg.qubits[i].qubit_config.to_is_communication()
        assert cfg.qubits[i].qubit_config.to_error_model() == T1T2NoiseModel
        assert cfg.qubits[i].qubit_config.to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.all_qubit_gates.gate_configs[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.all_qubit_gates.gate_configs[0].to_duration() == 4e3
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model() == DepolarNoiseModel
    assert cfg.all_qubit_gates.gate_configs[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }

    # check interface
    for i in [0, 1]:
        assert cfg.get_qubit_configs()[i].to_is_communication()
        assert cfg.get_qubit_configs()[i].to_error_model() == T1T2NoiseModel
        assert cfg.get_qubit_configs()[i].to_error_model_kwargs() == {
            "T1": 1e6,
            "T2": 3e6,
        }

    assert cfg.get_all_qubit_gate_configs()[0].to_instruction() == INSTR_ROT_X_ALL
    assert cfg.get_all_qubit_gate_configs()[0].to_duration() == 4e3
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model() == DepolarNoiseModel
    assert cfg.get_all_qubit_gate_configs()[0].to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_topology_config_file_reuse_gate_def():
    cfg = TopologyConfig.from_file(relative_path("configs/topology_cfg_5.yaml"))

    assert cfg.get_single_gate_configs()[0][0].to_instruction() == INSTR_X
    assert cfg.get_single_gate_configs()[0][1].to_instruction() == INSTR_Y
    assert cfg.get_single_gate_configs()[1][0].to_instruction() == INSTR_X


def test_topology_builder():
    topology = (
        TopologyConfigBuilder()
        .uniform_topology()
        .num_qubits(2)
        .no_decoherence()
        .default_generic_gates()
        .zero_gate_durations()
        .perfect_gate_fidelities()
        .comm_gate_fidelity("INSTR_X", 0.95)
        .build()
    )
    for single_gate_cfg in topology.single_gates:
        for gate_cfg in single_gate_cfg.gate_configs:
            if gate_cfg.name == "INSTR_X":
                expected = fidelity_to_prob_max_mixed(1, 0.95)
                assert gate_cfg.to_error_model_kwargs()["depolar_rate"] == expected


def test_topology_from_nv_params():
    params = NvParams()
    params.mem_t2 = 123
    params.comm_init_fidelity = 0.85
    params.two_gate_duration = 500_000

    cfg = TopologyConfig.from_nv_params(2, params)

    # Check T2 of mem qubit
    mem_qubit_cfg = cfg.qubits[1].qubit_config
    assert mem_qubit_cfg.noise_config.to_error_model_kwargs()["T2"] == 123

    # Check init fidelity on comm qubit (id 0)
    for gate_cfg in cfg.single_gates[0].gate_configs:
        if gate_cfg.name == "INSTR_INIT":
            assert gate_cfg.noise_config.to_error_model_kwargs()[
                "depolar_rate"
            ] == fidelity_to_prob_max_mixed(1, 0.85)

    # Check two-qubit gate duration
    for gate_cfg in cfg.multi_gates[0].gate_configs:
        assert gate_cfg.noise_config.to_duration() == 500_000


def test_qubit_config_file_registry():
    class QubitT1T2T3Config(QubitT1T2Config):
        T3: int

        @classmethod
        def from_dict(cls, dict: Any) -> QubitT1T2T3Config:
            return QubitT1T2T3Config(**dict)

        def to_error_model_kwargs(self) -> Dict[str, Any]:
            return {"T1": self.T1, "T2": self.T2, "T3": self.T3}

    class CustomQubitConfigRegistry(QubitConfigRegistry):
        @classmethod
        def map(cls) -> Dict[str, QubitNoiseConfigInterface]:
            return {"QubitT1T2T3Config": QubitT1T2T3Config}

    cfg = QubitConfig.from_file(
        relative_path("configs/qubit_cfg_2_custom_cls.yaml"),
        registry=[CustomQubitConfigRegistry],
    )

    assert cfg.is_communication
    assert cfg.to_is_communication()
    assert cfg.to_error_model() == T1T2NoiseModel
    assert cfg.to_error_model_kwargs() == {"T1": 1e6, "T2": 3e6, "T3": 5e6}


def test_gate_config_file_registry():
    class MyLeetNoise(GateNoiseConfigInterface, BaseModel):
        my_noise_param: int

        @classmethod
        def from_dict(cls, dict: Any) -> MyLeetNoise:
            return MyLeetNoise(**dict)

        def to_duration(self) -> int:
            return 42

        def to_error_model(self) -> Type[QuantumErrorModel]:
            return QuantumErrorModel

        def to_error_model_kwargs(self) -> Dict[str, Any]:
            return {"my_noise_param": self.my_noise_param}

    class CustomGateConfigRegistry(GateConfigRegistry):
        @classmethod
        def map(cls) -> Dict[str, GateNoiseConfigInterface]:
            return {"MyLeetNoise": MyLeetNoise}

    cfg = GateConfig.from_file(
        relative_path("configs/gate_cfg_2_custom_cls.yaml"),
        registry=[CustomGateConfigRegistry],
    )

    assert cfg.name == "INSTR_X"
    assert cfg.to_instruction() == INSTR_X
    assert cfg.to_duration() == 42
    assert cfg.to_error_model() == QuantumErrorModel
    assert cfg.to_error_model_kwargs() == {
        "my_noise_param": 1337,
    }


def test_custom_instruction():
    class CustomInstrRegistry(InstrConfigRegistry):
        @classmethod
        def map(cls) -> Dict[str, NetSquidInstruction]:
            return {"MY_CUSTOM_INSTR": INSTR_H}

    cfg = GateConfig.from_file(relative_path("configs/gate_cfg_2_custom_instr.yaml"))

    assert cfg.name == "MY_CUSTOM_INSTR"
    assert cfg.to_instruction(registry=[CustomInstrRegistry]) == INSTR_H
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == DepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_latencies_config_file():
    cfg = LatenciesConfig.from_file(relative_path("configs/latencies_cfg_1.yaml"))

    assert cfg.host_instr_time == 500
    assert cfg.qnos_instr_time == 2000
    assert cfg.host_peer_latency == 2e6

    # check interface
    assert cfg.get_host_instr_time() == 500
    assert cfg.get_qnos_instr_time() == 2000
    assert cfg.get_host_peer_latency() == 2e6


def test_latencies_config_file_default_values():
    cfg = LatenciesConfig.from_file(relative_path("configs/latencies_cfg_2.yaml"))

    # explicitly given by cfg file
    assert cfg.host_instr_time == 200

    # not given in the cfg file, so they should default to 0
    assert cfg.qnos_instr_time == 0
    assert cfg.host_peer_latency == 0

    # check interface
    assert cfg.get_host_instr_time() == 200
    assert cfg.get_qnos_instr_time() == 0
    assert cfg.get_host_peer_latency() == 0


def test_procnode_config_file():
    cfg = ProcNodeConfig.from_file(relative_path("configs/procnode_cfg_1.yaml"))

    # the topology used in this file is the same as in configs/topology_cfg_1.yaml
    expected_topology = TopologyConfig.from_file(
        relative_path("configs/topology_cfg_1.yaml")
    )

    assert cfg.node_name == "client_node"
    assert cfg.node_id == 2
    assert cfg.topology == expected_topology
    assert cfg.latencies.host_instr_time == 500
    assert cfg.latencies.qnos_instr_time == 2000
    assert cfg.latencies.host_peer_latency == 2e6
    assert cfg.ntf.to_ntf_interface() == GenericNtf


def test_procnode_config_file_default_values():
    cfg = ProcNodeConfig.from_file(relative_path("configs/procnode_cfg_2.yaml"))

    # following 2 items are not given in the cfg file, so they should default to 0
    assert cfg.latencies.host_instr_time == 0
    assert cfg.latencies.qnos_instr_time == 0

    # explicitly given by cfg file
    assert cfg.latencies.host_peer_latency == 2e6

    assert cfg.ntf.to_ntf_interface() == NvNtf


def test_perfect_sampler_config():
    cfg = PerfectSamplerConfig(cycle_time=10)

    assert cfg.to_sampler_factory() == PerfectStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {"cycle_time": 10}


def test_depolarise_sampler_config():
    cfg = DepolariseSamplerConfig(cycle_time=10, prob_max_mixed=0.3, prob_success=0.1)

    assert cfg.to_sampler_factory() == DepolariseWithFailureStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


def test_link_config():
    sampler_cfg = PerfectSamplerConfig(cycle_time=10)
    cfg = LinkConfig(
        state_delay=500,
        sampler_config_cls="PerfectSamplerConfig",
        sampler_config=sampler_cfg,
    )

    assert cfg.state_delay == 500
    assert cfg.to_state_delay() == 500
    assert cfg.to_sampler_factory() == PerfectStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {"cycle_time": 10}


def test_link_config_2():
    sampler_cfg = DepolariseSamplerConfig(
        cycle_time=10, prob_max_mixed=0.3, prob_success=0.1
    )
    cfg = LinkConfig(
        state_delay=500,
        sampler_config_cls="DepolariseSamplerConfig",
        sampler_config=sampler_cfg,
    )

    assert cfg.state_delay == 500
    assert cfg.to_state_delay() == 500
    assert cfg.to_sampler_factory() == DepolariseWithFailureStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


def test_link_config_perfect():
    cfg = LinkConfig.perfect_config(200)

    assert cfg.state_delay == 200
    assert cfg.to_state_delay() == 200
    assert cfg.to_sampler_factory() == PerfectStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {"cycle_time": 0}


def test_link_config_depolarise():
    cfg = LinkConfig.simple_depolarise_config(fidelity=0.8, state_delay=200)

    assert cfg.to_state_delay() == 200
    assert cfg.to_sampler_factory() == DepolariseWithFailureStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {
        "cycle_time": 0,
        "prob_max_mixed": fidelity_to_prob_max_mixed(2, 0.8),
        "prob_success": 1,
    }


def test_link_config_from_file():
    cfg = LinkConfig.from_file(relative_path("configs/link_cfg_1.yaml"))

    assert cfg.state_delay == 750
    assert cfg.to_state_delay() == 750
    assert cfg.to_sampler_factory() == PerfectStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {"cycle_time": 25}


def test_link_config_from_file_2():
    cfg = LinkConfig.from_file(relative_path("configs/link_cfg_2.yaml"))

    assert cfg.state_delay == 750
    assert cfg.to_state_delay() == 750
    assert cfg.to_sampler_factory() == DepolariseWithFailureStateSamplerFactory
    assert cfg.to_sampler_kwargs() == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


def test_link_between_nodes_from_file():
    cfg = LinkBetweenNodesConfig.from_file(relative_path("configs/link_cfg_3.yaml"))

    assert cfg.node_id1 == 2
    assert cfg.node_id2 == 5
    assert cfg.link_config.state_delay == 750
    assert cfg.link_config.to_state_delay() == 750
    assert (
        cfg.link_config.to_sampler_factory() == DepolariseWithFailureStateSamplerFactory
    )
    assert cfg.link_config.to_sampler_kwargs() == {
        "cycle_time": 10,
        "prob_max_mixed": 0.3,
        "prob_success": 0.1,
    }


if __name__ == "__main__":
    test_qubit_t1t2_config()
    test_qubit_t1t2_config_file()
    test_qubit_config()
    test_qubit_config_perfect()
    test_qubit_config_file()
    test_gate_simple_depolarise_config()
    test_gate_depolarise_config_file()
    test_gate_config()
    test_gate_config_perfect()
    test_gate_config_file()
    test_ntf_config()
    test_ntf_config_file()
    test_topology_config()
    test_topology_config_perfect_uniform()
    test_topology_config_file()
    test_topology_config_file_2()
    test_topology_config_multi_gate()
    test_topology_config_file_multi_gate()
    test_topology_config_multi_gate_perfect_uniform()
    test_topology_config_multi_gate_perfect_star()
    test_topology_config_all_qubit_gate()
    test_topology_config_all_qubit_gates_perfect_uniform()
    test_topology_config_file_all_qubit_gates()
    test_topology_config_file_reuse_gate_def()
    test_topology_builder()
    test_topology_from_nv_params()
    test_qubit_config_file_registry()
    test_gate_config_file_registry()
    test_custom_instruction()
    test_latencies_config_file()
    test_latencies_config_file_default_values()
    test_procnode_config_file()
    test_procnode_config_file_default_values()
    test_perfect_sampler_config()
    test_depolarise_sampler_config()
    test_link_config()
    test_link_config_2()
    test_link_config_perfect()
    test_link_config_depolarise()
    test_link_config_from_file()
    test_link_config_from_file_2()
    test_link_between_nodes_from_file()
