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

from qoala.runtime.config import GenericQDeviceConfig, NVQDeviceConfig
from qoala.runtime.lhi import LhiGateInfo, LhiQubitInfo, LhiTopology, LhiTopologyBuilder
from qoala.sim.build import (
    build_generic_qprocessor,
    build_nv_qprocessor,
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


if __name__ == "__main__":
    test_build_from_topology()
    test_build_perfect_topology()
    test_build_generic_perfect()
    test_build_nv_perfect()
