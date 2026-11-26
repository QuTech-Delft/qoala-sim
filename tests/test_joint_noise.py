"""Tests for qoala.runtime.noise.JointDepolarNoiseModel.

Background: NetSquid's DepolarNoiseModel depolarizes each qubit of a two-qubit
gate independently, while Qoala's fidelity -> probability conversion
(fidelity_to_prob_max_mixed) assumes the *joint* two-qubit channel
rho -> (1 - p) rho + p I/4. Configuring a two-qubit gate with fidelity F used to
give F_bell = 1 - 3p/2 + 3p^2/4 and F_00 = (1 - p/2)^2 instead of F.
See the docstring of qoala.runtime.noise for details.
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings

import netsquid as ns
import numpy as np
import pytest
from netsquid.components.instructions import INSTR_CNOT, INSTR_H, INSTR_I, INSTR_INIT
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction, QuantumProcessor
from netsquid.qubits import ketstates, qubitapi

from qoala.runtime.config import (
    AllQubitGateConfig,
    DefaultGateConfigRegistry,
    GateConfig,
    GateDepolariseConfig,
    GateJointDepolariseConfig,
    MultiGateConfig,
    QubitConfig,
    QubitIdConfig,
    SingleGateConfig,
    TopologyConfig,
)
from qoala.runtime.lhi import LhiGateInfo, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.noise import (
    JointDepolarNoiseModel,
    PerQubitDepolarNoiseWarning,
    joint_depolarize,
)
from qoala.runtime.ntf import GenericNtf
from qoala.sim.build import build_qprocessor_from_topology
from qoala.util.math import (
    B00_DENS,
    S00_DENS,
    TWO_MAX_MIXED,
    fidelity_to_prob_max_mixed,
    prob_max_mixed_to_fidelity,
)

PROBS = [0.0, 0.0133, 0.0667, 0.1333, 0.5, 1.0]
FIDELITIES = [0.99, 0.95, 0.90]


@pytest.fixture(autouse=True)
def dm_formalism():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)
    yield
    ns.sim_reset()


def make_bell_pair():
    q0, q1 = qubitapi.create_qubits(2)
    qubitapi.operate(q0, ns.H)
    qubitapi.operate([q0, q1], ns.CNOT)
    return q0, q1


def make_00():
    q0, q1 = qubitapi.create_qubits(2)
    qubitapi.combine_qubits([q0, q1])
    return q0, q1


# ---------------------------------------------------------------------------
# The channel itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prob", PROBS)
@pytest.mark.parametrize(
    "make_state, ideal_dm",
    [(make_bell_pair, B00_DENS), (make_00, S00_DENS)],
    ids=["bell", "00"],
)
def test_joint_depolarize_density_matrix(prob, make_state, ideal_dm):
    """The output is exactly (1 - p) rho + p I/4."""
    q0, q1 = make_state()
    joint_depolarize([q0, q1], prob)

    expected = (1 - prob) * ideal_dm + prob * TWO_MAX_MIXED
    assert np.allclose(qubitapi.reduced_dm([q0, q1]), expected)


@pytest.mark.parametrize("prob", PROBS)
@pytest.mark.parametrize(
    "make_state, ket",
    [(make_bell_pair, ketstates.b00), (make_00, ketstates.s00)],
    ids=["bell", "00"],
)
def test_joint_depolarize_fidelity_is_state_independent(prob, make_state, ket):
    """Both entangled and product states end with F = 1 - 3p/4."""
    q0, q1 = make_state()
    JointDepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation(
        [q0, q1]
    )
    fid = qubitapi.fidelity([q0, q1], ket, squared=True)
    assert fid == pytest.approx(prob_max_mixed_to_fidelity(2, prob))


@pytest.mark.parametrize("prob", [0.0133, 0.0667, 0.1333])
def test_netsquid_per_qubit_model_differs(prob):
    """Regression guard documenting the original bug.

    NetSquid's DepolarNoiseModel gives the state-dependent per-qubit fidelities
    from the bug report, which differ from the configured 1 - 3p/4. If this test
    starts failing, NetSquid changed its model and the need for
    JointDepolarNoiseModel should be re-evaluated.
    """
    model = DepolarNoiseModel(depolar_rate=prob, time_independent=True)

    q0, q1 = make_bell_pair()
    model.error_operation([q0, q1])
    f_bell = qubitapi.fidelity([q0, q1], ketstates.b00, squared=True)
    assert f_bell == pytest.approx((1 - 3 * prob / 4) ** 2 + 3 * (prob / 4) ** 2)

    q0, q1 = make_00()
    model.error_operation([q0, q1])
    f_00 = qubitapi.fidelity([q0, q1], ketstates.s00, squared=True)
    assert f_00 == pytest.approx((1 - prob / 2) ** 2)

    configured = prob_max_mixed_to_fidelity(2, prob)
    assert f_bell < configured
    assert f_00 < configured


@pytest.mark.parametrize("prob", PROBS)
def test_single_qubit_matches_netsquid(prob):
    """On one qubit the joint and per-qubit models are the same channel."""
    q_joint, q_ns = qubitapi.create_qubits(2)
    qubitapi.operate(q_joint, ns.H)
    qubitapi.operate(q_ns, ns.H)

    JointDepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation(
        [q_joint]
    )
    DepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation([q_ns])
    assert np.allclose(qubitapi.reduced_dm(q_joint), qubitapi.reduced_dm(q_ns))


def test_three_qubit_joint_channel():
    """The model generalizes to n qubits: rho -> (1 - p) rho + p I/8."""
    prob = 0.3
    qubits = qubitapi.create_qubits(3)
    qubitapi.operate(qubits[0], ns.H)
    qubitapi.operate([qubits[0], qubits[1]], ns.CNOT)
    qubitapi.operate([qubits[1], qubits[2]], ns.CNOT)
    ghz = np.zeros((8, 1))
    ghz[0] = ghz[7] = 1 / math.sqrt(2)

    JointDepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation(
        qubits
    )
    expected = (1 - prob) * np.outer(ghz, ghz.conj()) + prob * np.eye(8) / 8
    assert np.allclose(qubitapi.reduced_dm(qubits), expected)


def test_none_qubits_are_skipped():
    """Empty positions (None) are ignored, as in NetSquid's model."""
    prob = 0.4
    q0, q1 = make_bell_pair()
    JointDepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation(
        [q0, None, q1]
    )
    expected = (1 - prob) * B00_DENS + prob * TWO_MAX_MIXED
    assert np.allclose(qubitapi.reduced_dm([q0, q1]), expected)

    # Only None: no-op, no error.
    JointDepolarNoiseModel(depolar_rate=prob, time_independent=True).error_operation(
        [None, None]
    )


def test_time_dependent_mode():
    """With time_independent=False, p = 1 - exp(-dt * rate * 1e-9)."""
    rate = 1e6  # Hz
    delta_time = 500  # ns
    prob = 1 - math.exp(-delta_time * rate * 1e-9)

    q0, q1 = make_bell_pair()
    JointDepolarNoiseModel(depolar_rate=rate).error_operation(
        [q0, q1], delta_time=delta_time
    )
    expected = (1 - prob) * B00_DENS + prob * TWO_MAX_MIXED
    assert np.allclose(qubitapi.reduced_dm([q0, q1]), expected)


def test_invalid_probability():
    q0, q1 = make_bell_pair()
    with pytest.raises(ValueError):
        joint_depolarize([q0, q1], 1.5)
    with pytest.raises(ValueError):
        joint_depolarize([q0, q1], -0.1)


def test_ket_formalism_smoke():
    """In KET formalism one Pauli is sampled; the output stays a valid state."""
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)
    ns.set_random_state(seed=42)
    for _ in range(20):
        q0, q1 = make_bell_pair()
        joint_depolarize([q0, q1], 1.0)
        dm = qubitapi.reduced_dm([q0, q1])
        assert np.trace(dm) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Inside a NetSquid QuantumProcessor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_cnot_in_qprocessor(fidelity):
    """A noisy CNOT preparing a Bell pair has exactly the configured fidelity."""
    prob = fidelity_to_prob_max_mixed(2, fidelity)
    gate_noise = JointDepolarNoiseModel(depolar_rate=prob, time_independent=True)
    processor = QuantumProcessor(
        "processor",
        num_positions=2,
        phys_instructions=[
            PhysicalInstruction(INSTR_INIT, duration=1),
            PhysicalInstruction(INSTR_H, duration=1),
            PhysicalInstruction(INSTR_I, duration=1, quantum_noise_model=gate_noise),
            PhysicalInstruction(INSTR_CNOT, duration=1, quantum_noise_model=gate_noise),
        ],
    )
    for pos in (0, 1):
        processor.execute_instruction(INSTR_INIT, [pos])
        ns.sim_run()
    processor.execute_instruction(INSTR_H, [0])
    ns.sim_run()
    processor.execute_instruction(INSTR_CNOT, [0, 1])
    ns.sim_run()

    q0, q1 = processor.peek([0, 1])
    assert qubitapi.fidelity([q0, q1], ketstates.b00, squared=True) == pytest.approx(
        fidelity
    )


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_gate_joint_depolarise_config():
    cfg = GateJointDepolariseConfig(duration=4e3, depolarise_prob=0.2)
    assert cfg.to_duration() == 4e3
    assert cfg.to_error_model() == JointDepolarNoiseModel
    assert cfg.to_error_model_kwargs() == {
        "depolar_rate": 0.2,
        "time_independent": True,
    }


def test_gate_joint_depolarise_config_from_dict():
    """The new config is in the default registry, so YAML configs can use it."""
    assert (
        DefaultGateConfigRegistry.map()["GateJointDepolariseConfig"]
        == GateJointDepolariseConfig
    )
    cfg = GateConfig.from_dict(
        {
            "name": "INSTR_CNOT",
            "noise_config_cls": "GateJointDepolariseConfig",
            "noise_config": {"duration": 4e3, "depolarise_prob": 0.2},
        }
    )
    assert cfg.to_instruction() == INSTR_CNOT
    assert cfg.to_error_model() == JointDepolarNoiseModel
    assert cfg.to_error_model_kwargs()["depolar_rate"] == 0.2


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_with_two_gate_fidelity_uses_joint_model(fidelity):
    cfg = GateConfig.with_two_gate_fidelity("INSTR_CNOT", 1000, fidelity)
    assert cfg.noise_config_cls == "GateJointDepolariseConfig"
    assert cfg.to_error_model() == JointDepolarNoiseModel
    assert cfg.to_error_model_kwargs()["depolar_rate"] == pytest.approx(
        (1 - fidelity) * 4 / 3
    )


def test_with_all_qubit_gate_fidelity():
    one = GateConfig.with_all_qubit_gate_fidelity("INSTR_X", 1000, 0.99, num_qubits=1)
    assert isinstance(one.noise_config, GateDepolariseConfig)
    assert one.to_error_model() == DepolarNoiseModel

    two = GateConfig.with_all_qubit_gate_fidelity(
        "INSTR_CNOT", 1000, 0.99, num_qubits=2
    )
    assert isinstance(two.noise_config, GateJointDepolariseConfig)
    assert two.to_error_model() == JointDepolarNoiseModel


def test_with_single_gate_fidelity_unchanged():
    cfg = GateConfig.with_single_gate_fidelity("INSTR_X", 1000, 0.99)
    assert cfg.noise_config_cls == "GateDepolariseConfig"
    assert cfg.to_error_model() == DepolarNoiseModel


def test_lhi_to_ehi_accepts_joint_model():
    info = LhiGateInfo(
        instruction=INSTR_CNOT,
        duration=1000,
        error_model=JointDepolarNoiseModel,
        error_model_kwargs={"depolar_rate": 0.25, "time_independent": True},
    )
    ehi = LhiConverter.gate_info_to_ehi(info, GenericNtf())
    assert ehi.decoherence == 0.25


# ---------------------------------------------------------------------------
# End to end: TopologyConfig -> LhiTopology -> QuantumProcessor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fidelity", FIDELITIES)
@pytest.mark.parametrize(
    "apply_h, ket", [(True, ketstates.b00), (False, ketstates.s00)], ids=["bell", "00"]
)
def test_end_to_end_two_gate_fidelity(fidelity, apply_h, ket):
    """A CNOT built from a configured two-qubit fidelity reaches that fidelity,
    for both an entangled (Bell pair) and a product (|00>) output."""
    cfg = TopologyConfig.uniform_t1t2_qubits_uniform_imperfect_gates(
        num_qubits=2,
        t1=0,  # no memory noise
        t2=0,
        single_instructions=["INSTR_INIT", "INSTR_H"],
        single_duration=1,
        single_fid=1.0,
        two_instructions=["INSTR_CNOT"],
        two_duration=1,
        two_fid=fidelity,
    )
    topology = LhiTopologyBuilder.from_config(cfg)
    proc = build_qprocessor_from_topology("proc", topology)

    for pos in (0, 1):
        proc.execute_instruction(INSTR_INIT, [pos])
        ns.sim_run()
    if apply_h:
        proc.execute_instruction(INSTR_H, [0])
        ns.sim_run()
    proc.execute_instruction(INSTR_CNOT, [0, 1])
    ns.sim_run()

    q0, q1 = proc.peek([0, 1])
    assert qubitapi.fidelity([q0, q1], ket, squared=True) == pytest.approx(fidelity)


# ---------------------------------------------------------------------------
# Backwards compatibility: per-qubit model still works, but warns on
# multi-qubit gates
# ---------------------------------------------------------------------------


def build_topology_cfg(
    multi_gate: GateConfig = None, all_qubit_gate: GateConfig = None
) -> TopologyConfig:
    """Two perfect qubits with INSTR_INIT, a noisy INSTR_X, and the given multi/all-qubit gates."""
    return TopologyConfig(
        qubits=[
            QubitIdConfig(qubit_id=i, qubit_config=QubitConfig.perfect_config(True))
            for i in range(2)
        ],
        single_gates=[
            SingleGateConfig(
                qubit_id=i,
                gate_configs=[
                    GateConfig.perfect_config("INSTR_INIT", 1),
                    GateConfig.with_depolar_prob("INSTR_X", 1, 0.2),
                ],
            )
            for i in range(2)
        ],
        multi_gates=(
            []
            if multi_gate is None
            else [MultiGateConfig(qubit_ids=[0, 1], gate_configs=[multi_gate])]
        ),
        all_qubit_gates=(
            None
            if all_qubit_gate is None
            else AllQubitGateConfig(gate_configs=[all_qubit_gate])
        ),
    )


def build(cfg: TopologyConfig):
    return build_qprocessor_from_topology("proc", LhiTopologyBuilder.from_config(cfg))


OLD_STYLE_YAML = """
qubits:
  - qubit_id: 0
    qubit_config:
      is_communication: true
      noise_config_cls: "QubitT1T2Config"
      noise_config: {T1: 0, T2: 0}
  - qubit_id: 1
    qubit_config:
      is_communication: true
      noise_config_cls: "QubitT1T2Config"
      noise_config: {T1: 0, T2: 0}
single_gates: []
multi_gates:
  - qubit_ids: [0, 1]
    gate_configs:
      - name: "INSTR_CNOT"
        noise_config_cls: "GateDepolariseConfig"
        noise_config: {duration: 1000, depolarise_prob: 0.2}
all_qubit_gates:
  gate_configs:
    - name: "INSTR_ROT_X_ALL"
      noise_config_cls: "GateDepolariseConfig"
      noise_config: {duration: 1000, depolarise_prob: 0.2}
"""


def test_old_yaml_still_loads_and_warns(tmp_path):
    """A YAML file written before GateJointDepolariseConfig existed still loads,
    keeps the per-qubit NetSquid model (no silent change in results) and warns
    for each multi-qubit gate that uses it."""
    path = tmp_path / "old_topology.yaml"
    path.write_text(OLD_STYLE_YAML)

    topology = LhiTopologyBuilder.from_config(TopologyConfig.from_file(str(path)))
    for gate_infos in topology.multi_gate_infos.values():
        for info in gate_infos:
            assert info.error_model is DepolarNoiseModel
    for info in topology.all_qubit_gate_infos:
        assert info.error_model is DepolarNoiseModel

    with pytest.warns(PerQubitDepolarNoiseWarning) as record:
        build_qprocessor_from_topology("proc", topology)
    messages = [str(w.message) for w in record]
    # NetSquid instruction names: INSTR_CNOT.name, INSTR_ROT_X_ALL.name
    assert any("'cnot_gate'" in m for m in messages)
    assert any("'x_rot_all_gate'" in m for m in messages)


def test_per_qubit_multi_gate_warns():
    cfg = build_topology_cfg(
        multi_gate=GateConfig.with_depolar_prob("INSTR_CNOT", 1, 0.2)
    )
    with pytest.warns(PerQubitDepolarNoiseWarning, match="GateJointDepolariseConfig"):
        build(cfg)


def test_per_qubit_all_qubit_gate_warns():
    cfg = build_topology_cfg(
        all_qubit_gate=GateConfig.with_depolar_prob("INSTR_ROT_X_ALL", 1, 0.2)
    )
    with pytest.warns(PerQubitDepolarNoiseWarning):
        build(cfg)


@pytest.mark.parametrize(
    "multi_gate",
    [
        # Joint model: the recommended config.
        GateConfig.with_joint_depolar_prob("INSTR_CNOT", 1, 0.2),
        GateConfig.with_two_gate_fidelity("INSTR_CNOT", 1, 0.99),
        # Per-qubit model without noise is fine (both channels are identity).
        GateConfig.perfect_config("INSTR_CNOT", 1),
        GateConfig.with_depolar_prob("INSTR_CNOT", 1, 0.0),
        # No multi-qubit gate at all; noisy single-qubit gates never warn.
        None,
    ],
    ids=["joint", "two_gate_fidelity", "perfect", "zero_prob", "single_only"],
)
def test_no_warning(multi_gate):
    cfg = build_topology_cfg(multi_gate=multi_gate)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerQubitDepolarNoiseWarning)
        build(cfg)


def test_per_qubit_model_behaviour_unchanged():
    """GateDepolariseConfig on a CNOT still gives NetSquid's per-qubit result,
    so results produced before the joint model was added are reproducible."""
    prob = 0.2
    cfg = build_topology_cfg(
        multi_gate=GateConfig.with_depolar_prob("INSTR_CNOT", 1, prob)
    )
    with pytest.warns(PerQubitDepolarNoiseWarning):
        proc = build(cfg)

    proc.execute_instruction(INSTR_INIT, [0])
    ns.sim_run()
    proc.execute_instruction(INSTR_INIT, [1])
    ns.sim_run()
    proc.execute_instruction(INSTR_CNOT, [0, 1])
    ns.sim_run()

    q0, q1 = proc.peek([0, 1])
    assert qubitapi.fidelity([q0, q1], ketstates.s00, squared=True) == pytest.approx(
        (1 - prob / 2) ** 2
    )


def test_warning_module_does_not_import_netsquid():
    """pytest-xdist imports a warning's category module from a non-main thread,
    where importing NetSquid crashes (pydynaa installs a signal handler). The
    module defining PerQubitDepolarNoiseWarning must therefore stay NetSquid-free.
    Run in a subprocess, as NetSquid is already imported in this one."""
    code = (
        "import sys, qoala.runtime.noise_warnings; "
        "assert 'netsquid' not in sys.modules, 'netsquid was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert PerQubitDepolarNoiseWarning.__module__ == "qoala.runtime.noise_warnings"
