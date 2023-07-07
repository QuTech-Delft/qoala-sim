from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfigBuilder,
)
from qoala.runtime.program import ProgramInput
from qoala.util.runner import run_single_node_app


def get_config(t1: int = 0, t2: int = 0, gate_fidelity: float = 1.0) -> ProcNodeConfig:
    topology = (
        TopologyConfigBuilder()
        .uniform_topology()
        .num_qubits(1)
        .qubit_t1(t1)
        .qubit_t2(t2)
        .default_generic_gates()
        .zero_gate_durations()
        .perfect_gate_fidelities()
        .all_comm_gates_fidelity(gate_fidelity)
        .build()
    )
    return ProcNodeConfig(
        node_name="alice",
        node_id=0,
        topology=topology,
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    return program


@dataclass
class BusyProgramTomography:
    prob_x_0: float
    prob_y_0: float
    prob_z_0: float


def run_busy_program(num_iterations: int, node_cfg: ProcNodeConfig):
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("busy_program.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
    )

    all_results = app_results.batch_results["alice"].results
    x_outcomes = [result.values["m_x"] for result in all_results]
    y_outcomes = [result.values["m_y"] for result in all_results]
    z_outcomes = [result.values["m_z"] for result in all_results]

    prob_x_0 = sum(1 - x for x in x_outcomes) / num_iterations
    prob_y_0 = sum(1 - y for y in y_outcomes) / num_iterations
    prob_z_0 = sum(1 - z for z in z_outcomes) / num_iterations
    return BusyProgramTomography(prob_x_0, prob_y_0, prob_z_0)


def test_busy():
    ns.sim_reset()
    # LogManager.set_log_level("DEBUG")

    node_cfg = get_config()
    result = run_busy_program(200, node_cfg)

    print(f"prob_x_0: {result.prob_x_0}")
    print(f"prob_y_0: {result.prob_y_0}")
    print(f"prob_z_0: {result.prob_z_0}")

    assert 0.8 < result.prob_x_0
    assert 0.3 < result.prob_y_0 < 0.7
    assert 0.3 < result.prob_z_0 < 0.7


def test_busy_bad_coherence():
    ns.sim_reset()
    # LogManager.set_log_level("DEBUG")

    node_cfg = get_config(t1=10, t2=10)
    result = run_busy_program(200, node_cfg)

    print(f"prob_x_0: {result.prob_x_0}")
    print(f"prob_y_0: {result.prob_y_0}")
    print(f"prob_z_0: {result.prob_z_0}")
    # Because of decoherence, we expect the prepared state to be close to |0>
    # (i.e. Z-outcome should be mostly 0, X and Y uniformly random)
    assert 0.3 < result.prob_x_0 < 0.7
    assert 0.3 < result.prob_y_0 < 0.7
    assert 0.8 < result.prob_z_0


def test_busy_bad_gates():
    ns.sim_reset()
    # LogManager.set_log_level("DEBUG")

    node_cfg = get_config(gate_fidelity=0.5)
    result = run_busy_program(200, node_cfg)

    print(f"prob_x_0: {result.prob_x_0}")
    print(f"prob_y_0: {result.prob_y_0}")
    print(f"prob_z_0: {result.prob_z_0}")
    # Because of gate noise, the prepared state should be close to maximally mixed
    # (i.e. all outcomes should be uniformly random)
    assert 0.3 < result.prob_x_0 < 0.7
    assert 0.3 < result.prob_y_0 < 0.7
    assert 0.3 < result.prob_z_0 < 0.7


if __name__ == "__main__":
    test_busy()
    test_busy_bad_coherence()
    test_busy_bad_gates()
