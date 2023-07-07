from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.runner import run_two_node_app_separate_inputs


def create_procnode_cfg(name: str, id: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(1),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class ClassicalAppResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_classical_app(num_iterations: int) -> ClassicalAppResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id)
    bob_node_cfg = create_procnode_cfg("bob", bob_id)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 2500)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("classical_alice.iqoala")
    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "value": i}) for i in range(num_iterations)
    ]

    bob_program = load_program("classical_bob.iqoala")
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_inputs, "bob": bob_inputs},
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return ClassicalAppResult(alice_result, bob_result)


def test_classical_app():
    # LogManager.set_log_level("INFO")
    # LogManager.enable_task_logger(True)
    # LogManager.log_tasks_to_file("test_classical.log")
    num_iterations = 20

    result = run_classical_app(num_iterations=num_iterations)

    program_results = result.alice_results.results
    outcomes = [result.values["returned_value"] for result in program_results]
    print(outcomes)
    # assert all(outcome == 1 for outcome in outcomes)


if __name__ == "__main__":
    test_classical_app()
