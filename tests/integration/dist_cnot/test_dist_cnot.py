from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import run_two_node_app


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, determ: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(host_instr_time=500, qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class Result:
    alice_results: BatchResult
    bob_results: BatchResult


def run_dist_cnot(num_iterations: int) -> Result:
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("dist_cnot.log")
    LogManager.log_tasks_to_file("dist_cnot_tasks.log")
    ns.sim_reset()

    num_qubits = 4
    alice_id = 0
    bob_id = 1

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, determ=True)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e9)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]
    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=1_500, first_bin=0, bin_pattern=pattern, repeat_period=20_000
    )

    alice_program = load_program("dist_cnot_alice.iqoala")
    bob_program = load_program("dist_cnot_bob.iqoala")

    # state = 5 -> control is in |1> state
    alice_input = ProgramInput({"bob_id": bob_id, "state": 5})
    # state = 4 -> target is in |0> state
    bob_input = ProgramInput({"alice_id": alice_id, "state": 4})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return Result(alice_result, bob_result)


def test_dist_cnot():
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("dist_cnot.log")
    LogManager.log_tasks_to_file("dist_cnot_tasks.log")
    num_iterations = 2

    result = run_dist_cnot(num_iterations=num_iterations)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


if __name__ == "__main__":
    test_dist_cnot()
