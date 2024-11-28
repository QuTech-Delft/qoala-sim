from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import run_n_node_app


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
    charlie_results: BatchResult


def run_three_nodes(num_iterations: int) -> Result:
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("three_nodes.log")
    LogManager.log_tasks_to_file("three_nodes_tasks.log")
    ns.sim_reset()

    num_qubits = num_iterations * 2
    alice_id = 0
    bob_id = 1
    charlie_id = 2

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, determ=True)
    charlie_node_cfg = create_procnode_cfg(
        "charlie", charlie_id, num_qubits, determ=True
    )

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg, charlie_node_cfg], link_duration=500
    )
    bin_pattern = []
    for i in range(num_iterations):
        bin_pattern.append((alice_id, i, bob_id, i))
        bin_pattern.append((alice_id, i, charlie_id, i))
        bin_pattern.append((bob_id, i, charlie_id, i))
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=1000,
        first_bin=0,
        bin_pattern=bin_pattern,
        repeat_period=len(bin_pattern) * 1000,
    )

    alice_program = load_program("three_nodes_alice.iqoala")
    bob_program = load_program("three_nodes_bob.iqoala")
    charlie_program = load_program("three_nodes_charlie.iqoala")

    alice_input = ProgramInput({"bob_id": bob_id, "charlie_id": charlie_id})
    bob_input = ProgramInput({"alice_id": alice_id, "charlie_id": charlie_id})
    charlie_input = ProgramInput({"alice_id": alice_id, "bob_id": bob_id})

    app_result = run_n_node_app(
        num_iterations=num_iterations,
        programs={
            "alice": alice_program,
            "bob": bob_program,
            "charlie": charlie_program,
        },
        program_inputs={
            "alice": alice_input,
            "bob": bob_input,
            "charlie": charlie_input,
        },
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]
    charlie_result = app_result.batch_results["charlie"]

    return Result(alice_result, bob_result, charlie_result)


def test_three_nodes():
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("three_nodes.log")
    LogManager.log_tasks_to_file("three_nodes_tasks.log")
    num_iterations = 20

    result = run_three_nodes(num_iterations=num_iterations)
    for i in range(num_iterations):
        assert (
            result.alice_results.results[i].values["m_ab"]
            == result.bob_results.results[i].values["m_ba"]
        )
        assert (
            result.alice_results.results[i].values["m_ac"]
            == result.charlie_results.results[i].values["m_ca"]
        )
        assert (
            result.bob_results.results[i].values["m_bc"]
            == result.charlie_results.results[i].values["m_cb"]
        )


if __name__ == "__main__":
    test_three_nodes()
