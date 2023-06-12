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
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.runtime.task import TaskExecutionMode
from qoala.util.runner import run_application


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, tem: TaskExecutionMode
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=100_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        tem=tem.name,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class PingPongResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_pingpong(
    num_iterations: int, tem: TaskExecutionMode = TaskExecutionMode.BLOCK
) -> PingPongResult:
    ns.sim_reset()

    num_qubits = 3
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, tem)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, tem)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=500_000
    )

    alice_program = load_program("pingpong_alice.iqoala")
    alice_input = ProgramInput({"bob_id": bob_id})

    bob_program = load_program("pingpong_bob.iqoala")
    bob_input = ProgramInput({"alice_id": alice_id})

    app_result = run_application(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return PingPongResult(alice_result, bob_result)


def check_pingpong(num_iterations: int, tem: TaskExecutionMode):
    # LogManager.set_log_level("INFO")

    ns.sim_reset()
    result = run_pingpong(num_iterations=num_iterations, tem=tem)

    program_results = result.alice_result.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


def test_pingpong_block_tasks():
    check_pingpong(10, TaskExecutionMode.BLOCK)


def test_pingpong_qoala_tasks():
    check_pingpong(10, TaskExecutionMode.QOALA)


if __name__ == "__main__":
    test_pingpong_block_tasks()
    test_pingpong_qoala_tasks()
