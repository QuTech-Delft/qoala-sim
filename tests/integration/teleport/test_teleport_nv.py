from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import run_two_node_app, run_two_node_app_separate_inputs


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.from_nv_params(num_qubits=5, params=NvParams()),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
        determ_sched=True,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=NVFlavour()).parse()


@dataclass
class TeleportResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_teleport(num_iterations: int, different_inputs: bool = False) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 3
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )

    alice_program = load_program("teleport_nv_alice.iqoala")
    bob_program = load_program("teleport_nv_bob.iqoala")

    if different_inputs:
        alice_inputs: List[ProgramInput] = []
        bob_inputs: List[ProgramInput] = []
        for _ in range(num_iterations):
            basis = random.randint(0, 5)
            alice_inputs.append(ProgramInput({"bob_id": bob_id, "state": basis}))
            bob_inputs.append(ProgramInput({"alice_id": alice_id, "state": basis}))

        app_result = run_two_node_app_separate_inputs(
            num_iterations=num_iterations,
            programs={"alice": alice_program, "bob": bob_program},
            program_inputs={"alice": alice_inputs, "bob": bob_inputs},
            network_cfg=network_cfg,
            linear=True,
        )
    else:
        # state = 5 -> teleport |1> state
        alice_input = ProgramInput({"bob_id": bob_id, "state": 5})
        # state = 4 -> measure in +Z, i.e. expect a "1" outcome
        bob_input = ProgramInput({"alice_id": alice_id, "state": 4})

        app_result = run_two_node_app(
            num_iterations=num_iterations,
            programs={"alice": alice_program, "bob": bob_program},
            program_inputs={"alice": alice_input, "bob": bob_input},
            network_cfg=network_cfg,
        )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return TeleportResult(alice_result, bob_result)


def test_teleport():
    # LogManager.set_log_level("DEBUG")
    # LogManager.enable_task_logger(True)
    num_iterations = 10

    result = run_teleport(num_iterations=num_iterations)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


def test_teleport_different_inputs():
    # LogManager.set_log_level("DEBUG")
    LogManager.enable_task_logger(True)
    num_iterations = 10

    result = run_teleport(num_iterations=num_iterations, different_inputs=True)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 0 for outcome in outcomes)


if __name__ == "__main__":
    test_teleport()
    test_teleport_different_inputs()
