from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns
from netqasm.lang.instr.flavour import TrappedIonFlavour

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
from qoala.util.runner import run_two_node_app


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_tri_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=100_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("TrappedIonNtf"),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=TrappedIonFlavour()).parse()


@dataclass
class PingPongResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_pingpong(num_iterations: int) -> PingPongResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    # NOTE: make sure they both have only 2 qubits!!
    # More than 2 qubits makes the bichromatic gate behave differently,
    # and hence the decomposition of a CNOT does not work anymore
    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits=2)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits=2)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=500_000
    )

    alice_program = load_program("pingpong_tri_alice.iqoala")
    alice_input = ProgramInput({"bob_id": bob_id})

    bob_program = load_program("pingpong_tri_bob.iqoala")
    bob_input = ProgramInput({"alice_id": alice_id})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
        linear=True,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return PingPongResult(alice_result, bob_result)


def check_pingpong(num_iterations: int):
    # LogManager.set_log_level("INFO")

    ns.sim_reset()
    result = run_pingpong(num_iterations=num_iterations)

    program_results = result.alice_result.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


if __name__ == "__main__":
    check_pingpong(num_iterations=100)
