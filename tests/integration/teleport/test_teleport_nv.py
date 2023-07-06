from __future__ import annotations

import os
from dataclasses import dataclass

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
from qoala.util.runner import run_application


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.from_nv_params(
            num_qubits=num_qubits, params=NvParams()
        ),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
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


def run_teleport(num_iterations: int) -> TeleportResult:
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
    alice_input = ProgramInput({"bob_id": bob_id})

    bob_program = load_program("teleport_nv_bob.iqoala")
    bob_input = ProgramInput({"alice_id": alice_id})

    app_result = run_application(
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
    num_iterations = 10

    result = run_teleport(num_iterations=num_iterations)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


if __name__ == "__main__":
    test_teleport()
