from __future__ import annotations

import os
import time
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
from qoala.util.runner import run_two_node_app


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class QkdResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_qkd(
    num_iterations: int,
    alice_file: str,
    bob_file: str,
    num_pairs: int,
    linear: bool = True,
):
    num_qubits = 3
    alice_id = 0
    bob_id = 1

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )

    alice_program = load_program(alice_file)
    bob_program = load_program(bob_file)

    alice_input = ProgramInput({"bob_id": bob_id, "N": num_pairs})
    bob_input = ProgramInput({"alice_id": alice_id, "N": num_pairs})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
        linear=linear,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return QkdResult(alice_result, bob_result)


def qkd_npairs_md(num_iterations: int, num_pairs: int):
    num_pairs = int(num_pairs)
    ns.sim_reset()

    alice_file = "qkd_npairs_MD_alice.iqoala"
    bob_file = "qkd_npairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, num_pairs=num_pairs)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        # print(f"alice: {alice['outcomes']}")
        # print(f"bob  : {bob['outcomes']}")
        assert alice["outcomes"] == bob["outcomes"]


if __name__ == "__main__":
    start = time.time()

    qkd_npairs_md(num_iterations=100, num_pairs=100)

    end = time.time()
    print(f"duration: {round(end - start, 2)} s")
