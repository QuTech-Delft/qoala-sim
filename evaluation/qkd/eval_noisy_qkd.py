from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
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
    prob_max_mixed: float,
    attempt_success_prob: float,
    attempt_duration: float,
    state_delay: float,
    num_pairs: Optional[int] = None,
):
    num_qubits = 3
    alice_id = 0
    bob_id = 1

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    link_cfg = LinkConfig.depolarise_config(
        prob_max_mixed, attempt_success_prob, attempt_duration, state_delay
    )
    link_between_cfg = LinkBetweenNodesConfig(
        node_id1=alice_id, node_id2=bob_id, link_config=link_cfg
    )
    network_cfg = ProcNodeNetworkConfig(
        nodes=[alice_node_cfg, bob_node_cfg], links=[link_between_cfg]
    )

    alice_program = load_program(alice_file)
    bob_program = load_program(bob_file)

    if num_pairs is not None:
        alice_input = ProgramInput({"bob_id": bob_id, "N": num_pairs})
        bob_input = ProgramInput({"alice_id": alice_id, "N": num_pairs})
    else:
        alice_input = ProgramInput({"bob_id": bob_id})
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

    return QkdResult(alice_result, bob_result)


def test_qkd_md_1pair():
    ns.sim_reset()

    # LogManager.set_log_level("INFO")

    num_iterations = 1000

    alice_file = "qkd_1pair_MD_alice.iqoala"
    bob_file = "qkd_1pair_MD_bob.iqoala"

    qkd_result = run_qkd(
        num_iterations,
        alice_file,
        bob_file,
        prob_max_mixed=0.4,
        attempt_success_prob=0.001,
        attempt_duration=1000,
        state_delay=0.0,
    )
    assert len(qkd_result.alice_result.results) == num_iterations
    assert len(qkd_result.bob_result.results) == num_iterations

    count_equal_outcomes = 0
    durations = []
    for i in range(num_iterations):
        alice_result = qkd_result.alice_result.results[i].values
        bob_result = qkd_result.bob_result.results[i].values
        if alice_result["m0"] == bob_result["m0"]:
            count_equal_outcomes += 1

        alice_start, alice_end = qkd_result.alice_result.timestamps[i]
        bob_start, bob_end = qkd_result.alice_result.timestamps[i]
        duration = max(alice_end, bob_end) - min(alice_start, bob_start)
        durations.append(duration)

    avg_duration = sum(durations) / len(durations)

    # We used a link that produced the state 0.4 * <maximally mixed> + 0.6 * Phi+.
    # Hence we expect the ratio of pairs with equal outcomes to be
    # 0.5 * 0.4                     +    1.0 * 0.6                   = 0.8
    # (mixed state -> 50% success)       (Phi+ -> 100% success)
    assert (count_equal_outcomes / num_iterations) <= 0.85
    assert (count_equal_outcomes / num_iterations) >= 0.75

    # On average, 1000 attempts were needed per EPR pair, which took 1000 ns each.
    # The number of attempts hence follows a geometric distribution.
    # The number of attempts, averaged over all 1000 iterations, should be within
    # the range (918, 1081) with probability 0.99.
    print(avg_duration)
    assert avg_duration > 918 * 1000
    assert avg_duration < 1081 * 1000


def test_qkd_md_npairs():
    ns.sim_reset()

    # LogManager.set_log_level("INFO")

    num_iterations = 1

    alice_file = "qkd_npairs_MD_alice.iqoala"
    bob_file = "qkd_npairs_MD_bob.iqoala"

    qkd_result = run_qkd(
        num_iterations,
        alice_file,
        bob_file,
        prob_max_mixed=0.4,
        attempt_success_prob=0.001,
        attempt_duration=1000,
        state_delay=0.0,
        num_pairs=1000,
    )
    assert len(qkd_result.alice_result.results) == num_iterations
    assert len(qkd_result.bob_result.results) == num_iterations

    count_equal_outcomes = 0
    durations = []
    for i in range(num_iterations):
        alice_outcomes = qkd_result.alice_result.results[i].values["outcomes"]
        bob_outcomes = qkd_result.bob_result.results[i].values["outcomes"]
        for alice, bob in zip(alice_outcomes, bob_outcomes):
            if alice == bob:
                count_equal_outcomes += 1

        alice_start, alice_end = qkd_result.alice_result.timestamps[i]
        bob_start, bob_end = qkd_result.alice_result.timestamps[i]
        duration = max(alice_end, bob_end) - min(alice_start, bob_start)
        durations.append(duration)

    avg_duration = sum(durations) / len(durations)

    # We used a link that produced the state 0.4 * <maximally mixed> + 0.6 * Phi+.
    # Hence we expect the ratio of pairs with equal outcomes to be
    # 0.5 * 0.4                     +    1.0 * 0.6                   = 0.8
    # (mixed state -> 50% success)       (Phi+ -> 100% success)
    assert (count_equal_outcomes / (num_iterations * 1000)) <= 0.85
    assert (count_equal_outcomes / (num_iterations * 1000)) >= 0.75

    # On average, 1000 attempts were needed per EPR pair, which took 1000 ns each.
    # The number of attempts hence follows a geometric distribution.
    # The number of attempts, averaged over all 1000 pairs per program,
    # should be within the range (918, 1081) with probability 0.99.
    print(avg_duration)
    assert avg_duration > 918 * 1000 * 1000
    assert avg_duration < 1081 * 1000 * 1000


if __name__ == "__main__":
    test_qkd_md_1pair()
    test_qkd_md_npairs()
