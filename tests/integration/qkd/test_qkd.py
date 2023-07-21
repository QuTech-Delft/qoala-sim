from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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
from qoala.util.logging import LogManager
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
    num_pairs: Optional[int] = None,
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
        linear=linear,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return QkdResult(alice_result, bob_result)


def qkd_1pair_md():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_MD_alice.iqoala"
    bob_file = "qkd_1pair_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_1pair_ck():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_CK_alice.iqoala"
    bob_file = "qkd_1pair_CK_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_1pair_ck_cb():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_CK_cb_alice.iqoala"
    bob_file = "qkd_1pair_CK_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_2pairs_md():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_MD_alice.iqoala"
    bob_file = "qkd_2pairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_1qubit():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_1qubit_alice.iqoala"
    bob_file = "qkd_2pairs_CK_1qubit_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_1qubit_cb():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_1qubit_cb_alice.iqoala"
    bob_file = "qkd_2pairs_CK_1qubit_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, linear=False)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_2qubits_app_move():
    ns.sim_reset()
    LogManager.enable_task_logger(True)

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_2qubits_app_move_alice.iqoala"
    bob_file = "qkd_2pairs_CK_2qubits_app_move_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_2qubits_wait_all():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_2qubits_wait_all_alice.iqoala"
    bob_file = "qkd_2pairs_CK_2qubits_wait_all_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_100pairs_md():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_100pairs_MD_alice.iqoala"
    bob_file = "qkd_100pairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        print(f"alice: {alice['outcomes']}")
        print(f"bob: {bob['outcomes']}")
        assert alice["outcomes"] == bob["outcomes"]


def qkd_npairs_md():
    ns.sim_reset()

    num_iterations = 1
    alice_file = "qkd_npairs_MD_alice.iqoala"
    bob_file = "qkd_npairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, num_pairs=100)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        print(f"alice: {alice['outcomes']}")
        print(f"bob  : {bob['outcomes']}")
        assert alice["outcomes"] == bob["outcomes"]


def qkd_npairs_ck_1qubit_cb():
    ns.sim_reset()

    num_iterations = 1
    alice_file = "qkd_npairs_CK_cb_alice.iqoala"
    bob_file = "qkd_npairs_CK_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, num_pairs=100)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        print(f"alice: {alice['outcomes']}")
        print(f"bob  : {bob['outcomes']}")
        assert alice["outcomes"] == bob["outcomes"]


def test_qkd_1pair_md():
    qkd_1pair_md()


def test_qkd_1pair_ck():
    qkd_1pair_ck()


def test_qkd_1pair_ck_cb():
    qkd_1pair_ck_cb()


def test_qkd_2pairs_md():
    qkd_2pairs_md()


def test_qkd_2pairs_ck_1qubit():
    qkd_2pairs_ck_1qubit()


def test_qkd_2pairs_ck_1qubit_cb():
    qkd_2pairs_ck_1qubit_cb()


def test_qkd_2pairs_ck_2qubits_app_move():
    qkd_2pairs_ck_2qubits_app_move()


def test_qkd_2pairs_ck_2qubits_wait_all():
    qkd_2pairs_ck_2qubits_wait_all()


def test_qkd_100pairs_md():
    qkd_100pairs_md()


def test_qkd_npairs_md():
    qkd_npairs_md()


def test_qkd_npairs_ck_1qubit_cb():
    qkd_npairs_ck_1qubit_cb()


if __name__ == "__main__":
    test_qkd_1pair_md()
    test_qkd_1pair_ck()
    test_qkd_1pair_ck_cb()
    test_qkd_2pairs_md()
    test_qkd_2pairs_ck_1qubit()
    test_qkd_2pairs_ck_1qubit_cb()
    test_qkd_2pairs_ck_2qubits_app_move()
    test_qkd_2pairs_ck_2qubits_wait_all()
    test_qkd_100pairs_md()
    test_qkd_npairs_md()
    test_qkd_npairs_ck_1qubit_cb()
