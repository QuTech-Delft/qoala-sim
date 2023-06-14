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
from qoala.runtime.task import TaskExecutionMode
from qoala.util.runner import run_application


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, tem: TaskExecutionMode
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        tem=tem.name,
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
    tem: TaskExecutionMode = TaskExecutionMode.BLOCK,
):
    num_qubits = 3
    alice_id = 0
    bob_id = 1

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, tem)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, tem)

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

    app_result = run_application(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return QkdResult(alice_result, bob_result)


def qkd_1pair_md(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_MD_alice.iqoala"
    bob_file = "qkd_1pair_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem=tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_1pair_ck(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_CK_alice.iqoala"
    bob_file = "qkd_1pair_CK_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_1pair_ck_cb(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_1pair_CK_cb_alice.iqoala"
    bob_file = "qkd_1pair_CK_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def qkd_2pairs_md(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_MD_alice.iqoala"
    bob_file = "qkd_2pairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem=tem)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_1qubit(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_1qubit_alice.iqoala"
    bob_file = "qkd_2pairs_CK_1qubit_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_1qubit_cb(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_1qubit_cb_alice.iqoala"
    bob_file = "qkd_2pairs_CK_1qubit_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_2qubits_app_move(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_2qubits_app_move_alice.iqoala"
    bob_file = "qkd_2pairs_CK_2qubits_app_move_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_2pairs_ck_2qubits_wait_all(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_2pairs_CK_2qubits_wait_all_alice.iqoala"
    bob_file = "qkd_2pairs_CK_2qubits_wait_all_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


def qkd_100pairs_md(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_100pairs_MD_alice.iqoala"
    bob_file = "qkd_100pairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, tem)

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


def qkd_npairs_md(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 1
    alice_file = "qkd_npairs_MD_alice.iqoala"
    bob_file = "qkd_npairs_MD_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, num_pairs=100, tem=tem)

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


def qkd_npairs_ck_1qubit_cb(tem: TaskExecutionMode):
    ns.sim_reset()

    num_iterations = 1
    alice_file = "qkd_npairs_CK_cb_alice.iqoala"
    bob_file = "qkd_npairs_CK_cb_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file, num_pairs=100, tem=tem)
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


def test_qkd_1pair_md_qoala_tasks():
    qkd_1pair_md(tem=TaskExecutionMode.QOALA)


def test_qkd_1pair_md_block_tasks():
    qkd_1pair_md(tem=TaskExecutionMode.BLOCK)


def test_qkd_1pair_ck_qoala_tasks():
    qkd_1pair_ck(tem=TaskExecutionMode.QOALA)


def test_qkd_1pair_ck_block_tasks():
    qkd_1pair_ck(tem=TaskExecutionMode.BLOCK)


def test_qkd_1pair_ck_cb_qoala_tasks():
    qkd_1pair_ck_cb(tem=TaskExecutionMode.QOALA)


def test_qkd_1pair_ck_cb_block_tasks():
    qkd_1pair_ck_cb(tem=TaskExecutionMode.BLOCK)


def test_qkd_2pairs_md_qoala_tasks():
    qkd_2pairs_md(tem=TaskExecutionMode.QOALA)


def test_qkd_2pairs_md_block_tasks():
    qkd_2pairs_md(tem=TaskExecutionMode.BLOCK)


def test_qkd_2pairs_ck_1qubit_qoala_tasks():
    qkd_2pairs_ck_1qubit(tem=TaskExecutionMode.QOALA)


def test_qkd_2pairs_ck_1qubit_block_tasks():
    qkd_2pairs_ck_1qubit(tem=TaskExecutionMode.BLOCK)


def test_qkd_2pairs_ck_1qubit_cb_qoala_tasks():
    qkd_2pairs_ck_1qubit_cb(tem=TaskExecutionMode.QOALA)


def test_qkd_2pairs_ck_1qubit_cb_block_tasks():
    qkd_2pairs_ck_1qubit_cb(tem=TaskExecutionMode.BLOCK)


def test_qkd_2pairs_ck_2qubits_app_move_qoala_tasks():
    qkd_2pairs_ck_2qubits_app_move(tem=TaskExecutionMode.QOALA)


def test_qkd_2pairs_ck_2qubits_app_move_block_tasks():
    qkd_2pairs_ck_2qubits_app_move(tem=TaskExecutionMode.BLOCK)


def test_qkd_2pairs_ck_2qubits_wait_all_qoala_tasks():
    qkd_2pairs_ck_2qubits_wait_all(tem=TaskExecutionMode.QOALA)


def test_qkd_2pairs_ck_2qubits_wait_all_block_tasks():
    qkd_2pairs_ck_2qubits_wait_all(tem=TaskExecutionMode.BLOCK)


def test_qkd_100pairs_md_qoala_tasks():
    qkd_100pairs_md(tem=TaskExecutionMode.QOALA)


def test_qkd_100pairs_md_block_tasks():
    qkd_100pairs_md(tem=TaskExecutionMode.BLOCK)


def test_qkd_npairs_md_qoala_tasks():
    qkd_npairs_md(tem=TaskExecutionMode.QOALA)


def test_qkd_npairs_md_block_tasks():
    qkd_npairs_md(tem=TaskExecutionMode.BLOCK)


def test_qkd_npairs_ck_1qubit_cb_qoala_tasks():
    qkd_npairs_ck_1qubit_cb(tem=TaskExecutionMode.QOALA)


def test_qkd_npairs_ck_1qubit_cb_block_tasks():
    qkd_npairs_ck_1qubit_cb(tem=TaskExecutionMode.BLOCK)


if __name__ == "__main__":
    test_qkd_1pair_md_qoala_tasks()
    # test_qkd_1pair_md_block_tasks()
    # test_qkd_1pair_ck_qoala_tasks()
    # test_qkd_1pair_ck_block_tasks()
    # test_qkd_1pair_ck_cb_qoala_tasks()
    # test_qkd_1pair_ck_cb_block_tasks()
    # test_qkd_2pairs_md_qoala_tasks()
    # test_qkd_2pairs_md_block_tasks()
    # test_qkd_2pairs_ck_1qubit_qoala_tasks()
    # test_qkd_2pairs_ck_1qubit_block_tasks()
    # test_qkd_2pairs_ck_1qubit_cb_qoala_tasks()
    # test_qkd_2pairs_ck_1qubit_cb_block_tasks()
    # test_qkd_2pairs_ck_2qubits_app_move_qoala_tasks()
    # test_qkd_2pairs_ck_2qubits_app_move_block_tasks()
    # test_qkd_2pairs_ck_2qubits_wait_all_qoala_tasks()
    # test_qkd_2pairs_ck_2qubits_wait_all_block_tasks()
    # test_qkd_100pairs_md_qoala_tasks()
    # test_qkd_100pairs_md_block_tasks()
    # test_qkd_npairs_md_qoala_tasks()
    # test_qkd_npairs_md_block_tasks()
    # test_qkd_npairs_ck_1qubit_cb_qoala_tasks()
    # test_qkd_npairs_ck_1qubit_cb_block_tasks()
