from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
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
from qoala.runtime.program import BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager
from qoala.util.runner import (
    AppResult,
    create_batch,
    run_two_node_app,
    run_two_node_app_separate_inputs,
)


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, determ: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class AnonymousTransferResult:
    alice_results: BatchResult
    bob_results: BatchResult
    charlie_results: BatchResult


def run_anonymous_transfer(num_iterations: int) -> AnonymousTransferResult:
    ns.sim_reset()

    num_qubits = 4
    alice_id = 0
    bob_id = 1
    charlie_id = 2

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, determ=True)
    charlie_node_cfg = create_procnode_cfg(
        "charlie", charlie_id, num_qubits, determ=True
    )

    cconn_ab = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e9)
    cconn_ac = ClassicalConnectionConfig.from_nodes(alice_id, charlie_id, 1e9)
    cconn_bc = ClassicalConnectionConfig.from_nodes(bob_id, charlie_id, 1e9)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg, charlie_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn_ab, cconn_ac, cconn_bc]

    alice_program = load_program("anonymous_transfer_alice.iqoala")
    bob_program = load_program("anonymous_transfer_bob.iqoala")
    charlie_program = load_program("anonymous_transfer_charlie.iqoala")

    alice_input = [
        ProgramInput({"bob_id": bob_id, "charlie_id": charlie_id, "b": 0})
        for _ in range(num_iterations)
    ]
    bob_input = [
        ProgramInput({"alice_id": alice_id, "charlie_id": charlie_id})
        for _ in range(num_iterations)
    ]
    charlie_input = [
        ProgramInput({"alice_id": alice_id, "bob_id": bob_id, "b": 0})
        for _ in range(num_iterations)
    ]

    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)
    names = ["alice", "bob", "charlie"]
    programs = {"alice": alice_program, "bob": bob_program, "charlie": charlie_program}
    program_inputs = {"alice": alice_input, "bob": bob_input, "charlie": charlie_input}
    batches: Dict[str, ProgramBatch] = {}  # node -> batch

    for name in names:
        procnode = network.nodes[name]
        program = programs[name]
        inputs = program_inputs[name]

        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch_info = create_batch(program, unit_module, inputs, num_iterations)
        batches[name] = procnode.submit_batch(batch_info)

    for batch in batches.values():
        assert batch.batch_id == 0
        assert [p.pid for p in batch.instances] == [i for i in range(num_iterations)]

    for name in names:
        procnode = network.nodes[name]
        remote_pids = {0: [i for i in range(num_iterations)]}
        procnode.initialize_processes(remote_pids=remote_pids, linear=True)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in names:
        procnode = network.nodes[name]
        # only one batch (ID = 0), so get value at index 0
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    total_duration = ns.sim_time()
    app_result = AppResult(results, statistics, total_duration)

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]
    charlie_result = app_result.batch_results["charlie"]

    return AnonymousTransferResult(alice_result, bob_result, charlie_result)


def anonymous_transfer(num_iterations: int):
    result = run_anonymous_transfer(num_iterations=num_iterations)

    alice_results = result.alice_results.results
    alice_outcomes = [result.values["outcome"] for result in alice_results]
    print(alice_outcomes)

    bob_results = result.bob_results.results
    bob_outcomes = [result.values["outcome"] for result in bob_results]
    print(bob_outcomes)

    charlie_results = result.charlie_results.results
    charlie_outcomes = [result.values["outcome"] for result in charlie_results]
    print(charlie_outcomes)

    assert alice_outcomes == bob_outcomes == charlie_outcomes


if __name__ == "__main__":
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("anon.log")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_tasks_to_file("anon_tasks.log")

    # Alice = sender
    # Charlie = receiver
    anonymous_transfer(num_iterations=1)
