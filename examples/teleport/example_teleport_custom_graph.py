from __future__ import annotations

import os
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
from qoala.runtime.task import TaskGraph
from qoala.runtime.taskbuilder import TaskGraphBuilder
from qoala.sim.build import build_network_from_config
from qoala.util.runner import create_batch


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
        is_predictable=True,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class TeleportResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_teleport(num_iterations: int) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 4
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, determ=True)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e9)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]
    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=1_500, first_bin=0, bin_pattern=pattern, repeat_period=2e7
    )

    alice_program = load_program("teleport_alice.iqoala")
    bob_program = load_program("teleport_bob.iqoala")

    # state = 5 -> teleport |1> state
    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "state": 5}) for _ in range(num_iterations)
    ]
    # state = 4 -> measure in +Z, i.e. expect a "1" outcome
    bob_inputs = [
        ProgramInput({"alice_id": alice_id, "state": 4}) for _ in range(num_iterations)
    ]

    network = build_network_from_config(network_cfg)

    batches: Dict[str, ProgramBatch] = {}  # node -> batch

    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch_info = create_batch(
        alice_program, alice_unit_module, alice_inputs, num_iterations
    )
    batches["alice"] = alice_procnode.submit_batch(alice_batch_info)

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch_info = create_batch(
        bob_program, bob_unit_module, bob_inputs, num_iterations
    )
    batches["bob"] = bob_procnode.submit_batch(bob_batch_info)

    alice_remote_pids = {
        batches["bob"].batch_id: [p.pid for p in batches["bob"].instances]
    }
    alice_procnode.initialize_processes(alice_remote_pids)
    bob_remote_pids = {
        batches["alice"].batch_id: [p.pid for p in batches["alice"].instances]
    }
    bob_procnode.initialize_processes(bob_remote_pids)

    alice_task_counter = 0
    alice_tasks: List[TaskGraph] = []
    for i, inst in enumerate(batches["alice"].instances):
        tasks = TaskGraphBuilder.from_program(
            alice_program,
            inst.pid,
            alice_procnode.local_ehi,
            alice_procnode.network_ehi,
            first_task_id=alice_task_counter,
            prog_input=alice_inputs[i].values,
        )
        alice_tasks.append(tasks)
        alice_task_counter += len(tasks.get_tasks())
    alice_full_graph = TaskGraphBuilder.merge_linear(alice_tasks)
    alice_procnode.scheduler.upload_task_graph(alice_full_graph)

    bob_task_counter = 0
    bob_tasks: List[TaskGraph] = []
    for i, inst in enumerate(batches["bob"].instances):
        tasks = TaskGraphBuilder.from_program(
            bob_program,
            inst.pid,
            bob_procnode.local_ehi,
            bob_procnode.network_ehi,
            first_task_id=bob_task_counter,
            prog_input=bob_inputs[i].values,
        )
        bob_tasks.append(tasks)
        bob_task_counter += len(tasks.get_tasks())
    bob_full_graph = TaskGraphBuilder.merge_linear(bob_tasks)
    bob_procnode.scheduler.upload_task_graph(bob_full_graph)

    # app_result = run_two_node_app(
    #     num_iterations=num_iterations,
    #     programs={"alice": alice_program, "bob": bob_program},
    #     program_inputs={"alice": alice_input, "bob": bob_input},
    #     network_cfg=network_cfg,
    # )

    network.start()
    ns.sim_run()

    alice_result = alice_procnode.scheduler.get_batch_results()[0]
    bob_result = bob_procnode.scheduler.get_batch_results()[0]

    return TeleportResult(alice_result, bob_result)


if __name__ == "__main__":
    # LogManager.set_log_level("DEBUG")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_to_file("teleport.log")
    # LogManager.log_tasks_to_file("teleport_tasks.log")
    num_iterations = 3

    result = run_teleport(num_iterations=num_iterations)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)
