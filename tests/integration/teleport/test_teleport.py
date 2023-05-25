from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.task import TaskGraphBuilder
from qoala.sim.build import build_network


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


def create_procnode_cfg_nv(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_nv_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def create_batch(
    program: QoalaProgram,
    unit_module: UnitModule,
    inputs: List[ProgramInput],
    num_iterations: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        unit_module=unit_module,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
    )


@dataclass
class TeleportResult:
    alice_results: Dict[int, BatchResult]
    bob_results: Dict[int, BatchResult]


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
    network = build_network(network_cfg)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_program = load_program("teleport_alice.iqoala")
    alice_inputs = [ProgramInput({"bob_id": bob_id}) for _ in range(num_iterations)]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(
        alice_program, alice_unit_module, alice_inputs, num_iterations
    )
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_tasks = alice_procnode.scheduler.get_tasks_to_schedule()
    alice_merged = TaskGraphBuilder.merge_linear(alice_tasks)
    alice_procnode.scheduler.upload_task_graph(alice_merged)

    bob_program = load_program("teleport_bob.iqoala")
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, num_iterations)
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_tasks = bob_procnode.scheduler.get_tasks_to_schedule()
    bob_merged = TaskGraphBuilder.merge_linear(bob_tasks)
    bob_procnode.scheduler.upload_task_graph(bob_merged)

    network.start()
    ns.sim_run()

    alice_results = alice_procnode.scheduler.get_batch_results()
    bob_results = bob_procnode.scheduler.get_batch_results()

    return TeleportResult(alice_results, bob_results)


def test_teleport():
    # LogManager.set_log_level("INFO")

    def check(num_iterations):
        ns.sim_reset()
        result = run_teleport(num_iterations=num_iterations)
        assert len(result.alice_results) > 0
        assert len(result.bob_results) > 0

        bob_batch_results = result.bob_results
        for _, batch_results in bob_batch_results.items():
            program_results = batch_results.results
            outcomes = [result.values["outcome"] for result in program_results]
            assert all(outcome == 1 for outcome in outcomes)

    check(10)


if __name__ == "__main__":
    test_teleport()
