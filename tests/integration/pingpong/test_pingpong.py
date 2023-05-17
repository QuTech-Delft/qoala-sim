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
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.task import TaskGraphBuilder
from qoala.sim.build import build_network
from qoala.util.logging import LogManager


def create_network_info(names: List[str]) -> NetworkInfo:
    env = NetworkInfo.with_nodes({i: name for i, name in enumerate(names)})
    env.set_global_schedule([0, 1, 2])
    env.set_timeslot_len(1e6)
    return env


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=100_000, qnos_instr_time=1000
        ),
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
class PingPongResult:
    alice_results: Dict[int, BatchResult]
    bob_results: Dict[int, BatchResult]


def run_pingpong(num_iterations: int) -> PingPongResult:
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["bob", "alice"])
    alice_id = network_info.get_node_id("alice")
    bob_id = network_info.get_node_id("bob")

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=500_000
    )
    network = build_network(network_cfg, network_info)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_program = load_program("pingpong_alice.iqoala")
    alice_inputs = [ProgramInput({"bob_id": bob_id}) for _ in range(num_iterations)]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(
        alice_program, alice_unit_module, alice_inputs, num_iterations
    )
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_task_graphs = alice_procnode.scheduler.get_tasks_to_schedule()
    alice_merged = TaskGraphBuilder.merge_linear(alice_task_graphs)
    alice_tasks = [tinfo.task for tinfo in alice_merged.get_tasks().values()]
    print("Alice tasks:")
    print([str(t) for t in alice_tasks])
    alice_task_start_times = [
        (alice_tasks[0], 0),
        (alice_tasks[1], 500),
        (alice_tasks[2], 25_000),
        (alice_tasks[3], 600_000),
        (alice_tasks[4], 850_000),
        (alice_tasks[5], 1_200_000),
        (alice_tasks[6], 1_700_000),
        (alice_tasks[7], 2_100_000),
        (alice_tasks[8], 2_200_500),
        (alice_tasks[9], 2_220_000),
        (alice_tasks[10], 2_228_000),
    ]
    alice_graph = TaskGraphBuilder.linear_tasks_with_start_times(alice_task_start_times)
    print("\nAlice graph:")
    print(alice_graph)
    alice_procnode.scheduler.upload_task_graph(alice_graph)

    bob_program = load_program("pingpong_bob.iqoala")
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(bob_program, bob_unit_module, bob_inputs, num_iterations)
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_task_graphs = bob_procnode.scheduler.get_tasks_to_schedule()
    bob_merged = TaskGraphBuilder.merge_linear(bob_task_graphs)
    bob_tasks = [tinfo.task for tinfo in bob_merged.get_tasks().values()]
    print("\n\nBob tasks:")
    print([str(t) for t in bob_tasks])
    bob_task_start_times = [
        (bob_tasks[0], 0),
        (bob_tasks[1], 25_000),
        (bob_tasks[2], 900_000),
        (bob_tasks[3], 1_000_000),
        (bob_tasks[4], 1_100_000),
        (bob_tasks[5], 1_200_000),
        (bob_tasks[6], 1_700_000),
        (bob_tasks[7], 1_800_000),
    ]
    bob_graph = TaskGraphBuilder.linear_tasks_with_start_times(bob_task_start_times)
    print("\nBob graph:")
    print(bob_graph)
    bob_procnode.scheduler.upload_task_graph(bob_graph)

    network.start()
    ns.sim_run()

    alice_results = alice_procnode.scheduler.get_batch_results()
    bob_results = bob_procnode.scheduler.get_batch_results()

    return PingPongResult(alice_results, bob_results)


def test_pingpong():
    LogManager.set_log_level("INFO")

    def check(num_iterations):
        ns.sim_reset()
        result = run_pingpong(num_iterations=num_iterations)
        assert len(result.alice_results) > 0
        assert len(result.bob_results) > 0

        alice_batch_results = result.alice_results
        for _, batch_results in alice_batch_results.items():
            program_results = batch_results.results
            outcomes = [result.values["outcome"] for result in program_results]
            assert all(outcome == 1 for outcome in outcomes)

    check(1)


if __name__ == "__main__":
    test_pingpong()
