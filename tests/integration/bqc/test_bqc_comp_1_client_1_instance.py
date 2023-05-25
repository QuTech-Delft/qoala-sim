from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import netsquid as ns

from qoala.lang.ehi import EhiNodeInfo, UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.task import TaskExecutionMode, TaskGraph, TaskGraphBuilder
from qoala.sim.build import build_network


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=1_000_000, qnos_instr_time=1000
        ),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class SimpleBqcResult:
    client_result: Dict[str, int]
    server_result: Dict[str, int]


def instantiate(
    program: QoalaProgram,
    ehi: EhiNodeInfo,
    pid: int = 0,
    inputs: Optional[ProgramInput] = None,
) -> ProgramInstance:
    unit_module = UnitModule.from_full_ehi(ehi)

    if inputs is None:
        inputs = ProgramInput.empty()

    return ProgramInstance(
        pid,
        program,
        inputs,
        unit_module=unit_module,
        task_graph=TaskGraph(),
    )


def run_bqc(
    alpha,
    beta,
    theta1,
    theta2,
    tem: TaskExecutionMode = TaskExecutionMode.BLOCK,
) -> SimpleBqcResult:
    ns.sim_reset()

    num_qubits = 3
    client_id = 0
    server_id = 1

    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    server_node_cfg.tem = tem.name
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)
    client_node_cfg.tem = tem.name

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )
    network = build_network(network_cfg)
    server_procnode = network.nodes["server"]
    client_procnode = network.nodes["client"]

    server_program = load_program("bqc_server.iqoala")
    server_input = ProgramInput({"client_id": client_id})
    server_instance = instantiate(
        server_program, server_procnode.local_ehi, 0, server_input
    )
    server_procnode.scheduler.submit_program_instance(server_instance)

    client_program = load_program("bqc_client.iqoala")
    client_input = ProgramInput(
        {
            "server_id": server_id,
            "alpha": alpha,
            "beta": beta,
            "theta1": theta1,
            "theta2": theta2,
        }
    )
    client_instance = instantiate(
        client_program, client_procnode.local_ehi, 0, client_input
    )
    client_procnode.scheduler.submit_program_instance(client_instance)

    tasks_server = TaskGraphBuilder.from_file_block_tasks(
        server_program,
        0,
        server_procnode.local_ehi,
        server_procnode.network_ehi,
        client_id,
    )
    tasks_client = TaskGraphBuilder.from_file_block_tasks(
        client_program,
        0,
        client_procnode.local_ehi,
        client_procnode.network_ehi,
        server_id,
    )

    server_procnode.scheduler.upload_task_graph(tasks_server)
    client_procnode.scheduler.upload_task_graph(tasks_client)

    network.start()
    ns.sim_run()

    client_result = client_procnode.memmgr.get_process(0).result.values
    server_result = server_procnode.memmgr.get_process(0).result.values

    return SimpleBqcResult(client_result, server_result)


def check(alpha, beta, theta1, theta2, expected, num_iterations, tem):
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    ns.sim_reset()

    for _ in range(num_iterations):
        bqc_result = run_bqc(
            alpha=alpha, beta=beta, theta1=theta1, theta2=theta2, tem=tem
        )
        assert bqc_result.server_result["m2"] == expected


def test_bqc_block_tasks():
    tem = TaskExecutionMode.BLOCK
    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=10, tem=tem)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1, num_iterations=10, tem=tem)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0, num_iterations=10, tem=tem)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1, num_iterations=10, tem=tem)


def test_bqc_qoala_tasks():
    tem = TaskExecutionMode.QOALA
    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=10, tem=tem)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1, num_iterations=10, tem=tem)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0, num_iterations=10, tem=tem)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1, num_iterations=10, tem=tem)


if __name__ == "__main__":
    test_bqc_block_tasks()
    test_bqc_qoala_tasks()
