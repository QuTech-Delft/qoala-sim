from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import netsquid as ns

from qoala.lang.ehi import ExposedHardwareInfo, UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.schedule import TaskSchedule
from qoala.runtime.task import TaskCreator, TaskExecutionMode
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
    ehi: ExposedHardwareInfo,
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
        block_tasks=[],
    )


def run_bqc(alpha, beta, theta1, theta2) -> SimpleBqcResult:
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["client", "server"])
    server_id = network_info.get_node_id("server")
    client_id = network_info.get_node_id("client")

    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )
    network = build_network(network_cfg, network_info)
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

    task_creator = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC)
    tasks_server = task_creator.from_program(
        server_program,
        0,
        server_procnode.local_ehi,
        server_procnode.network_ehi,
        client_id,
    )
    tasks_client = task_creator.from_program(
        client_program,
        0,
        client_procnode.local_ehi,
        client_procnode.network_ehi,
        server_id,
    )

    schedule_server = TaskSchedule.consecutive(tasks_server)
    schedule_client = TaskSchedule.consecutive(tasks_client)
    server_procnode.scheduler.upload_schedule(schedule_server)
    client_procnode.scheduler.upload_schedule(schedule_client)

    network.start()
    ns.sim_run()

    client_result = client_procnode.memmgr.get_process(0).result.values
    server_result = server_procnode.memmgr.get_process(0).result.values

    return SimpleBqcResult(client_result, server_result)


def test_bqc():
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    LogManager.set_log_level("WARNING")

    def check(alpha, beta, theta1, theta2, expected, num_iterations):
        ns.sim_reset()

        for _ in range(num_iterations):
            bqc_result = run_bqc(alpha=alpha, beta=beta, theta1=theta1, theta2=theta2)
            assert bqc_result.server_result["m2"] == expected

    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=10)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1, num_iterations=10)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0, num_iterations=10)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1, num_iterations=10)


if __name__ == "__main__":
    test_bqc()
