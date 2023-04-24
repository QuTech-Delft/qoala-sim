from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    GenericQDeviceConfig,
    LatenciesConfig,
    LinkBetweenNodesConfig,
    LinkConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.schedule import StaticSchedule
from qoala.sim.build import build_network
from qoala.sim.network import ProcNodeNetwork

INSTR_LATENCY = 1e5
CC_LATENCY = 1e6
QC_EXPECTATION = 30e6


def create_network_info(
    num_clients: int, global_schedule: List[int], timeslot_len: int
) -> NetworkInfo:
    nodes = {i: f"client_{i}" for i in range(1, num_clients + 1)}
    nodes[0] = "server"
    env = NetworkInfo.with_nodes(nodes)

    env.set_global_schedule(global_schedule)
    env.set_timeslot_len(timeslot_len)
    return env


def topology_config(num_qubits: int) -> TopologyConfig:
    return TopologyConfig.perfect_config_uniform(
        num_qubits,
        single_instructions=[
            "INSTR_INIT",
            "INSTR_ROT_X",
            "INSTR_ROT_Y",
            "INSTR_ROT_Z",
            "INSTR_X",
            "INSTR_Y",
            "INSTR_Z",
            "INSTR_H",
            "INSTR_MEASURE",
        ],
        single_duration=1e3,
        two_instructions=["INSTR_CNOT", "INSTR_CZ"],
        two_duration=100e3,
    )


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client_{id}",
        node_id=id,
        topology=topology_config(1),
        latencies=LatenciesConfig(
            host_instr_time=INSTR_LATENCY,
            host_peer_latency=CC_LATENCY,
            qnos_instr_time=INSTR_LATENCY,
        ),
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    config_file = relative_to_cwd("node_config.yaml")
    qdevice_cfg = GenericQDeviceConfig.from_file(config_file)
    qdevice_cfg.num_qubits = num_qubits
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=topology_config(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=INSTR_LATENCY,
            host_peer_latency=CC_LATENCY,
            qnos_instr_time=INSTR_LATENCY,
        ),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
    global_schedule: List[int],
    timeslot_len: int,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    network_info = create_network_info(num_clients, global_schedule, timeslot_len)

    node_cfgs = [server_cfg] + client_configs

    link_config = LinkConfig.from_file(relative_to_cwd("link_config.yaml"))
    links = [
        LinkBetweenNodesConfig(
            node_id1=server_cfg.node_id, node_id2=cfg.node_id, link_config=link_config
        )
        for cfg in client_configs
    ]

    network_cfg = ProcNodeNetworkConfig(nodes=node_cfgs, links=links)
    return build_network(network_cfg, network_info)


@dataclass
class BqcResult:
    client_batches: List[Dict[int, ProgramBatch]]
    client_results: List[Dict[int, BatchResult]]


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def load_server_program(remote_name: str) -> QoalaProgram:
    path = relative_to_cwd("server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = QoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> QoalaProgram:
    path = relative_to_cwd("client.iqoala")
    with open(path) as file:
        client_text = file.read()
    return QoalaParser(client_text).parse()


def create_server_batch(
    client_id: int,
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
    deadline: int,
) -> BatchInfo:
    server_program = load_server_program(remote_name=f"client_{client_id}")
    return BatchInfo(
        program=server_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=deadline,
    )


def create_client_batch(
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
    deadline: int,
) -> BatchInfo:
    client_program = load_client_program()
    return BatchInfo(
        program=client_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=deadline,
    )


def run_bqc(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    num_iterations: List[int],
    deadlines: List[int],
    num_clients: int,
    global_schedule: List[int],
    timeslot_len: int,
):
    # server needs to have 2 qubits per client
    server_num_qubits = num_clients * 2
    server_config = get_server_config(id=0, num_qubits=server_num_qubits)
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(
        server_config, client_configs, num_clients, global_schedule, timeslot_len
    )

    server_procnode = network.nodes["server"]

    for client_id in range(1, num_clients + 1):
        # index in num_iterations and deadlines list
        index = client_id - 1

        server_inputs = [
            ProgramInput({"client_id": client_id}) for _ in range(num_iterations[index])
        ]

        server_unit_module = UnitModule.from_full_ehi(server_procnode.memmgr.get_ehi())
        server_batch_info = create_server_batch(
            client_id=client_id,
            inputs=server_inputs,
            unit_module=server_unit_module,
            num_iterations=num_iterations[index],
            deadline=deadlines[index],
        )

        server_procnode.submit_batch(server_batch_info)
    server_procnode.initialize_processes()
    server_tasks = server_procnode.scheduler.get_tasks_to_schedule()
    server_schedule = StaticSchedule.consecutive_block_tasks(server_tasks)
    server_procnode.scheduler.upload_schedule(server_schedule)

    for client_id in range(1, num_clients + 1):
        # index in num_iterations and deadlines list
        index = client_id - 1

        client_inputs = [
            ProgramInput(
                {
                    "server_id": 0,
                    "alpha": alpha,
                    "beta": beta,
                    "theta1": theta1,
                    "theta2": theta2,
                    "dummy0": dummy0,
                    "dummy1": dummy1,
                }
            )
            for _ in range(num_iterations[index])
        ]

        client_procnode = network.nodes[f"client_{client_id}"]
        client_unit_module = UnitModule.from_full_ehi(client_procnode.memmgr.get_ehi())

        client_batch_info = create_client_batch(
            client_inputs, client_unit_module, num_iterations[index], deadlines[index]
        )

        client_procnode.submit_batch(client_batch_info)
        client_procnode.initialize_processes()
        client_tasks = client_procnode.scheduler.get_tasks_to_schedule()
        client_schedule = StaticSchedule.consecutive_block_tasks(client_tasks)
        client_procnode.scheduler.upload_schedule(client_schedule)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    client_procnodes = [network.nodes[f"client_{i}"] for i in range(1, num_clients + 1)]
    client_batches = [node.get_batches() for node in client_procnodes]

    client_results: List[Dict[int, BatchResult]]
    client_results = [node.scheduler.get_batch_results() for node in client_procnodes]

    return BqcResult(client_batches, client_results), makespan


def check(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    expected,
    num_iterations,
    deadlines,
    num_clients,
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.sim_reset()

    bqc_result, makespan = run_bqc(
        alpha=alpha,
        beta=beta,
        theta1=theta1,
        theta2=theta2,
        dummy0=dummy0,
        dummy1=dummy1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )

    batch_success_probabilities: List[float] = []

    for i in range(num_clients):
        assert len(bqc_result.client_results[i]) == 1
        batch_result = bqc_result.client_results[i][0]
        assert len(bqc_result.client_batches[i]) == 1
        program_batch = bqc_result.client_batches[i][0]
        batch_iterations = program_batch.info.num_iterations

        p1s = [result.values["p1"] for result in batch_result.results]
        p2s = [result.values["p2"] for result in batch_result.results]
        m1s = [result.values["m1"] for result in batch_result.results]
        m2s = [result.values["m2"] for result in batch_result.results]

        if dummy0 == 0:
            num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
        else:  # dummy0 = 1
            num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

        frac_fail = round(num_fails / batch_iterations, 2)
        batch_success_probabilities.append(1 - frac_fail)

    return batch_success_probabilities, makespan


def compute_succ_prob(
    num_clients: int,
    num_iterations: List[int],
    deadlines: List[int],
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    return check(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=1,
        expected=1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )


def test_bqc():
    # LogManager.set_log_level("INFO")
    num_clients = 10
    succ_probs, makespan = compute_succ_prob(
        num_clients=num_clients,
        num_iterations=[30] * num_clients,
        deadlines=[1e8] * num_clients,
        global_schedule=[i for i in range(num_clients)],
        timeslot_len=50e6,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


if __name__ == "__main__":
    test_bqc()
