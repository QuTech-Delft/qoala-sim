from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.sim.build import build_network_from_config
from qoala.sim.network import ProcNodeNetwork


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client_{id}",
        node_id=id,
        topology=TopologyConfig.perfect_nv_default_params(1),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("NvNtf"),
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=TopologyConfig.perfect_nv_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("NvNtf"),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    node_cfgs = [server_cfg] + client_configs
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )
    return build_network_from_config(network_cfg)


@dataclass
class BqcResult:
    client_batches: List[Dict[int, ProgramBatch]]
    client_results: List[Dict[int, BatchResult]]


def load_server_program(remote_name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_nv_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = QoalaParser(server_text, flavour=NVFlavour()).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_nv_client.iqoala")
    with open(path) as file:
        client_text = file.read()
    return QoalaParser(client_text, flavour=NVFlavour()).parse()


def create_server_batch(
    client_id: int,
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
) -> BatchInfo:
    server_program = load_server_program(remote_name=f"client_{client_id}")
    return BatchInfo(
        program=server_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )


def create_client_batch(
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
) -> BatchInfo:
    client_program = load_client_program()
    return BatchInfo(
        program=client_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )


def run_bqc(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    num_iterations: List[int],
    num_clients: int,
):
    ns.sim_reset()

    # server needs to have 2 qubits per client
    server_num_qubits = num_clients * 2
    server_config = get_server_config(id=0, num_qubits=server_num_qubits)
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(server_config, client_configs, num_clients)
    server_procnode = network.nodes["server"]
    server_batches: Dict[int, ProgramBatch] = {}  # client ID -> server batch
    client_batches: Dict[int, ProgramBatch] = {}  # client ID -> client batch

    for client_id in range(1, num_clients + 1):
        # index in num_iterations list
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
        )

        server_batches[client_id] = server_procnode.submit_batch(server_batch_info)

    for client_id in range(1, num_clients + 1):
        # index in num_iterations list
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
            client_inputs, client_unit_module, num_iterations[index]
        )

        client_batches[client_id] = client_procnode.submit_batch(client_batch_info)
        batch_id = client_batches[client_id].batch_id
        server_pids = [inst.pid for inst in server_batches[client_id].instances]
        # print(
        #     f"client ID: {client_id}, batch ID: {batch_id}, server PIDs: {server_pids}"
        # )
        client_procnode.initialize_processes(
            remote_pids={batch_id: server_pids}, linear=True
        )

    client_pids = {
        server_batches[client_id].batch_id: [
            inst.pid for inst in client_batches[client_id].instances
        ]
        for client_id in range(1, num_clients + 1)
    }
    # print(f"client PIDs: {client_pids}")
    server_procnode.initialize_processes(remote_pids=client_pids, linear=True)

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


def check_computation(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    expected,
    num_iterations,
    num_clients,
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
        num_clients=num_clients,
    )

    batch_success_probabilities: List[float] = []

    for i in range(num_clients):
        assert len(bqc_result.client_results[i]) == 1
        batch_result = bqc_result.client_results[i][0]
        assert len(bqc_result.client_batches[i]) == 1
        program_batch = bqc_result.client_batches[i][0]
        batch_iterations = program_batch.info.num_iterations

        m2s = [result.values["m2"] for result in batch_result.results]
        correct_outcomes = len([m2 for m2 in m2s if m2 == expected])
        succ_prob = round(correct_outcomes / batch_iterations, 2)
        batch_success_probabilities.append(succ_prob)

    return batch_success_probabilities, makespan


def compute_succ_prob_computation(
    num_clients: int,
    num_iterations: List[int],
) -> Tuple[float, float]:
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    # theta1, theta2
    params = [
        (0, 0),
        (0, 7),
        (2, 22),
        (6, 15),
    ]

    all_probs = []
    all_makespans = []
    for theta1, theta2 in params:
        probs, makespan = check_computation(
            alpha=8,
            beta=24,
            theta1=theta1,
            theta2=theta2,
            dummy0=0,
            dummy1=0,
            expected=1,
            num_iterations=num_iterations,
            num_clients=num_clients,
        )
        all_probs.append(probs[0])
        all_makespans.append(makespan)

    prob = sum(all_probs) / len(all_probs)
    makespan = sum(all_makespans)
    return prob, makespan


def check_trap(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    num_iterations,
    num_clients,
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
        num_clients=num_clients,
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
            # corresponds to "dummy = 1"
            # do normal rotations on qubit 0
            # no rotations on qubit 1
            num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
        else:  # dummy0 = 1
            # corresponds to "dummy = 2"
            # no rotations on qubit 0
            # do normal rotations on qubit 1
            num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

        frac_fail = round(num_fails / batch_iterations, 2)
        batch_success_probabilities.append(1 - frac_fail)

    return batch_success_probabilities, makespan


def compute_succ_prob_trap(
    num_clients: int,
    num_iterations: List[int],
) -> Tuple[float, float]:
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    # dummy0, dummy1
    params = [
        (0, 1),
        (1, 0),
    ]

    all_probs = []
    all_makespans = []
    for dummy0, dummy1 in params:
        probs, makespan = check_trap(
            alpha=8,
            beta=24,
            theta1=2,
            theta2=22,
            dummy0=dummy0,
            dummy1=dummy1,
            num_iterations=num_iterations,
            num_clients=num_clients,
        )
        all_probs.append(probs[0])
        all_makespans.append(makespan)

    prob = sum(all_probs) / len(all_probs)
    makespan = sum(all_makespans)
    return prob, makespan


def bqc_computation(num_clients: int, num_iterations: int):
    succ_probs, makespan = compute_succ_prob_computation(
        num_clients=num_clients,
        num_iterations=[num_iterations] * num_clients,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


def bqc_trap(num_clients: int, num_iterations: int):
    succ_probs, makespan = compute_succ_prob_trap(
        num_clients=num_clients,
        num_iterations=[num_iterations] * num_clients,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


if __name__ == "__main__":
    start = time.time()

    bqc_computation(1, 100)
    bqc_trap(1, 100)

    end = time.time()
    print(f"duration: {round(end - start, 2)} s")
