from __future__ import annotations

import datetime
import json
import math
import os
import random
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.sim.build import build_network_from_config
from qoala.sim.network import ProcNodeNetwork
from qoala.util.logging import LogManager


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


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
        # 50e3 breaks this when -n=5
        single_duration=90e3,  # original value 1e3
        two_instructions=["INSTR_CNOT", "INSTR_CZ"],
        two_duration=100e3,
    )


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client_{id}",
        node_id=id,
        topology=topology_config(20),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=topology_config(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
    num_iterations: List[int],
    cc: float,
    use_netschedule: bool,
    bin_length: float,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    node_cfgs = [server_cfg] + client_configs
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )

    pattern = []
    server_pid_index = 0
    for i in range(num_clients):
        for j in range(num_iterations[i]):
            pattern.append((i + 1, j, 0, server_pid_index))
            server_pid_index += 1

    if use_netschedule:
        network_cfg.netschedule = NetworkScheduleConfig(
            bin_length=bin_length,
            first_bin=0,
            bin_pattern=pattern,
            repeat_period=bin_length * num_clients * max(num_iterations),
        )

    cconns = [
        ClassicalConnectionConfig.from_nodes(i, 0, cc)
        for i in range(1, num_clients + 1)
    ]
    network_cfg.cconns = cconns
    return build_network_from_config(network_cfg)


@dataclass
class BqcResult:
    client_batches: List[Dict[int, ProgramBatch]]
    client_results: List[Dict[int, BatchResult]]


def load_server_program(remote_name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = QoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_client.iqoala")
    with open(path) as file:
        client_text = file.read()
    return QoalaParser(client_text).parse()


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
    alpha: int,
    beta: int,
    theta1: int,
    theta2: int,
    dummy0: int,
    dummy1: int,
    num_iterations: List[int],
    num_clients: int,
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    bin_length: float,
):
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    # seed = random.randint(0, 1000)
    seed = 1  # Fix the seed so that results are the same each time.
    ns.set_random_state(seed=seed)

    server_config = get_server_config(id=0, num_qubits=server_num_qubits)
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(
        server_config,
        client_configs,
        num_clients,
        num_iterations,
        cc,
        use_netschedule,
        bin_length,
    )
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
            remote_pids={batch_id: server_pids}, linear=linear
        )

    client_pids = {
        server_batches[client_id].batch_id: [
            inst.pid for inst in client_batches[client_id].instances
        ]
        for client_id in range(1, num_clients + 1)
    }
    # print(f"client PIDs: {client_pids}")
    server_procnode.initialize_processes(remote_pids=client_pids, linear=linear)

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
    alpha: int,
    beta: int,
    theta1: int,
    theta2: int,
    dummy0: int,
    dummy1: int,
    expected: int,
    num_iterations: int,
    num_clients: int,
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    bin_length: float,
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
        linear=linear,
        cc=cc,
        server_num_qubits=server_num_qubits,
        use_netschedule=use_netschedule,
        bin_length=bin_length,
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
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    bin_length: float,
) -> Tuple[float, float]:
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    all_probs = []
    all_makespans = []
    probs, makespan = check_computation(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=0,
        expected=1,
        num_iterations=num_iterations,
        num_clients=num_clients,
        linear=linear,
        cc=cc,
        server_num_qubits=server_num_qubits,
        use_netschedule=use_netschedule,
        bin_length=bin_length,
    )
    all_probs.append(probs[0])
    all_makespans.append(makespan)

    prob = sum(all_probs) / len(all_probs)
    makespan = sum(all_makespans)
    return prob, makespan


def bqc_computation(
    num_clients: int,
    num_iterations: int,
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    bin_length: float,
) -> float:
    succ_probs, makespan = compute_succ_prob_computation(
        num_clients=num_clients,
        num_iterations=[num_iterations] * num_clients,
        linear=linear,
        cc=cc,
        server_num_qubits=server_num_qubits,
        use_netschedule=use_netschedule,
        bin_length=bin_length,
    )
    # print(f"success probabilities: {succ_probs}")
    # print(f"makespan: {makespan:_}")
    return makespan


def wilson_score_interval(p_hat, n, z):
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p_hat + z**2 / (2 * n)
    adjusted_standard_deviation = z * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n)) / n
    )

    lower_bound = (
        centre_adjusted_probability - adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + adjusted_standard_deviation
    ) / denominator

    return (lower_bound, upper_bound)


@dataclass
class DataPoint:
    num_qubits_server: int
    makespan_linear: float
    makespan_intlv_1: float
    makespan_intlv_2: float
    improv_1: float
    improv_2: float


@dataclass
class DataMeta:
    timestamp: str
    num_clients: int
    num_iterations: int
    cc: float
    intlv_1_bin_length: float
    intlv_2_bin_length: float
    sim_duration: float


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


if __name__ == "__main__":
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("DEBUG")
    LogManager.log_tasks_to_file("multi_bqc_debug_TASKS.log")
    LogManager.log_to_file("multi_bqc_debug.log")

    parser = ArgumentParser()
    parser.add_argument("--num_clients", "-c", type=int, required=True)
    parser.add_argument("--num_iterations", "-n", type=int, required=True)

    args = parser.parse_args()
    num_clients = args.num_clients
    num_iterations = args.num_iterations

    cc = 1e5
    intlv_1_bin_length = 5e4
    intlv_2_bin_length = 1e5

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()

    data_points: List[DataPoint] = []
    for num_qubits in [5]:
        # makespan_linear = bqc_computation(
        #     num_clients, num_iterations, True, cc, num_qubits, False, 0
        # )
        # makespan_interleaved_bin_5e4 = bqc_computation(
        #     num_clients, num_iterations, False, cc, num_qubits, True, intlv_1_bin_length
        # )
        makespan_interleaved_bin_1e5 = bqc_computation(
            num_clients, num_iterations, False, cc, num_qubits, True, intlv_2_bin_length
        )

        print(f"# qubits = {num_qubits}:")
        # print(f"linear: {makespan_linear:_}")
        #     print(f"interleaved, bin 5e4: {makespan_interleaved_bin_5e4:_}")
        print(f"interleaved, bin 1e5: {makespan_interleaved_bin_1e5:_}")

    #     improv_5e4 = round(1 - makespan_interleaved_bin_5e4 / makespan_linear, 3)
    #     improv_1e5 = round(1 - makespan_interleaved_bin_1e5 / makespan_linear, 3)
    #     print(f"improvement 5e4: {improv_5e4}")
    #     print(f"improvement 1e5: {improv_1e5}")

    #     point = DataPoint(
    #         num_qubits_server=num_qubits,
    #         makespan_linear=makespan_linear,
    #         makespan_intlv_1=makespan_interleaved_bin_5e4,
    #         makespan_intlv_2=makespan_interleaved_bin_1e5,
    #         improv_1=improv_5e4,
    #         improv_2=improv_1e5,
    #     )
    #     data_points.append(point)

    # end = time.time()
    # duration = round(end - start, 2)

    # abs_dir = relative_to_cwd(f"data")
    # Path(abs_dir).mkdir(parents=True, exist_ok=True)
    # last_path = os.path.join(abs_dir, "LAST.json")
    # timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    # meta = DataMeta(
    #     timestamp=timestamp,
    #     num_clients=num_clients,
    #     num_iterations=num_iterations,
    #     cc=cc,
    #     intlv_1_bin_length=intlv_1_bin_length,
    #     intlv_2_bin_length=intlv_2_bin_length,
    #     sim_duration=duration,
    # )
    # data = Data(meta=meta, data_points=data_points)
    # json_data = asdict(data)

    # with open(last_path, "w") as datafile:
    #     json.dump(json_data, datafile)
    # with open(timestamp_path, "w") as datafile:
    #     json.dump(json_data, datafile)
