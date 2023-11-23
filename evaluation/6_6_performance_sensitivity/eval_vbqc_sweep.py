from __future__ import annotations

import datetime
import json
import math
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def topology_config(num_qubits: int, t1: int, t2: int) -> TopologyConfig:
    return TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
        num_qubits, t1=t1, t2=t2
    )


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client_{id}",
        node_id=id,
        topology=topology_config(1, t1=0, t2=0),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def get_server_config(
    id: int, num_qubits: int, t1: int, t2: int, sched_latency: float
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=topology_config(num_qubits, t1=t1, t2=t2),
        latencies=LatenciesConfig(
            host_instr_time=500,
            host_peer_latency=30_000,
            qnos_instr_time=1000,
            internal_sched_latency=sched_latency,
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
    cc: float,
    bin_length: float,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    node_cfgs = [server_cfg] + client_configs
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )

    pattern = [
        (client_cfg.node_id, 0, server_cfg.node_id, i)
        for i, client_cfg in enumerate(client_configs)
    ]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=bin_length,
        first_bin=0,
        bin_pattern=pattern,
        repeat_period=bin_length * num_clients,
    )

    cconns = [
        ClassicalConnectionConfig.from_nodes(server_cfg.node_id, client_cfg.node_id, cc)
        for client_cfg in client_configs
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
    t1: int,
    t2: int,
    cc: float,
    sched_latency: float,
    bin_length: float,
):
    ns.sim_reset()

    # server needs to have 2 qubits per client
    server_num_qubits = num_clients * 2
    server_config = get_server_config(
        id=0, num_qubits=server_num_qubits, t1=t1, t2=t2, sched_latency=sched_latency
    )
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(
        server_config,
        client_configs,
        num_clients,
        cc=cc,
        bin_length=bin_length,
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
            remote_pids={batch_id: server_pids}, linear=True
        )

    client_pids = {
        server_batches[client_id].batch_id: [
            inst.pid for inst in client_batches[client_id].instances
        ]
        for client_id in range(1, num_clients + 1)
    }
    # print(f"client PIDs: {client_pids}")
    server_procnode.initialize_processes(remote_pids=client_pids, linear=False)

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


def check_trap(
    alpha: int,
    beta: int,
    theta1: int,
    theta2: int,
    dummy0: int,
    dummy1: int,
    num_iterations: List[int],
    num_clients: int,
    t1: int,
    t2: int,
    cc: float,
    sched_latency: float,
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
        t1=t1,
        t2=t2,
        cc=cc,
        sched_latency=sched_latency,
        bin_length=bin_length,
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
    t1: int,
    t2: int,
    cc: float,
    sched_latency: float,
    bin_length: float,
):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    return check_trap(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=1,
        num_iterations=num_iterations,
        num_clients=num_clients,
        t1=t1,
        t2=t2,
        cc=cc,
        sched_latency=sched_latency,
        bin_length=bin_length,
    )


def bqc_trap(
    num_clients: int,
    num_iterations: int,
    t1: int,
    t2: int,
    cc: float,
    sched_latency: float,
    bin_length: float,
) -> Tuple[float, float]:
    succ_probs, makespan = compute_succ_prob_trap(
        num_clients=num_clients,
        num_iterations=[num_iterations] * num_clients,
        t1=t1,
        t2=t2,
        cc=cc,
        sched_latency=sched_latency,
        bin_length=bin_length,
    )
    # print(f"success probabilities: {succ_probs}")
    # print(f"makespan: {makespan}")

    avg_succ_prob = sum(succ_probs) / len(succ_probs)
    return avg_succ_prob, makespan


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
    sweep_param: str
    sweep_value: float
    succ_prob: float
    succ_lo: float
    succ_hi: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_runs: int
    sim_duration: float
    num_clients: int
    t1: float
    t2: float
    sweep_param: str
    cc: Optional[float]
    sched_latency: Optional[float]
    bin_length: Optional[float]


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def sweep_cc(num_runs: int) -> Data:
    num_clients = 10
    t1 = 1e8
    t2 = 1e7
    sched_latency = 0
    bin_length = 1e5

    data_points: List[DataPoint] = []

    start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for cc in [1e5, 1e6, 1e7]:
        successes: List[float] = []
        makespans: List[float] = []

        for _ in range(num_runs):
            succ, makespan = bqc_trap(
                num_clients,
                1,
                t1=t1,
                t2=t2,
                cc=cc,
                sched_latency=sched_latency,
                bin_length=bin_length,
            )
            successes.append(succ)
            makespans.append(makespan)

        avg_succ = round(sum(successes) / len(successes), 3)
        total_makespan = sum(makespans)

        succ_lo, succ_hi = wilson_score_interval(
            p_hat=avg_succ, n=len(successes) * num_clients, z=1.96
        )
        succ_lo = round(succ_lo, 3)
        succ_hi = round(succ_hi, 3)

        print(
            f"CC = {cc}: succ = {avg_succ} ({succ_lo}, {succ_hi}), makespan = {total_makespan:_}"
        )

        data_point = DataPoint(
            sweep_param="cc",
            sweep_value=cc,
            succ_prob=avg_succ,
            succ_lo=succ_lo,
            succ_hi=succ_hi,
            makespan=total_makespan,
        )
        data_points.append(data_point)

    end = time.time()
    duration = round(end - start, 2)
    meta = DataMeta(
        timestamp=timestamp,
        num_runs=num_runs,
        sim_duration=duration,
        num_clients=num_clients,
        t1=t1,
        t2=t2,
        sweep_param="cc",
        cc=None,
        sched_latency=sched_latency,
        bin_length=bin_length,
    )
    return Data(meta, data_points)


def sweep_sched_latency(num_runs: int) -> Data:
    num_clients = 10
    t1 = 1e8
    t2 = 1e7
    cc = 1e5
    bin_length = 1e5

    data_points: List[DataPoint] = []

    start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for sched_latency in [1e3, 1e5, 1e7]:
        successes: List[float] = []
        makespans: List[float] = []

        for _ in range(num_runs):
            succ, makespan = bqc_trap(
                num_clients,
                1,
                t1=t1,
                t2=t2,
                cc=cc,
                sched_latency=sched_latency,
                bin_length=bin_length,
            )
            successes.append(succ)
            makespans.append(makespan)

        avg_succ = round(sum(successes) / len(successes), 3)
        total_makespan = sum(makespans)

        succ_lo, succ_hi = wilson_score_interval(
            p_hat=avg_succ, n=len(successes) * num_clients, z=1.96
        )
        succ_lo = round(succ_lo, 3)
        succ_hi = round(succ_hi, 3)

        print(
            f"Sched latency = {sched_latency}: succ = {avg_succ} ({succ_lo}, {succ_hi}), "
            f"makespan = {total_makespan:_}"
        )

        data_point = DataPoint(
            sweep_param="sched_latency",
            sweep_value=sched_latency,
            succ_prob=avg_succ,
            succ_lo=succ_lo,
            succ_hi=succ_hi,
            makespan=total_makespan,
        )
        data_points.append(data_point)

    end = time.time()
    duration = round(end - start, 2)
    meta = DataMeta(
        timestamp=timestamp,
        num_runs=num_runs,
        sim_duration=duration,
        num_clients=num_clients,
        t1=t1,
        t2=t2,
        sweep_param="sched_latency",
        cc=cc,
        sched_latency=None,
        bin_length=bin_length,
    )
    return Data(meta, data_points)


def sweep_bin_length(num_runs: int) -> Data:
    num_clients = 10
    t1 = 1e8
    t2 = 1e7
    cc = 1e5
    sched_latency = 0

    data_points: List[DataPoint] = []

    start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for bin_length in [1e5, 1e6, 1e7]:
        successes: List[float] = []
        makespans: List[float] = []

        for _ in range(num_runs):
            succ, makespan = bqc_trap(
                num_clients,
                1,
                t1=t1,
                t2=t2,
                cc=cc,
                sched_latency=sched_latency,
                bin_length=bin_length,
            )
            successes.append(succ)
            makespans.append(makespan)

        avg_succ = round(sum(successes) / len(successes), 3)
        total_makespan = sum(makespans)

        succ_lo, succ_hi = wilson_score_interval(
            p_hat=avg_succ, n=len(successes) * num_clients, z=1.96
        )
        succ_lo = round(succ_lo, 3)
        succ_hi = round(succ_hi, 3)

        print(
            f"Sched latency = {sched_latency}: succ = {avg_succ} ({succ_lo}, {succ_hi}), "
            f"makespan = {total_makespan:_}"
        )

        data_point = DataPoint(
            sweep_param="bin_length",
            sweep_value=bin_length,
            succ_prob=avg_succ,
            succ_lo=succ_lo,
            succ_hi=succ_hi,
            makespan=total_makespan,
        )
        data_points.append(data_point)

    end = time.time()
    duration = round(end - start, 2)
    meta = DataMeta(
        timestamp=timestamp,
        num_runs=num_runs,
        sim_duration=duration,
        num_clients=num_clients,
        t1=t1,
        t2=t2,
        sweep_param="bin_length",
        cc=cc,
        sched_latency=sched_latency,
        bin_length=None,
    )
    return Data(meta, data_points)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_runs", "-n", type=int, required=True)
    parser.add_argument("--sweep", "-s", type=str, required=True)

    args = parser.parse_args()
    num_runs = args.num_runs
    sweep = args.sweep

    data: Data
    output_dir: str

    if sweep == "cc":
        output_dir = "sweep_cc"
        data = sweep_cc(num_runs)
    elif sweep == "sched_latency":
        output_dir = "sweep_sched_latency"
        data = sweep_sched_latency(num_runs)
    elif sweep == "bin_length":
        output_dir = "sweep_bin_length"
        data = sweep_bin_length(num_runs)
    else:
        raise ValueError

    abs_dir = relative_to_cwd(f"data/{output_dir}")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp = data.meta.timestamp
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    json_data = asdict(data)

    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
