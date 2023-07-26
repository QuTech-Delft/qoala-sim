from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import (
    run_1_server_n_clients,
    run_two_node_app_separate_inputs,
    run_two_node_app_separate_inputs_plus_local_program,
)


def create_procnode_cfg(
    name: str,
    id: int,
    t1: float,
    t2: float,
    determ: bool,
    deadlines: bool,
    num_qubits: int,
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
            num_qubits, t1, t2
        ),
        latencies=LatenciesConfig(qnos_instr_time=1000, host_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
        use_deadlines=deadlines,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class TeleportResult:
    alice_result: BatchResult
    bob_result: BatchResult
    local_result: BatchResult
    total_duration: float


def run_apps(
    num_iterations: int,
    num_local_iterations: int,
    num_qubits_bob: int,
    t1: float,
    t2: float,
    cc_latency: float,
    network_bin_len: int,
    network_period: int,
    network_first_bin: int,
) -> TeleportResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice", alice_id, t1, t2, determ=True, deadlines=True, num_qubits=10
    )
    bob_node_cfg = create_procnode_cfg(
        "bob", bob_id, t1, t2, determ=True, deadlines=True, num_qubits=num_qubits_bob
    )

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc_latency)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=network_bin_len,
        first_bin=network_first_bin,
        bin_pattern=pattern,
        repeat_period=network_period,
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("programs/teleport_alice.iqoala")
    bob_program = load_program("programs/teleport_bob.iqoala")
    local_program = load_program("programs/local_quantum.iqoala")

    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "state": i % 6}) for i in range(num_iterations)
    ]
    bob_inputs = [
        ProgramInput({"alice_id": alice_id, "state": i % 6})
        for i in range(num_iterations)
    ]

    app_result = run_two_node_app_separate_inputs_plus_local_program(
        num_iterations=num_iterations,
        num_local_iterations=num_local_iterations,
        node1="alice",
        node2="bob",
        prog_node1=alice_program,
        prog_node1_inputs=alice_inputs,
        prog_node2=bob_program,
        prog_node2_inputs=bob_inputs,
        local_prog_node2=local_program,
        local_prog_node2_inputs=[
            ProgramInput.empty() for _ in range(num_local_iterations)
        ],
        network_cfg=network_cfg,
        linear_for={"alice": True, "bob": False},
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]
    local_result = app_result.batch_results["local"]
    print(local_result)

    return TeleportResult(
        alice_result, bob_result, local_result, app_result.total_duration
    )


@dataclass
class DataPoint:
    t2: float
    cc_latency: float
    tel_succ_prob: float
    tel_succ_prob_lower: float
    tel_succ_prob_upper: float
    loc_succ_prob: float
    loc_succ_prob_lower: float
    loc_succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    latency_factor: float
    net_period_factors: List[float]


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


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


def get_metrics(
    num_iterations: int,
    num_local_iterations: int,
    num_qubits_bob: int,
    t1: int,
    t2: int,
    latency_factor: float,
    network_bin_len: int,
    network_period: int,
    network_first_bin: int,
) -> DataPoint:
    teleport_successes: List[bool] = []
    local_successes: List[bool] = []

    cc_latency = latency_factor * t2

    result = run_apps(
        num_iterations=num_iterations,
        num_local_iterations=num_local_iterations,
        num_qubits_bob=num_qubits_bob,
        t1=t1,
        t2=t2,
        cc_latency=cc_latency,
        network_period=network_period,
        network_bin_len=network_bin_len,
        network_first_bin=network_first_bin,
    )
    teleport_results = result.bob_result.results
    outcomes = [result.values["outcome"] for result in teleport_results]
    assert len(outcomes) == num_iterations
    teleport_successes.extend([outcomes[i] == 0 for i in range(num_iterations)])

    local_results = result.local_result.results
    outcomes = [result.values["outcome"] for result in local_results]
    assert len(outcomes) == num_local_iterations
    local_successes.extend([outcomes[i] == 1 for i in range(num_local_iterations)])

    tel_avg_succ_prob = sum([s for s in teleport_successes if s]) / len(
        teleport_successes
    )
    tel_succ_prob_lower, tel_succ_prob_upper = wilson_score_interval(
        p_hat=tel_avg_succ_prob, n=len(teleport_successes), z=1.96
    )
    tel_lower_rounded = round(tel_succ_prob_lower, 3)
    tel_upper_rounded = round(tel_succ_prob_upper, 3)
    tel_succprob_rounded = round(tel_avg_succ_prob, 3)

    loc_avg_succ_prob = sum([s for s in local_successes if s]) / len(local_successes)
    loc_succ_prob_lower, loc_succ_prob_upper = wilson_score_interval(
        p_hat=loc_avg_succ_prob, n=len(local_successes), z=1.96
    )
    loc_lower_rounded = round(loc_succ_prob_lower, 3)
    loc_upper_rounded = round(loc_succ_prob_upper, 3)
    loc_succprob_rounded = round(loc_avg_succ_prob, 3)

    makespan = result.total_duration

    print(
        f"teleport succ prob: {tel_succprob_rounded} ({tel_lower_rounded}, {tel_upper_rounded})"
    )
    print(
        f"local succ prob: {loc_succprob_rounded} ({loc_lower_rounded}, {loc_upper_rounded})"
    )
    print(f"makespan: {makespan:_}")

    return DataPoint(
        t2=t2,
        cc_latency=cc_latency,
        tel_succ_prob=tel_avg_succ_prob,
        tel_succ_prob_lower=tel_succ_prob_lower,
        tel_succ_prob_upper=tel_succ_prob_upper,
        loc_succ_prob=loc_avg_succ_prob,
        loc_succ_prob_lower=loc_succ_prob_lower,
        loc_succ_prob_upper=loc_succ_prob_upper,
        makespan=makespan,
    )


def run(output_dir: str, save: bool = True):
    # LogManager.set_log_level("INFO")
    # LogManager.log_to_file("quantum_multitasking.log")
    # LogManager.set_task_log_level("INFO")
    # LogManager.log_tasks_to_file("quantum_multitasking_tasks.log")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t1 = 1e10
    t2 = 1e8

    num_iterations = 100
    num_local_iterations = 1000
    num_qubits_bob = 50
    latency_factor = 1.0  # CC = 10_000_000
    net_period_factors = [0.01, 0.1, 1, 10, 100, 1000]
    network_bin_len = int((latency_factor * t2) / 5)  # 2_000_000
    network_first_bin = 1_000_000

    data_points: List[DataPoint] = []

    network_period = (num_iterations + 1) * network_bin_len

    data_point = get_metrics(
        num_iterations=num_iterations,
        num_local_iterations=num_local_iterations,
        num_qubits_bob=num_qubits_bob,
        t1=t1,
        t2=t2,
        latency_factor=latency_factor,
        network_bin_len=network_bin_len,
        network_period=network_period,
        network_first_bin=network_first_bin,
    )
    data_points.append(data_point)

    meta = DataMeta(
        timestamp=timestamp,
        num_iterations=num_iterations,
        latency_factor=latency_factor,
        net_period_factors=net_period_factors,
    )

    end_time = time.time()
    makespan = data_point.makespan
    num_bins = makespan / network_bin_len
    # print(f"total duration: {end_time - start_time}s")

    print(f"cc latency: {latency_factor * t2:_}")
    print(f"network period: {network_period:_}")
    print(f"network bin len: {network_bin_len:_}")
    print(f"makespan: {makespan}")
    print(f"num bins: {num_bins}")

    data = Data(meta=meta, data_points=data_points)

    json_data = asdict(data)

    abs_dir = relative_to_cwd(f"data/{output_dir}")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, f"LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")
    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)


if __name__ == "__main__":
    run("net_period")
