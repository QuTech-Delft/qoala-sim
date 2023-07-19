from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
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
from qoala.util.runner import run_1_server_n_clients, run_two_node_app_separate_inputs


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
    alice_results: List[BatchResult]
    bob_results: BatchResult
    total_duration: float


def run_apps(
    num_iterations: int,
    t1: float,
    t2: float,
    cc_latency: float,
    network_bin_len: int,
    network_period: int,
) -> TeleportResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice", alice_id, t1, t2, determ=True, deadlines=True, num_qubits=6
    )
    bob_node_cfg = create_procnode_cfg(
        "bob", bob_id, t1, t2, determ=True, deadlines=True, num_qubits=3
    )

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc_latency)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=network_bin_len,
        first_bin=0,
        bin_pattern=pattern,
        repeat_period=network_period,
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("programs/teleport_alice.iqoala")
    bob_program = load_program("programs/teleport_bob.iqoala")

    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "state": i % 6}) for i in range(num_iterations)
    ]
    bob_inputs = [
        ProgramInput({"alice_id": alice_id, "state": i % 6})
        for i in range(num_iterations)
    ]

    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_inputs, "bob": bob_inputs},
        network_cfg=network_cfg,
        linear_for={"alice": True, "bob": False},
    )

    alice_results = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return TeleportResult(alice_results, bob_result, app_result.total_duration)


@dataclass
class DataPoint:
    t2: float
    cc_latency: float
    succ_prob: float
    succ_prob_lower: float
    succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    latency_factors: List[float]


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
    t1: int,
    t2: int,
    latency_factor: float,
    network_bin_len: int,
    network_period: int,
) -> DataPoint:
    successes: List[bool] = []

    cc_latency = latency_factor * t2

    result = run_apps(
        num_iterations=num_iterations,
        t1=t1,
        t2=t2,
        cc_latency=cc_latency,
        network_period=network_period,
        network_bin_len=network_bin_len,
    )
    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    assert len(outcomes) == num_iterations
    successes.extend([outcomes[i] == 0 for i in range(num_iterations)])

    avg_succ_prob = sum([s for s in successes if s]) / len(successes)
    succ_prob_lower, succ_prob_upper = wilson_score_interval(
        p_hat=avg_succ_prob, n=len(successes), z=1.96
    )
    lower_rounded = round(succ_prob_lower, 3)
    upper_rounded = round(succ_prob_upper, 3)
    succprob_rounded = round(avg_succ_prob, 3)

    makespan = result.total_duration

    print(f"succ prob: {succprob_rounded} ({lower_rounded}, {upper_rounded})")
    print(f"makespan: {makespan:_}")

    return DataPoint(
        t2=t2,
        cc_latency=cc_latency,
        succ_prob=avg_succ_prob,
        succ_prob_lower=succ_prob_lower,
        succ_prob_upper=succ_prob_upper,
        makespan=makespan,
    )


def run():
    LogManager.set_log_level("INFO")
    LogManager.log_to_file("quantum_multitasking.log")
    LogManager.enable_task_logger(True)
    LogManager.log_tasks_to_file("quantum_multitasking_tasks.log")

    start_time = time.time()

    t1 = 1e10
    t2 = 1e8

    num_iterations = 40
    latency_factor = 0.1
    net_period_factor = 10
    network_period = int(latency_factor * t2 * net_period_factor)
    network_bin_len = int(0.5 * network_period / num_iterations)
    # network_period = 200_000

    data_points: List[DataPoint] = []

    data_point = get_metrics(
        num_iterations=num_iterations,
        t1=t1,
        t2=t2,
        latency_factor=latency_factor,
        network_bin_len=network_bin_len,
        network_period=network_period,
    )
    data_points.append(data_points)

    end_time = time.time()
    # print(f"total duration: {end_time - start_time}s")

    print(f"cc latency: {latency_factor * t2:_}")
    print(f"network period: {network_period:_}")
    print(f"network bin len: {network_bin_len:_}")


if __name__ == "__main__":
    run()
