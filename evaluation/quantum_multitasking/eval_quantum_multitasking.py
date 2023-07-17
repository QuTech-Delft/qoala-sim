from __future__ import annotations

import json
import math
import os
import time
from audioop import avg
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List

import netsquid as ns
from matplotlib import use

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import run_1_server_n_clients


def create_procnode_cfg(
    name: str, id: int, t1: float, t2: float, determ: bool, deadlines: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
            5, t1, t2
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
    num_clients: int,
    t1: float,
    t2: float,
    cc_latency: float,
) -> TeleportResult:
    ns.sim_reset()

    bob_id = 0

    alice_node_cfgs = [
        create_procnode_cfg(f"alice_{i}", i, t1, t2, determ=True, deadlines=True)
        for i in range(1, num_clients + 1)
    ]
    bob_node_cfg = create_procnode_cfg(
        "bob", bob_id, t1, t2, determ=True, deadlines=True
    )

    cconns = [
        ClassicalConnectionConfig.from_nodes(i, bob_id, cc_latency)
        for i in range(1, num_clients + 1)
    ]
    node_cfgs = alice_node_cfgs + [bob_node_cfg]
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )
    network_cfg.cconns = cconns

    alice_program = load_program("programs/teleport_alice.iqoala")
    bob_program = load_program("programs/teleport_bob.iqoala")

    states = [i % 6 for i in range(num_iterations)]

    alice_inputs = {
        f"alice_{i}": [ProgramInput({"bob_id": 0, "state": states[i]})]
        for i in range(1, num_clients + 1)
    }
    bob_inputs = [
        ProgramInput({"alice_id": i, "state": states[i]})
        for i in range(1, num_clients + 1)
    ]

    app_result = run_1_server_n_clients(
        client_names=[f"alice_{i}" for i in range(1, num_clients + 1)],
        client_program=alice_program,
        client_inputs=alice_inputs,
        server_name="bob",
        server_program=bob_program,
        server_inputs=bob_inputs,
        network_cfg=network_cfg,
    )

    alice_results = [
        app_result.batch_results[f"alice_{i}"] for i in range(1, num_clients + 1)
    ]
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
    num_clients: int,
    t1: int,
    t2: int,
    latency_factor: float,
) -> DataPoint:
    successes: List[bool] = []
    makespans: List[float] = []

    cc_latency = latency_factor * t2

    for i in range(num_iterations):
        result = run_apps(
            num_iterations=num_iterations,
            num_clients=num_clients,
            t1=t1,
            t2=t2,
            cc_latency=cc_latency,
        )
        program_results = result.bob_results.results
        outcomes = [result.values["outcome"] for result in program_results]
        assert len(outcomes) == 1
        successes.append(outcomes[0] == 0)
        makespans.append(result.total_duration)

    avg_succ_prob = sum([s for s in successes if s]) / len(successes)
    succ_prob_lower, succ_prob_upper = wilson_score_interval(
        p_hat=avg_succ_prob, n=len(successes), z=1.96
    )
    lower_rounded = round(succ_prob_lower, 3)
    upper_rounded = round(succ_prob_upper, 3)
    succprob_rounded = round(avg_succ_prob, 3)
    print(f"succ prob: {succprob_rounded} ({lower_rounded}, {upper_rounded})")

    total_makespan = sum(makespans)

    return DataPoint(
        t2=t2,
        cc_latency=cc_latency,
        succ_prob=avg_succ_prob,
        succ_prob_lower=succ_prob_lower,
        succ_prob_upper=succ_prob_upper,
        makespan=total_makespan,
    )


def run():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("quantum_multitasking.log")
    LogManager.enable_task_logger(True)
    LogManager.log_tasks_to_file("quantum_multitasking_tasks.log")

    start_time = time.time()

    t1 = 1e10
    t2 = 1e8

    num_clients = 1
    num_iterations = 1
    latency_factor = 0.1

    data_points: List[DataPoint] = []

    data_point = get_metrics(
        num_iterations=num_iterations,
        num_clients=num_clients,
        t1=t1,
        t2=t2,
        latency_factor=latency_factor,
    )
    data_points.append(data_points)

    end_time = time.time()
    print(f"total duration: {end_time - start_time}s")


if __name__ == "__main__":
    run()
