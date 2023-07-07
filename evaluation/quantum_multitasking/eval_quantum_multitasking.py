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
from netqasm.lang.instr.flavour import NVFlavour

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
from qoala.runtime.task import TaskExecutionMode
from qoala.util.logging import LogManager
from qoala.util.runner import (
    run_1_server_n_clients,
    run_two_node_app_separate_inputs,
    run_two_node_app_separate_inputs_plus_3rd_program,
)


def create_procnode_cfg(
    name: str, id: int, t1: float, t2: float, determ: bool, deadlines: bool
) -> ProcNodeConfig:
    nv_params = NvParams()
    nv_params.comm_t1 = t1
    nv_params.comm_t2 = t2
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.from_nv_params(num_qubits=5, params=nv_params),
        latencies=LatenciesConfig(qnos_instr_time=1000, host_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
        tem=TaskExecutionMode.QOALA.name,
        determ_sched=determ,
        use_deadlines=deadlines,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=NVFlavour()).parse()


@dataclass
class TeleportResult:
    alice_results: List[BatchResult]
    bob_results: BatchResult
    total_duration: float


class TeleportBasis(Enum):
    X0 = 0
    X1 = 1
    Y0 = 2
    Y1 = 3
    Z0 = 4
    Z1 = 5


def run_teleport(
    num_clients: int,
    alice_bases: List[TeleportBasis],
    bob_bases: List[TeleportBasis],
    t1: float,
    t2: float,
    cc_latency: float,
    determ_sched: bool,
    deadlines: bool,
) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 3
    alice_id = 1
    bob_id = 0

    alice_node_cfgs = [
        create_procnode_cfg(
            f"alice_{i}", i, t1, t2, determ=determ_sched, deadlines=deadlines
        )
        for i in range(1, num_clients + 1)
    ]
    bob_node_cfg = create_procnode_cfg(
        "bob", bob_id, t1, t2, determ=determ_sched, deadlines=deadlines
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

    alice_program = load_program("teleport_nv_alice.iqoala")
    bob_program = load_program("teleport_nv_bob.iqoala")

    alice_inputs = {
        f"alice_{i}": [ProgramInput({"bob_id": 0, "state": alice_bases[i - 1].value})]
        for i in range(1, num_clients + 1)
    }
    bob_inputs = [
        ProgramInput({"alice_id": i, "state": bob_bases[i - 1].value})
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
    use_deadlines: bool
    latency_factor: float
    busy_factor: float
    hog_prob: float
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


def test_rsp():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("quantum_multitasking.log")
    LogManager.enable_task_logger(True)
    LogManager.log_tasks_to_file("quantum_multitasking_tasks.log")

    start_time = time.time()

    t1 = 1e10
    t2 = 1e8

    determ = False
    use_deadlines = False
    cc = 1e8

    num_clients = 2

    data_points: List[DataPoint] = []

    print(f"use deadlines: {use_deadlines}")

    alice_bases = [TeleportBasis.Y0 for _ in range(num_clients)]
    bob_bases = [TeleportBasis.Y1 for _ in range(num_clients)]
    result = run_teleport(
        num_clients=num_clients,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        t1=t1,
        t2=t2,
        cc_latency=cc,
        determ_sched=determ,
        deadlines=use_deadlines,
    )
    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]

    end_time = time.time()
    print(f"total duration: {end_time - start_time}s")


if __name__ == "__main__":
    test_rsp()
