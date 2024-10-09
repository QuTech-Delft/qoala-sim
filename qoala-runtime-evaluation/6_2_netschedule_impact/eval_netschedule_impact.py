from __future__ import annotations

import json
import math
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import netsquid as ns
import numpy as np

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
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.runner import run_two_node_app_separate_inputs


def create_procnode_cfg(
    name: str,
    id: int,
    t1: float,
    t2: float,
    determ: bool,
    deadlines: bool,
    num_qubits: int,
    prio_epr: bool,
) -> ProcNodeConfig:
    topology = TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
        num_qubits, t1, t2
    )
    if name == "alice":
        latencies = LatenciesConfig(qnos_instr_time=1000, host_instr_time=1000)
    elif name == "bob":
        latencies = LatenciesConfig(qnos_instr_time=100_000, host_instr_time=1000)

    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=topology,
        latencies=latencies,
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
        use_deadlines=deadlines,
        prio_epr=prio_epr,
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
    total_duration: float


def run_apps(
    num_iterations: int,
    num_qubits_bob: int,
    t1: float,
    t2: float,
    cc_latency: float,
    network_bin_len: int,
    network_period: int,
    network_first_bin: int,
    prio_epr: bool,
) -> TeleportResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice",
        alice_id,
        t1,
        t2,
        determ=True,
        deadlines=True,
        num_qubits=20,
        prio_epr=prio_epr,
    )
    bob_node_cfg = create_procnode_cfg(
        "bob",
        bob_id,
        t1,
        t2,
        determ=True,
        deadlines=True,
        num_qubits=num_qubits_bob,
        prio_epr=prio_epr,
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
        linear=False,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]
    # print(local_result)

    return TeleportResult(alice_result, bob_result, app_result.total_duration)


@dataclass
class DataPoint:
    cc_latency: float
    num_qubits_bob: int
    tel_succ_prob: float
    tel_succ_prob_lower: float
    tel_succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    latency_factor: float
    net_bin_factors: List[float]
    prio_epr: bool
    num_qubits: int
    num_steps: int
    t1: float
    t2: float


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


def get_datapoint(
    num_runs: int,
    num_iterations: int,
    num_qubits_bob: int,
    t1: float,
    t2: float,
    latency_factor: float,
    network_bin_len: int,
    network_period: int,
    network_first_bin: int,
    prio_epr: bool,
) -> DataPoint:
    teleport_successes: List[bool] = []
    makespans: List[float] = []

    for _ in range(num_runs):
        cc_latency = latency_factor * t2

        result = run_apps(
            num_iterations=num_iterations,
            num_qubits_bob=num_qubits_bob,
            t1=t1,
            t2=t2,
            cc_latency=cc_latency,
            network_period=network_period,
            network_bin_len=network_bin_len,
            network_first_bin=network_first_bin,
            prio_epr=prio_epr,
        )
        teleport_results = result.bob_result.results
        outcomes = [result.values["outcome"] for result in teleport_results]
        assert len(outcomes) == num_iterations
        teleport_successes.extend([outcomes[i] == 0 for i in range(num_iterations)])

        makespans.append(result.total_duration)

    tel_avg_succ_prob = sum([s for s in teleport_successes if s]) / len(
        teleport_successes
    )
    tel_succ_prob_lower, tel_succ_prob_upper = wilson_score_interval(
        p_hat=tel_avg_succ_prob, n=len(teleport_successes), z=1.96
    )
    tel_lower_rounded = round(tel_succ_prob_lower, 3)
    tel_upper_rounded = round(tel_succ_prob_upper, 3)
    tel_succprob_rounded = round(tel_avg_succ_prob, 3)

    makespan = sum(makespans) / len(makespans)

    print(
        f"teleport succ prob: {tel_succprob_rounded} ({tel_lower_rounded}, {tel_upper_rounded})"
    )
    print(f"makespan: {makespan:_}")

    return DataPoint(
        cc_latency=cc_latency,
        num_qubits_bob=num_qubits_bob,
        tel_succ_prob=tel_avg_succ_prob,
        tel_succ_prob_lower=tel_succ_prob_lower,
        tel_succ_prob_upper=tel_succ_prob_upper,
        makespan=makespan,
    )


def run(output_dir: str, num_runs: int, num_qubits: int, num_steps: int, dump: bool):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t1 = 1e10
    t2 = 1e8
    prio_epr = False

    num_iterations = 5
    latency_factor = 0.1  # CC = 10_000_000
    net_bin_factors = list(np.linspace(0.2, 0.6, num_steps + 1))
    net_bin_factors = [round(f, 2) for f in net_bin_factors]
    network_gap = 0

    data_points: List[DataPoint] = []

    for bin_factor in net_bin_factors:
        network_bin_len = int((latency_factor * t2) * bin_factor)
        network_first_bin = int(network_bin_len / 2)
        network_period = (num_iterations + network_gap) * network_bin_len

        data_point = get_datapoint(
            num_runs=num_runs,
            num_iterations=num_iterations,
            num_qubits_bob=num_qubits,
            t1=t1,
            t2=t2,
            latency_factor=latency_factor,
            network_bin_len=network_bin_len,
            network_period=network_period,
            network_first_bin=network_first_bin,
            prio_epr=prio_epr,
        )
        makespan = data_point.makespan
        num_bins = math.ceil(makespan / network_bin_len)
        # print(f"total duration: {end_time - start_time}s")

        print(f"cc latency: {latency_factor * t2:_}")
        print(f"network period: {network_period:_}")
        print(f"network bin len: {network_bin_len:_}")
        print(f"makespan: {makespan}")
        print(f"num bins: {num_bins}")

        data_points.append(data_point)

    meta = DataMeta(
        timestamp=timestamp,
        num_iterations=num_iterations,
        latency_factor=latency_factor,
        net_bin_factors=net_bin_factors,
        prio_epr=prio_epr,
        num_qubits=num_qubits,
        num_steps=num_steps,
        t1=t1,
        t2=t2,
    )

    end_time = time.time()
    print(f"simulation took {end_time - start_time}s")

    if dump:
        data = Data(meta=meta, data_points=data_points)
        json_data = asdict(data)

        abs_dir = relative_to_cwd(f"data/{output_dir}")
        Path(abs_dir).mkdir(parents=True, exist_ok=True)
        last_path = os.path.join(abs_dir, "LAST.json")
        timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")
        with open(last_path, "w") as datafile:
            json.dump(json_data, datafile)
        with open(timestamp_path, "w") as datafile:
            json.dump(json_data, datafile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_qubits", "-q", type=int, required=True)
    parser.add_argument("--num_steps", "-s", type=int, required=True)
    parser.add_argument("--num_runs", "-n", type=int, required=True)
    parser.add_argument("--dump", "-d", action="store_true")

    args = parser.parse_args()

    dump = args.dump
    num_qubits = args.num_qubits
    num_steps = args.num_steps
    num_runs = args.num_runs

    run(
        f"sweep_bin_length_{num_qubits}_{num_steps}",
        num_runs=num_runs,
        num_qubits=num_qubits,
        num_steps=num_steps,
        dump=dump,
    )
