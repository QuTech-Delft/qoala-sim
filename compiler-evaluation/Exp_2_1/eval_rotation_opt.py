from __future__ import annotations

import datetime
import json
import math
import os
import time
import random
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List
from numpy import arange

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.runner import run_two_node_app_separate_inputs


def relative_to_cwd(file: str) -> str:
    """
    :param file: The file name
    :return: The path to the file relative to the current working directory
    """
    return os.path.join(os.path.dirname(__file__), file)


def create_procnode_cfg(
    name: str,
    id: int,
    num_qubits: int,
    t1: int,
    t2: int,
    single_gate_duration: int,
    single_gate_fid: float,
    qnos_instr_time: float,
) -> ProcNodeConfig:
    """
    Create the configuration object for a processing node

    :return: The processor node configuration object
    """
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        # TODO configuration for topology based on other params...
        topology=TopologyConfig.uniform_t1t2_qubits_uniform_single_gate_duration_and_noise(
            num_qubits,
            t1=t1,
            t2=t2,
            single_gate_duration=single_gate_duration,
            single_gate_fid=single_gate_fid,
        ),
        latencies=LatenciesConfig(qnos_instr_time=qnos_instr_time),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=True,
    )


def load_program(path: str) -> QoalaProgram:
    """
    Load a Qoala Program

    :param path: The path to the .iqoala file
    :return: Parsed Qoala program
    """
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class RotationExpResult:
    client_results: BatchResult
    server_results: BatchResult
    total_duration: float


def run_rotation_exp(
    num_iterations: int,
    theta0: float,
    theta1: float,
    theta2: float,
    theta3: float,
    theta4: float,
    theta5: float,
    theta6: float,
    theta7: float,
    naive: bool,
    t1: int,
    t2: int,
    cc: float,
    single_gate_fid: float,
    single_gate_duration: float,
    qnos_instr_time: float,
) -> RotationExpResult:
    ns.sim_reset()

    client_id = 1
    server_id = 0

    # Create the configuration for the server and client
    client_node_cfg = create_procnode_cfg(
        "client",
        client_id,
        1,
        t1,
        t2,
        single_gate_duration,
        single_gate_fid,
        qnos_instr_time,
    )
    server_node_cfg = create_procnode_cfg(
        "server",
        server_id,
        1,
        t1,
        t2,
        single_gate_duration,
        single_gate_fid,
        qnos_instr_time,
    )

    # Configure the network
    cconn = ClassicalConnectionConfig.from_nodes(client_id, server_id, cc)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[client_node_cfg, server_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    # Load the program onto the client and server
    if naive:
        client_program = load_program("rotation_naive_client.iqoala")
        server_program = load_program("rotation_naive_server.iqoala")
    else:
        # Naive client is used so that only the server program is varied.
        client_program = load_program("rotation_naive_client.iqoala")
        server_program = load_program("rotation_opt_server.iqoala")

    theta0_int = int(theta0 * 16 / math.pi)
    theta1_int = int(theta1 * 16 / math.pi)
    theta2_int = int(theta2 * 16 / math.pi)
    theta3_int = int(theta3 * 16 / math.pi)
    theta4_int = int(theta4 * 16 / math.pi)
    theta5_int = int(theta5 * 16 / math.pi)
    theta6_int = int(theta6 * 16 / math.pi)
    theta7_int = int(theta7 * 16 / math.pi)

    # Input parameters for client program
    client_inputs = [
        ProgramInput(
            {
                "server_id": server_id,
                "theta0": theta0_int,
                "theta1": theta1_int,
                "theta2": theta2_int,
                "theta3": theta3_int,
                "theta4": theta4_int,
                "theta5": theta5_int,
                "theta6": theta6_int,
                "theta7": theta7_int,
            }
        )
        for _ in range(num_iterations)
    ]

    # Input parameters for server program
    # Randomly vary the initial state for each program instance
    random.seed(0)
    server_inputs = [
        ProgramInput({"client_id": client_id, "state": random.randint(0, 5)})
        for _ in range(num_iterations)
    ]

    # Run the applications
    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"client": client_program, "server": server_program},
        program_inputs={"client": client_inputs, "server": server_inputs},
        network_cfg=network_cfg,
        linear=True,
    )

    # Get the results
    client_result = app_result.batch_results["client"]
    server_result = app_result.batch_results["server"]

    return RotationExpResult(client_result, server_result, app_result.total_duration)


@dataclass
class DataPoint:
    naive: bool
    succ_prob: float
    makespan: float
    param_name: str
    param_value: float
    succ_std_dev: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    theta0: float
    theta1: float
    theta2: float
    theta3: float
    theta4: float
    theta5: float
    theta6: float
    theta7: float
    t1: float
    t2: float
    cc: float
    single_gate_fid: float
    single_gate_duration: float
    qnos_instr_time: float
    sim_duration: float
    param_name: str


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def rotation_exp(
    num_iterations: int,
    theta0: float,
    theta1: float,
    theta2: float,
    theta3: float,
    theta4: float,
    theta5: float,
    theta6: float,
    theta7: float,
    naive: bool,
    t1: int,
    t2: int,
    cc: float,
    single_gate_fid: float,
    single_gate_duration: float,
    qnos_instr_time: float,
) -> float:
    result = run_rotation_exp(
        num_iterations,
        theta0,
        theta1,
        theta2,
        theta3,
        theta4,
        theta5,
        theta6,
        theta7,
        naive,
        t1,
        t2,
        cc,
        single_gate_fid,
        single_gate_duration,
        qnos_instr_time,
    )
    program_results = result.client_results.results

    results = [result.values["result"] for result in program_results]
    avg_makespan = result.total_duration / num_iterations

    # A success is when the program measures a 0
    # (the measurement outcome is the same as the initial state)
    successes = 0
    for res in results:
        if res == 0:
            successes += 1
    avg_succ_prob = round(successes / num_iterations, 3)

    # population STD Dev
    # succ_std_dev = math.sqrt(sum([((1 if res == 0 else 0) - avg_succ_prob)**2 for res in results]) / (num_iterations))

    # batched population STD Dev
    n_batches = 10
    batch_size = int(num_iterations / n_batches)
    batches = [
        [results[j] for j in range(i * batch_size, (i + 1) * batch_size)]
        for i in range(0, n_batches)
    ]
    batches_avgs = [(batch_size - sum(batch)) / len(batch) for batch in batches]
    succ_batch_std = math.sqrt(
        sum([(batch_avg - avg_succ_prob) ** 2 for batch_avg in batches_avgs])
        / (n_batches)
    )

    print(
        f"succ prob: {avg_succ_prob}, std dev:{succ_batch_std}, makespan: {avg_makespan}"
    )
    return avg_succ_prob, succ_batch_std, avg_makespan


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_iterations", "-n", type=int, required=True)
    parser.add_argument("--sweepRange", type=str, nargs="+", required=False)
    parser.add_argument("--sweepList", type=str, nargs="+", required=False)

    args = parser.parse_args()
    num_iterations = args.num_iterations

    sweep_range = args.sweepRange
    sweep_list = args.sweepList

    param_name = ""
    param_vals = []
    if sweep_range is not None and len(sweep_range) == 4:
        param_name = sweep_range[0]
        lower_val = float(sweep_range[1])
        upper_val = float(sweep_range[2])
        stepsize = float(sweep_range[3])
        param_vals = [val for val in arange(lower_val, upper_val, stepsize)]
    elif sweep_list is not None and len(sweep_list) > 1:
        param_name = sweep_list[0]
        param_vals = [float(val) for val in sweep_list[1:]]
    else:
        print(
            "Error: either the --sweepRange or the --sweepList argument must be used."
        )
        exit()

    # Memory
    t1 = 1e9  # 1 second
    t2 = 1e7  # 1e7 10ms

    # Gate Noise
    single_gate_fid = 1  # perfect gates

    # Gate execution time
    single_gate_duration = 5e3  # 5 micro seconds

    # Time to process a quant instruction
    qnos_instr_time = 50e3  # 50 micro seconds

    # Classical Communication latency
    cc = 0  # 0 seconds

    # Convert thetas
    theta0 = math.pi / 4
    theta1 = math.pi / 4
    theta2 = math.pi / 4
    theta3 = math.pi / 4
    theta4 = math.pi / 4
    theta5 = math.pi / 4
    theta6 = math.pi / 4
    theta7 = math.pi / 4

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()
    data_points: List[DataPoint] = []

    for param_val in param_vals:
        if param_name == "q_mem":
            t2 = param_val
        elif param_name == "g_fid":
            single_gate_fid = param_val
        elif param_name == "g_dur":
            single_gate_duration = param_val
        elif param_name == "instr_time":
            qnos_instr_time = param_val
        elif param_name == "cc_dur":
            cc = param_val
        else:
            print(
                "Error. Param name should be one of the five: q_mem, g_fid, g_dur, instr_time, cc_dur"
            )
            exit()

        # Run the naive program and get results
        succ_prob_naive, succ_std_dev_naive, makespan_naive = rotation_exp(
            naive=True,
            num_iterations=num_iterations,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
            theta5=theta5,
            theta6=theta6,
            theta7=theta7,
            t1=t1,
            t2=t2,
            cc=cc,
            single_gate_fid=single_gate_fid,
            single_gate_duration=single_gate_duration,
            qnos_instr_time=qnos_instr_time,
        )
        # Store the naive datapoint
        data_points.append(
            DataPoint(
                naive=True,
                succ_prob=succ_prob_naive,
                makespan=makespan_naive,
                param_name=param_name,
                param_value=param_val,
                succ_std_dev=succ_std_dev_naive,
            )
        )

        # Run the optimal program and get results
        succ_prob_opt, succ_std_dev_opt, makespan_opt = rotation_exp(
            naive=False,
            num_iterations=num_iterations,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
            theta5=theta5,
            theta6=theta6,
            theta7=theta7,
            t1=t1,
            t2=t2,
            cc=cc,
            single_gate_fid=single_gate_fid,
            single_gate_duration=single_gate_duration,
            qnos_instr_time=qnos_instr_time,
        )
        # Store the optimal datapoint
        data_points.append(
            DataPoint(
                naive=False,
                succ_prob=succ_prob_opt,
                makespan=makespan_opt,
                param_name=param_name,
                param_value=param_val,
                succ_std_dev=succ_std_dev_opt,
            )
        )

    # Finish computing how long the experiment took to run
    end = time.time()
    duration = round(end - start, 2)

    # compute the path to the directory for storing data
    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}_{param_name}.json")

    # Store the metadata about the experiment
    meta = DataMeta(
        timestamp=timestamp,
        num_iterations=num_iterations,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        theta4=theta4,
        theta5=theta5,
        theta6=theta6,
        theta7=theta7,
        t1=t1,
        t2=t2,
        cc=cc,
        single_gate_fid=single_gate_fid,
        single_gate_duration=single_gate_duration,
        qnos_instr_time=qnos_instr_time,
        sim_duration=duration,
        param_name=param_name,
    )

    # Format the metadata and datapoints into a json object
    data = Data(meta=meta, data_points=data_points)
    json_data = asdict(data)

    # Write the data
    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
