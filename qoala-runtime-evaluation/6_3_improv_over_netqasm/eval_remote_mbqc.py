from __future__ import annotations

import datetime
import json
import math
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

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
    return os.path.join(os.path.dirname(__file__), file)


# Create a node and configure it
def create_procnode_cfg(name: str, id: int, t1: int, t2: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
            3, t1, t2
        ),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=True,
    )


def load_program(path: str) -> QoalaProgram:
    """Load a Qoala Program"""
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class RemoteMbqcResult:
    client_results: BatchResult
    server_results: BatchResult


def run_remote_mbqc(
    num_iterations: int,
    theta0: float,
    theta1: float,
    theta2: float,
    naive: bool,
    t1: int,
    t2: int,
    cc: float,
) -> RemoteMbqcResult:
    ns.sim_reset()

    client_id = 1
    server_id = 0

    # Create the configuration for the server and client
    client_node_cfg = create_procnode_cfg("client", client_id, t1, t2)
    server_node_cfg = create_procnode_cfg("server", server_id, t1, t2)

    # Configure the network
    cconn = ClassicalConnectionConfig.from_nodes(client_id, server_id, cc)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[client_node_cfg, server_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    # Load the program onto the client and server
    if naive:
        client_program = load_program("remote_mbqc_naive_client.iqoala")
        server_program = load_program("remote_mbqc_naive_server.iqoala")
    else:
        client_program = load_program("remote_mbqc_opt_client.iqoala")
        server_program = load_program("remote_mbqc_opt_server.iqoala")

    theta0_int = int(theta0 * 16 / math.pi)
    theta1_int = int(theta1 * 16 / math.pi)
    theta2_int = int(theta2 * 16 / math.pi)

    # Input parameters for client program
    client_inputs = [
        ProgramInput(
            {
                "server_id": server_id,
                "theta0": theta0_int,
                "theta1": theta1_int,
                "theta2": theta2_int,
            }
        )
        for _ in range(num_iterations)
    ]

    # Input parameters for server program
    server_inputs = [
        ProgramInput({"client_id": client_id}) for _ in range(num_iterations)
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

    return RemoteMbqcResult(client_result, server_result)


@dataclass
class DataPoint:
    naive: bool
    succ_prob: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    theta0: float
    theta1: float
    theta2: float
    t1: float
    t2: float
    cc: float
    sim_duration: float


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]

# Runs the MBQC Experiment and returns the success probability
def remote_mbqc(
    naive: bool,
    num_iterations: int,
    t1: float,
    t2: float,
    cc: float,
    theta0: float,
    theta1: float,
    theta2: float,
) -> float:
    result = run_remote_mbqc(num_iterations, theta0, theta1, theta2, naive, t1, t2, cc)
    program_results = result.client_results.results
    # print(program_results)
    m0s = [result.values["m0"] for result in program_results]
    m1s = [result.values["m1"] for result in program_results]
    m2s = [result.values["m2"] for result in program_results]
    # print(m0s)
    # print(m1s)
    # print(m2s)

    successes = 0
    for m0, m1, m2 in zip(m0s, m1s, m2s):
        if m2 == (m0 == m1):
            successes += 1
    succ_prob = round(successes / num_iterations, 3)
    print(f"succ prob: {succ_prob}")
    return succ_prob


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_iterations", "-n", type=int, required=True)

    args = parser.parse_args()
    num_iterations = args.num_iterations

    t1 = 1e10
    t2 = 1e6
    cc = 1e5

    theta0 = math.pi / 2
    theta1 = 0
    theta2 = math.pi / 2

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()
    data_points: List[DataPoint] = []

    # Run the mbqc protocol using the naive iqoala program
    succ_prob_naive = remote_mbqc(
        naive=True,
        num_iterations=num_iterations,
        t1=t1,
        t2=t2,
        cc=cc,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
    )
    # Store the result as a data point
    data_points.append(DataPoint(naive=True, succ_prob=succ_prob_naive))

    # Run the protocol using the optimal iqoala program
    succ_prob_qoala = remote_mbqc(
        naive=False,
        num_iterations=num_iterations,
        t1=t1,
        t2=t2,
        cc=cc,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
    )

    data_points.append(DataPoint(naive=False, succ_prob=succ_prob_qoala))

    end = time.time()
    duration = round(end - start, 2)

    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    meta = DataMeta(
        timestamp=timestamp,
        num_iterations=num_iterations,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        t1=t1,
        t2=t2,
        cc=cc,
        sim_duration=duration,
    )
    data = Data(meta=meta, data_points=data_points)
    json_data = asdict(data)

    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
