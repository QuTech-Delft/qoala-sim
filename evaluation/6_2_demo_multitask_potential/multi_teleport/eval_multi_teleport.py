from __future__ import annotations

import datetime
import json
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from math import ceil
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
from qoala.runtime.program import ProgramInput
from qoala.util.runner import AppResult, run_two_node_app


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, determ: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def run_teleport(
    num_iterations: int,
    linear: bool,
    num_qubits_alice: int,
    num_qubits_bob: int,
    cc: float,
) -> AppResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice", alice_id, num_qubits_alice, determ=True
    )
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits_bob, determ=True)
    bob_node_cfg.topology.qubits[0].qubit_config.noise_config.to_error_model_kwargs()[
        "T2"
    ] = 1

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("teleport_alice.iqoala")
    bob_program = load_program("teleport_bob.iqoala")

    # state = 5 -> teleport |1> state
    alice_input = ProgramInput({"bob_id": bob_id, "state": 5})
    # state = 4 -> measure in +Z, i.e. expect a "1" outcome
    bob_input = ProgramInput({"alice_id": alice_id, "state": 4})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
        linear_for={"alice": True, "bob": linear},
    )
    # print(f"makespan: {app_result.total_duration:_}")

    return app_result


def get_teleport_makespan(
    num_iterations: int,
    linear: bool,
    num_qubits_alice: int,
    num_qubits_bob: int,
    cc: float,
) -> float:
    result = run_teleport(
        num_iterations=num_iterations,
        linear=linear,
        num_qubits_alice=num_qubits_alice,
        num_qubits_bob=num_qubits_bob,
        cc=cc,
    )

    bob_result = result.batch_results["bob"]

    program_results = bob_result.results
    outcomes = [result.values["outcome"] for result in program_results]
    # print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)

    return result.total_duration


@dataclass
class DataPoint:
    linear: bool
    num_qubits_bob: int
    makespan: float
    approx_seq: float
    approx_int: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    num_qubits_alice: int
    cc: float
    sim_duration: float


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_iterations", "-n", type=int, required=True)

    args = parser.parse_args()
    num_iterations = args.num_iterations

    num_qubits_alice = 2
    cc = 1e7

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()
    data_points: List[DataPoint] = []

    for linear in [False, True]:
        for num_qubits_bob in range(1, 6):
            print(f"{linear}, {num_qubits_alice}, {num_qubits_bob}")
            makespan = get_teleport_makespan(
                num_iterations, linear, num_qubits_alice, num_qubits_bob, cc
            )
            approx_seq = num_iterations * cc
            approx_int = ceil(num_iterations / num_qubits_bob) * cc
            point = DataPoint(linear, num_qubits_bob, makespan, approx_seq, approx_int)
            data_points.append(point)

    end = time.time()
    duration = round(end - start, 2)

    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    meta = DataMeta(timestamp, num_iterations, num_qubits_alice, cc, duration)
    data = Data(meta=meta, data_points=data_points)
    json_data = asdict(data)

    # for p in data:
    #     print(f"{p.linear}, {p.num_qubits_alice}, {p.num_qubits_bob}, {p.makespan:_}")

    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
