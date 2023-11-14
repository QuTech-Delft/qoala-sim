from __future__ import annotations

import datetime
import json
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour, TrappedIonFlavour

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.runner import run_two_node_app


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, hardware: str
) -> ProcNodeConfig:
    if hardware == "generic":
        return ProcNodeConfig(
            node_name=name,
            node_id=id,
            topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
            latencies=LatenciesConfig(qnos_instr_time=1000),
            ntf=NtfConfig.from_cls_name("GenericNtf"),
        )
    elif hardware == "nv":
        return ProcNodeConfig(
            node_name=name,
            node_id=id,
            topology=TopologyConfig.from_nv_params(
                num_qubits=num_qubits, params=NvParams()
            ),
            latencies=LatenciesConfig(qnos_instr_time=1000),
            ntf=NtfConfig.from_cls_name("NvNtf"),
        )
    elif hardware == "tri":
        return ProcNodeConfig(
            node_name=name,
            node_id=id,
            topology=TopologyConfig.perfect_tri_default_params(num_qubits=num_qubits),
            latencies=LatenciesConfig(qnos_instr_time=1000),
            ntf=NtfConfig.from_cls_name("TrappedIonNtf"),
        )


def load_program(path: str, hardware: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()

    if hardware == "generic":
        flavour = None
    elif hardware == "nv":
        flavour = NVFlavour()
    elif hardware == "tri":
        flavour = TrappedIonFlavour()
    return QoalaParser(text, flavour=flavour).parse()


@dataclass
class QkdResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_qkd(
    hardware: str,
    num_iterations: int,
    num_pairs: int,
    linear: bool = True,
):
    num_qubits = 3
    alice_id = 0
    bob_id = 1

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, hardware)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, hardware)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )

    alice_file = "qkd_npairs_MD_alice.iqoala"
    bob_file = "qkd_npairs_MD_bob.iqoala"
    alice_program = load_program(alice_file, hardware)
    bob_program = load_program(bob_file, hardware)

    alice_input = ProgramInput({"bob_id": bob_id, "N": num_pairs})
    bob_input = ProgramInput({"alice_id": alice_id, "N": num_pairs})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
        linear=linear,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return QkdResult(alice_result, bob_result)


def qkd_npairs_md(hardware: str, num_iterations: int, num_pairs: int):
    num_pairs = int(num_pairs)
    ns.sim_reset()

    qkd_result = run_qkd(hardware, num_iterations, num_pairs=num_pairs)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        # print(f"alice: {alice['outcomes']}")
        # print(f"bob  : {bob['outcomes']}")
        assert alice["outcomes"] == bob["outcomes"]


@dataclass
class DataPoint:
    success: bool
    sim_duration: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    num_pairs: int


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def run_hardware(hardware: str, num_iterations: int, num_pairs: int) -> DataPoint:
    start = time.time()

    success: bool
    try:
        qkd_npairs_md(hardware, num_iterations, num_pairs)
        success = True
    except AssertionError:
        success = False

    end = time.time()
    duration = round(end - start, 2)
    print(f"duration: {duration} s")

    return DataPoint(success, duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_iterations", "-n", type=int, required=True)
    parser.add_argument("--num_pairs", "-p", type=int, required=True)

    args = parser.parse_args()
    num_iterations = args.num_iterations
    num_pairs = args.num_pairs

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_points: List[DataPoint] = []
    for hardware in ["generic", "nv", "tri"]:
        data_point = run_hardware(hardware, num_iterations, num_pairs)
        data_points.append(data_point)

    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    meta = DataMeta(
        timestamp=timestamp, num_iterations=num_iterations, num_pairs=num_pairs
    )
    data = Data(meta=meta, data_points=data_points)
    json_data = asdict(data)

    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
