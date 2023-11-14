from __future__ import annotations

import datetime
import json
import os
import random
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour, TrappedIonFlavour

from qoala.lang.ehi import UnitModule
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
from qoala.runtime.program import BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.sim.build import build_network_from_config
from qoala.util.runner import (
    AppResult,
    create_batch,
    run_two_node_app,
    run_two_node_app_separate_inputs,
)


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
            topology=TopologyConfig.perfect_nv_default_params(num_qubits=num_qubits),
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
class TeleportResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_teleport(
    hardware: str, num_iterations: int, different_inputs: bool = False
) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 4
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, hardware)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, hardware)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e9)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_file = f"teleport_{hardware}_alice.iqoala"
    bob_file = f"teleport_{hardware}_bob.iqoala"
    alice_program = load_program(alice_file, hardware)
    bob_program = load_program(bob_file, hardware)

    if different_inputs:
        alice_inputs: List[ProgramInput] = []
        bob_inputs: List[ProgramInput] = []
        for _ in range(num_iterations):
            basis = random.randint(0, 5)
            alice_inputs.append(ProgramInput({"bob_id": bob_id, "state": basis}))
            bob_inputs.append(ProgramInput({"alice_id": alice_id, "state": basis}))

        app_result = run_two_node_app_separate_inputs(
            num_iterations=num_iterations,
            programs={"alice": alice_program, "bob": bob_program},
            program_inputs={"alice": alice_inputs, "bob": bob_inputs},
            network_cfg=network_cfg,
        )
    else:
        # state = 5 -> teleport |1> state
        alice_input = ProgramInput({"bob_id": bob_id, "state": 5})
        # state = 4 -> measure in +Z, i.e. expect a "1" outcome
        bob_input = ProgramInput({"alice_id": alice_id, "state": 4})

        app_result = run_two_node_app(
            num_iterations=num_iterations,
            programs={"alice": alice_program, "bob": bob_program},
            program_inputs={"alice": alice_input, "bob": bob_input},
            network_cfg=network_cfg,
        )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return TeleportResult(alice_result, bob_result)


def teleport_different_inputs(hardware: str, num_iterations: int):
    result = run_teleport(hardware, num_iterations, different_inputs=True)

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    # print(outcomes)
    assert all(outcome == 0 for outcome in outcomes)


@dataclass
class DataPoint:
    success: bool
    sim_duration: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def run_hardware(hardware: str, num_iterations: int) -> DataPoint:
    start = time.time()

    success: bool
    try:
        teleport_different_inputs(hardware, num_iterations)
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

    args = parser.parse_args()
    num_iterations = args.num_iterations

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_points: List[DataPoint] = []
    for hardware in ["generic", "nv", "tri"]:
        data_point = run_hardware(hardware, num_iterations)
        data_points.append(data_point)

    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")

    meta = DataMeta(timestamp=timestamp, num_iterations=num_iterations)
    data = Data(meta=meta, data_points=data_points)
    json_data = asdict(data)

    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)
