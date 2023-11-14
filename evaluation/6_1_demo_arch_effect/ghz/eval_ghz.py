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
from qoala.util.runner import AppResult, create_batch


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
class GhzResult:
    alice_results: BatchResult
    bob_results: BatchResult
    charlie_results: BatchResult


def run_ghz(hardware: str, num_iterations: int) -> GhzResult:
    ns.sim_reset()

    alice_id = 0
    bob_id = 1
    charlie_id = 2

    alice_node_cfg = create_procnode_cfg("alice", alice_id, 1, hardware)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, 2, hardware)
    charlie_node_cfg = create_procnode_cfg("charlie", charlie_id, 2, hardware)

    cconn_ab = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e6)
    cconn_ac = ClassicalConnectionConfig.from_nodes(alice_id, charlie_id, 1e6)
    cconn_bc = ClassicalConnectionConfig.from_nodes(bob_id, charlie_id, 1e6)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg, charlie_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn_ab, cconn_ac, cconn_bc]

    alice_file = f"ghz_{hardware}_alice.iqoala"
    bob_file = f"ghz_{hardware}_bob.iqoala"
    charlie_file = f"ghz_{hardware}_charlie.iqoala"
    alice_program = load_program(alice_file, hardware)
    bob_program = load_program(bob_file, hardware)
    charlie_program = load_program(charlie_file, hardware)

    alice_input = [
        ProgramInput({"bob_id": bob_id, "charlie_id": charlie_id})
        for _ in range(num_iterations)
    ]
    bob_input = [
        ProgramInput({"alice_id": alice_id, "charlie_id": charlie_id})
        for _ in range(num_iterations)
    ]
    charlie_input = [
        ProgramInput({"alice_id": alice_id, "bob_id": bob_id})
        for _ in range(num_iterations)
    ]

    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)
    names = ["alice", "bob", "charlie"]
    programs = {"alice": alice_program, "bob": bob_program, "charlie": charlie_program}
    program_inputs = {"alice": alice_input, "bob": bob_input, "charlie": charlie_input}
    batches: Dict[str, ProgramBatch] = {}  # node -> batch

    for name in names:
        procnode = network.nodes[name]
        program = programs[name]
        inputs = program_inputs[name]

        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch_info = create_batch(program, unit_module, inputs, num_iterations)
        batches[name] = procnode.submit_batch(batch_info)

    for batch in batches.values():
        assert batch.batch_id == 0
        assert [p.pid for p in batch.instances] == [i for i in range(num_iterations)]

    for name in names:
        procnode = network.nodes[name]
        remote_pids = {0: [i for i in range(num_iterations)]}
        procnode.initialize_processes(remote_pids=remote_pids, linear=True)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in names:
        procnode = network.nodes[name]
        # only one batch (ID = 0), so get value at index 0
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    total_duration = ns.sim_time()
    app_result = AppResult(results, statistics, total_duration)
    print(f"{app_result.total_duration:_}")

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]
    charlie_result = app_result.batch_results["charlie"]

    return GhzResult(alice_result, bob_result, charlie_result)


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


def test_ghz(hardware: str, num_iterations: int) -> None:
    result = run_ghz(hardware, num_iterations)

    alice_results = result.alice_results.results
    alice_outcomes = [result.values["outcome"] for result in alice_results]
    # print(alice_outcomes)

    bob_results = result.bob_results.results
    bob_outcomes = [result.values["outcome"] for result in bob_results]
    # print(bob_outcomes)

    charlie_results = result.charlie_results.results
    charlie_outcomes = [result.values["outcome"] for result in charlie_results]
    # print(charlie_outcomes)

    assert alice_outcomes == bob_outcomes == charlie_outcomes


def run_hardware(hardware: str, num_iterations: int) -> DataPoint:
    start = time.time()

    success: bool
    try:
        test_ghz(hardware, num_iterations)
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
