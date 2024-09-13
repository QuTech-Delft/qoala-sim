from __future__ import annotations

import os
from typing import List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.ehi import UnitModule
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
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.sim.build import build_network_from_config
from qoala.sim.network import ProcNodeNetwork


def get_config() -> ProcNodeConfig:
    params = NvParams()
    topology = TopologyConfig.from_nv_params(num_qubits=5, params=params)
    return ProcNodeConfig(
        node_name="alice",
        node_id=0,
        topology=topology,
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
    )


def create_network(
    node_cfg: ProcNodeConfig,
) -> ProcNodeNetwork:
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    return build_network_from_config(network_cfg)


def load_program(name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text, flavour=NVFlavour()).parse()

    return program


def create_batch(
    program: QoalaProgram,
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
    deadline: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=deadline,
    )


def run(path: str) -> BatchResult:
    ns.sim_reset()

    node_config = get_config()
    network = create_network(node_config)
    procnode = network.nodes["alice"]

    num_iterations = 100
    inputs = [ProgramInput({}) for i in range(num_iterations)]

    unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())

    program = load_program(path)
    batch_info = create_batch(
        program=program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )

    procnode.submit_batch(batch_info)
    procnode.initialize_processes(linear=True)

    network.start_all_nodes()
    ns.sim_run()

    all_results = procnode.scheduler.get_batch_results()
    return all_results[0]


def test_1node_1qubit():
    batch_result = run("nv_1node_1qubit.iqoala")
    results = [result.values["m"] for result in batch_result.results]
    assert all(r == 1 for r in results)


def test_1node_2qubits_sg():
    batch_result = run("nv_1node_2qubits_sg.iqoala")
    results = [result.values["m"] for result in batch_result.results]
    assert all(r == [1, 1] for r in results)


def test_1node_2qubits_mg():
    batch_result = run("nv_1node_2qubits_mg.iqoala")
    results = [result.values["m"] for result in batch_result.results]
    assert all(r == [0, 0] for r in results)


def test_1node_5qubits():
    batch_result = run("nv_1node_5qubits.iqoala")
    results = [result.values["m"] for result in batch_result.results]
    assert all(r == [0, 0, 1, 0, 0] for r in results)


if __name__ == "__main__":
    # LogManager.set_log_level("DEBUG")
    test_1node_1qubit()
    test_1node_2qubits_sg()
    test_1node_2qubits_mg()
    test_1node_5qubits()
