from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour

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
from qoala.util.logging import LogManager
from qoala.util.runner import run_two_node_app_separate_inputs


def create_procnode_cfg(name: str, id: int, determ: bool) -> ProcNodeConfig:
    nv_params = NvParams()
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.from_nv_params(num_qubits=5, params=nv_params),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
        determ_sched=determ,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=NVFlavour()).parse()


@dataclass
class RspResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_rsp(num_iterations: int, angles: List[int]) -> RspResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, determ=True)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )

    alice_program = load_program("rsp_alice.iqoala")
    bob_program = load_program("rsp_bob.iqoala")

    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "angle": angle}) for angle in angles
    ]
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_inputs, "bob": bob_inputs},
        network_cfg=network_cfg,
        linear=True,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return RspResult(alice_result, bob_result)


def test_rsp():
    # LogManager.set_log_level("DEBUG")
    LogManager.enable_task_logger(True)
    # LogManager.log_tasks_to_file("teleport_plus_local.log")
    num_iterations = 20

    angles = [16 for _ in range(num_iterations)]

    result = run_rsp(num_iterations, angles)
    program_results = result.bob_results.results
    print(program_results)
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)


if __name__ == "__main__":
    test_rsp()
