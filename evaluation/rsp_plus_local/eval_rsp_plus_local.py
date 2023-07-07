from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import netsquid as ns
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
from qoala.util.logging import LogManager


def create_procnode_cfg(
    name: str, id: int, t1: float, t2: float, determ: bool
) -> ProcNodeConfig:
    nv_params = NvParams()
    nv_params.comm_t1 = t1
    nv_params.comm_t2 = t2
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


def run_rsp(
    num_iterations: int,
    busy_iterations: int,
    angles: List[int],
    t1: float,
    t2: float,
    cc_latency: float,
    busy_duration: float,
    determ_sched: bool,
) -> RspResult:
    ns.sim_reset()

    num_qubits = 3
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, t1, t2, determ=determ_sched)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, t1, t2, determ=determ_sched)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc_latency)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("rsp_alice.iqoala")
    bob_program = load_program("rsp_bob.iqoala")
    busy_program = load_program("busy_classical.iqoala")

    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "angle": angle}) for angle in angles
    ]
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    app_result = run_two_node_app_separate_inputs_plus_3rd_program(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_inputs, "bob": bob_inputs},
        third_program=busy_program,
        third_program_input=ProgramInput({"duration": busy_duration}),
        third_program_num_iterations=busy_iterations,
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return RspResult(alice_result, bob_result)


def test_rsp():
    # LogManager.set_log_level("DEBUG")
    LogManager.enable_task_logger(True)
    LogManager.log_tasks_to_file("rsp_plus_local.log")
    num_iterations = 50
    busy_iterations = 100

    t1 = 1e6
    t2 = 1e6

    angles = [16 for _ in range(num_iterations)]
    cc = 1e1
    determ = False

    # for busy in [1e9, 1e10, 1e11, 1e12]:
    for busy in [1e12]:
        result = run_rsp(
            num_iterations,
            busy_iterations,
            angles,
            t1,
            t2,
            cc,
            busy,
            determ_sched=determ,
        )
        program_results = result.bob_results.results
        outcomes = [result.values["outcome"] for result in program_results]
        succ_prob = (len([x for x in outcomes if x == 1])) / len(outcomes)
        print(f"cc = {cc}: {outcomes}")
        print(f"cc = {cc}: succ prob: {succ_prob}")


if __name__ == "__main__":
    test_rsp()
