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
from qoala.util.runner import run_single_node_app_separate_inputs


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
        latencies=LatenciesConfig(qnos_instr_time=1000, host_instr_time=1000),
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
    t1: float,
    t2: float,
    busy_duration: float,
    determ_sched: bool,
) -> RspResult:
    ns.sim_reset()

    bob_id = 0

    bob_node_cfg = create_procnode_cfg("bob", bob_id, t1, t2, determ=determ_sched)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[bob_node_cfg], link_duration=1000
    )

    bob_program = load_program("local_quantum.iqoala")

    bob_inputs = [
        ProgramInput({"duration": busy_duration}) for _ in range(num_iterations)
    ]

    app_result = run_single_node_app_separate_inputs(
        num_iterations=num_iterations,
        program_name="bob",
        program=bob_program,
        program_input=bob_inputs,
        network_cfg=network_cfg,
        linear=True,
    )

    bob_result = app_result.batch_results["bob"]

    bob_stats = app_result.statistics["bob"]
    # cpu_hog_tasks = [t for t in bob_stats._cpu_tasks_executed.values() if t.pid == 1]
    # for task in cpu_hog_tasks:
    #     start = bob_stats._cpu_task_starts[task.task_id]
    #     print(f"{start} : {task}")

    qpu_tasks = [t for t in bob_stats._qpu_tasks_executed.values() if t.pid == 0]
    qpu_starts = [bob_stats._qpu_task_starts[task.task_id] for task in qpu_tasks]
    qpu_ends = [bob_stats._qpu_task_ends[task.task_id] for task in qpu_tasks]
    assert len(qpu_starts) == len(qpu_ends)
    # print(qpu_starts)
    # print(qpu_ends)
    qpu_waits = [qpu_starts[i + 1] - qpu_ends[i] for i in range(len(qpu_starts) - 1)]
    print(qpu_waits)

    return RspResult(None, bob_result)


def test_rsp():
    # LogManager.set_log_level("DEBUG")
    # LogManager.enable_task_logger(True)
    # LogManager.log_tasks_to_file("classical_multitasking.log")
    num_iterations = 1

    t1 = 1e7
    t2 = 1e3

    determ = False

    # for busy in [1e6, 2e6, 3e6, 5e6, 1e7, 5e7, 1e8]:
    for busy in [1e3]:
        succ_probs: List[float] = []
        for _ in range(100):
            result = run_rsp(num_iterations, t1, t2, busy, determ_sched=determ)
            program_results = result.bob_results.results
            outcomes = [result.values["outcome"] for result in program_results]
            succ_prob = (len([x for x in outcomes if x == 1])) / len(outcomes)
            # print(f"busy = {busy}, cc = {cc}: succ prob: {succ_prob}")
            succ_probs.append(succ_prob)
        # print(f"cc = {cc}: {outcomes}")
        avg_succ = sum(succ_probs) / len(succ_probs)
        print(f"busy = {busy}: avg succ: {avg_succ}")


if __name__ == "__main__":
    test_rsp()
