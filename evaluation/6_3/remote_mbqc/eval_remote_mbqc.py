from __future__ import annotations

import math
import os
from dataclasses import dataclass

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

    client_node_cfg = create_procnode_cfg("client", client_id, t1, t2)
    server_node_cfg = create_procnode_cfg("server", server_id, t1, t2)

    cconn = ClassicalConnectionConfig.from_nodes(client_id, server_id, cc)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[client_node_cfg, server_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    if naive:
        client_program = load_program("remote_mbqc_naive_client.iqoala")
        server_program = load_program("remote_mbqc_naive_server.iqoala")
    else:
        client_program = load_program("remote_mbqc_opt_client.iqoala")
        server_program = load_program("remote_mbqc_opt_server.iqoala")

    theta0_int = int(theta0 * 16 / math.pi)
    theta1_int = int(theta1 * 16 / math.pi)
    theta2_int = int(theta2 * 16 / math.pi)
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
    server_inputs = [
        ProgramInput({"client_id": client_id}) for _ in range(num_iterations)
    ]

    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"client": client_program, "server": server_program},
        program_inputs={"client": client_inputs, "server": server_inputs},
        network_cfg=network_cfg,
        linear=True,
    )

    client_result = app_result.batch_results["client"]
    server_result = app_result.batch_results["server"]

    return RemoteMbqcResult(client_result, server_result)


def remote_mbqc(naive: bool):
    t1 = 1e10
    t2 = 1e6
    cc = 1e5

    num_iterations = 1000

    theta0 = math.pi / 2
    theta1 = 0
    theta2 = math.pi / 2
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
    print(f"succ prob: {round(successes / num_iterations, 2)}")


if __name__ == "__main__":
    remote_mbqc(naive=True)
    remote_mbqc(naive=False)
