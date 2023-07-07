from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.util.runner import run_two_node_app


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def create_batch(
    program: QoalaProgram,
    unit_module: UnitModule,
    inputs: List[ProgramInput],
    num_iterations: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        unit_module=unit_module,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
    )


@dataclass
class BqcResult:
    client_results: BatchResult
    server_results: BatchResult


def run_bqc(
    alpha,
    beta,
    theta1,
    theta2,
    num_iterations: int,
) -> BqcResult:
    ns.sim_reset()

    num_qubits = 3
    client_id = 0
    server_id = 1

    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )

    server_program = load_program("bqc_server.iqoala")
    server_input = ProgramInput({"client_id": client_id})

    client_program = load_program("bqc_client.iqoala")
    client_input = ProgramInput(
        {
            "server_id": server_id,
            "alpha": alpha,
            "beta": beta,
            "theta1": theta1,
            "theta2": theta2,
        }
    )

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"server": server_program, "client": client_program},
        program_inputs={"server": server_input, "client": client_input},
        network_cfg=network_cfg,
        linear=True,
    )

    client_results = app_result.batch_results["client"]
    server_results = app_result.batch_results["server"]

    return BqcResult(client_results, server_results)


def check(alpha, beta, theta1, theta2, expected, num_iterations):
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    ns.sim_reset()
    bqc_result = run_bqc(
        alpha=alpha,
        beta=beta,
        theta1=theta1,
        theta2=theta2,
        num_iterations=num_iterations,
    )
    assert len(bqc_result.client_results.results) > 0
    assert len(bqc_result.server_results.results) > 0

    program_results = bqc_result.server_results.results
    m2s = [result.values["m2"] for result in program_results]
    assert all(m2 == expected for m2 in m2s)


def test_bqc():
    # LogManager.enable_task_logger(True)
    # LogManager.log_tasks_to_file("bqc_comp_1.log")
    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=10)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1, num_iterations=10)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0, num_iterations=10)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1, num_iterations=10)


if __name__ == "__main__":
    test_bqc()
