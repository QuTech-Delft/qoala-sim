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
from qoala.util.runner import run_two_node_app_separate_inputs


def create_procnode_cfg(name: str, id: int, determ: bool) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(3),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
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


def run_remote_mbqc(num_iterations: int) -> RemoteMbqcResult:
    ns.sim_reset()

    client_id = 1
    server_id = 0

    client_node_cfg = create_procnode_cfg("client", client_id, determ=True)
    server_node_cfg = create_procnode_cfg("server", server_id, determ=True)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[client_node_cfg, server_node_cfg], link_duration=1000
    )

    client_program = load_program("remote_mbqc_client.iqoala")
    server_program = load_program("remote_mbqc_server.iqoala")

    client_inputs = [
        ProgramInput({"server_id": server_id, "theta0": 0, "theta1": 0, "theta2": 0})
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


def remote_mbqc():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_tasks_to_file("teleport_plus_local.log")
    num_iterations = 1

    result = run_remote_mbqc(num_iterations)
    program_results = result.client_results.results
    print(program_results)
    m0s = [result.values["m0"] for result in program_results]
    m1s = [result.values["m1"] for result in program_results]
    m2s = [result.values["m2"] for result in program_results]
    print(m0s)
    print(m1s)
    print(m2s)


if __name__ == "__main__":
    remote_mbqc()
