from __future__ import annotations

import os
import random
from dataclasses import dataclass
from enum import Enum, auto
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
from qoala.runtime.task import TaskExecutionMode
from qoala.util.logging import LogManager
from qoala.util.runner import run_two_node_app, run_two_node_app_separate_inputs


def create_procnode_cfg(name: str, id: int, t1: float, t2: float) -> ProcNodeConfig:
    nv_params = NvParams()
    nv_params.comm_t1 = t1
    nv_params.comm_t2 = t2
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.from_nv_params(num_qubits=5, params=nv_params),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
        tem=TaskExecutionMode.QOALA.name,
        determ_sched=True,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=NVFlavour()).parse()


@dataclass
class TeleportResult:
    alice_results: BatchResult
    bob_results: BatchResult


class TeleportBasis(Enum):
    X0 = 0
    X1 = 1
    Y0 = 2
    Y1 = 3
    Z0 = 4
    Z1 = 5


def run_teleport(
    num_iterations: int,
    alice_bases: List[TeleportBasis],
    bob_bases: List[TeleportBasis],
    t1: float,
    t2: float,
    cc_latency: float,
) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 3
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, t1, t2)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, t1, t2)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc_latency)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("teleport_nv_alice.iqoala")
    bob_program = load_program("teleport_nv_bob.iqoala")

    alice_inputs = [
        ProgramInput({"bob_id": bob_id, "state": b.value}) for b in alice_bases
    ]
    bob_inputs = [
        ProgramInput({"alice_id": alice_id, "state": b.value}) for b in bob_bases
    ]

    app_result = run_two_node_app_separate_inputs(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_inputs, "bob": bob_inputs},
        network_cfg=network_cfg,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return TeleportResult(alice_result, bob_result)


def test_teleport_different_inputs():
    # LogManager.set_log_level("DEBUG")
    # LogManager.enable_task_logger(True)
    num_iterations = 10

    t1 = 1e9
    t2 = 1e9

    alice_bases = [TeleportBasis.Y0 for _ in range(num_iterations)]
    bob_bases = [TeleportBasis.Y1 for _ in range(num_iterations)]

    for cc in [1e6, 1e7, 1e8, 1e9, 1e10]:
        result = run_teleport(num_iterations, alice_bases, bob_bases, t1, t2, cc)
        program_results = result.bob_results.results
        outcomes = [result.values["outcome"] for result in program_results]
        print(f"cc = {cc}: {outcomes}")


if __name__ == "__main__":
    test_teleport_different_inputs()
