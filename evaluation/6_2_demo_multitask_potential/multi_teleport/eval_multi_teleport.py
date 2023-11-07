from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

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
from qoala.runtime.program import ProgramInput
from qoala.util.runner import AppResult, run_two_node_app


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, determ: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def run_teleport(
    num_iterations: int, linear: bool, num_qubits_alice: int, num_qubits_bob: int
) -> AppResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice", alice_id, num_qubits_alice, determ=True
    )
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits_bob, determ=True)
    bob_node_cfg.topology.qubits[0].qubit_config.noise_config.to_error_model_kwargs()[
        "T2"
    ] = 1

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 1e7)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("teleport_alice.iqoala")
    bob_program = load_program("teleport_bob.iqoala")

    # state = 5 -> teleport |1> state
    alice_input = ProgramInput({"bob_id": bob_id, "state": 5})
    # state = 4 -> measure in +Z, i.e. expect a "1" outcome
    bob_input = ProgramInput({"alice_id": alice_id, "state": 4})

    app_result = run_two_node_app(
        num_iterations=num_iterations,
        programs={"alice": alice_program, "bob": bob_program},
        program_inputs={"alice": alice_input, "bob": bob_input},
        network_cfg=network_cfg,
        linear_for={"alice": True, "bob": linear},
    )
    # print(f"makespan: {app_result.total_duration:_}")

    return app_result


def get_teleport_makespan(
    num_iterations: int, linear: bool, num_qubits_alice: int, num_qubits_bob: int
) -> float:
    result = run_teleport(
        num_iterations=num_iterations,
        linear=linear,
        num_qubits_alice=num_qubits_alice,
        num_qubits_bob=num_qubits_bob,
    )

    bob_result = result.batch_results["bob"]

    program_results = bob_result.results
    outcomes = [result.values["outcome"] for result in program_results]
    # print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)

    return result.total_duration


@dataclass
class DataPoint:
    linear: bool
    num_qubits_alice: int
    num_qubits_bob: int
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int


if __name__ == "__main__":
    # LogManager.set_log_level("INFO")
    # LogManager.log_to_file("multi_teleport.log")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_tasks_to_file("multi_teleport_tasks.log")

    data: List[DataPoint] = []

    num_qubits_alice = 2

    for linear in [False, True]:
        for num_qubits_bob in range(1, 6):
            print(f"{linear}, {num_qubits_alice}, {num_qubits_bob}")
            makespan = get_teleport_makespan(
                100, linear, num_qubits_alice, num_qubits_bob
            )
            point = DataPoint(linear, num_qubits_alice, num_qubits_bob, makespan)
            data.append(point)

    for p in data:
        print(f"{p.linear}, {p.num_qubits_alice}, {p.num_qubits_bob}, {p.makespan:_}")
