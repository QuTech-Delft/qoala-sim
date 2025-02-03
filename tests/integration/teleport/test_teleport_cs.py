from __future__ import annotations

import os
from dataclasses import dataclass

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import run_two_node_app


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


@dataclass
class TeleportResult:
    alice_results: BatchResult
    bob_results: BatchResult


def run_teleport(num_iterations: int, cs: bool = False) -> TeleportResult:
    ns.sim_reset()

    num_qubits = 4
    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits, determ=True)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits, determ=True)

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, 4e6)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1e6
    )
    network_cfg.cconns = [cconn]
    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    network_cfg.netschedule = NetworkScheduleConfig(
        bin_length=1.1e6,
        first_bin=0,
        bin_pattern=pattern,
        repeat_period=(1.2e6 * num_iterations),
    )

    alice_program = load_program("teleport_alice.iqoala")
    if cs:
        bob_program = load_program("teleport_bob_cs.iqoala")
    else:
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
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    return TeleportResult(alice_result, bob_result)


def run_teleport_no_cs() -> int:
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("teleport.log")
    LogManager.log_tasks_to_file("teleport_tasks.log")
    num_iterations = 2

    ns.sim_reset()

    start = ns.sim_time()
    print(f"start: {start}")
    result = run_teleport(num_iterations=num_iterations)
    end = ns.sim_time()
    print(f"end: {end}")

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)

    makespan = end - start
    return makespan


def run_teleport_cs() -> int:
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("INFO")
    LogManager.log_to_file("teleport_cs.log")
    LogManager.log_tasks_to_file("teleport_cs_tasks.log")
    num_iterations = 2

    ns.sim_reset()

    start = ns.sim_time()
    print(f"start: {start}")
    result = run_teleport(num_iterations=num_iterations, cs=True)
    end = ns.sim_time()
    print(f"end: {end}")

    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)

    makespan = end - start
    return makespan


def test_compare_cs_vs_no_cs():
    makespan1 = run_teleport_no_cs()
    makespan2 = run_teleport_cs()

    print(makespan1)
    print(makespan2)

    # Because the CS does not allow interleaving, we expect a larger makespan
    # in this case. Note that in general this is highly dependent on the exact
    # program, CS, and network schedule though!
    assert makespan2 > makespan1


if __name__ == "__main__":
    test_compare_cs_vs_no_cs()
