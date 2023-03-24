from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.hostlang import RunRequestOp
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.schedule import NoTimeSolver, ProgramTaskList, TaskBuilder
from qoala.sim.build import build_network


def create_global_env(names: List[str]) -> GlobalEnvironment:
    env = GlobalEnvironment()
    for i, name in enumerate(names):
        env.add_node(i, GlobalNodeInfo(name, i))
    env.set_global_schedule([0, 1, 2])
    env.set_timeslot_len(1e6)
    return env


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


def load_program(path: str) -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return IqoalaParser(text).parse()


def create_batch(
    program: IqoalaProgram,
    unit_module: UnitModule,
    inputs: List[ProgramInput],
    num_iterations: int,
    tasks: ProgramTaskList,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        unit_module=unit_module,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
        tasks=tasks,
    )


def create_alice_tasks(alice_program: IqoalaProgram) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    qc_dur = 1e6

    for i, instr in enumerate(alice_program.instructions):
        if isinstance(instr, RunRequestOp):
            req_name = instr.req_routine
            tasks.append(TaskBuilder.QC(qc_dur, i, req_name))
        else:
            tasks.append(TaskBuilder.CL(cl_dur, i))

    return ProgramTaskList(alice_program, {i: task for i, task in enumerate(tasks)})


def create_bob_tasks(bob_program: IqoalaProgram) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    qc_dur = 1e6
    for i, instr in enumerate(bob_program.instructions):
        if isinstance(instr, RunRequestOp):
            req_name = instr.req_routine
            tasks.append(TaskBuilder.QC(qc_dur, i, req_name))
        else:
            tasks.append(TaskBuilder.CL(cl_dur, i))

    return ProgramTaskList(bob_program, {i: task for i, task in enumerate(tasks)})


@dataclass
class QkdResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_qkd(num_iterations: int, alice_file: str, bob_file: str):
    ns.sim_reset()

    num_qubits = 3
    global_env = create_global_env(names=["alice", "bob"])
    alice_id = global_env.get_node_id("alice")
    bob_id = global_env.get_node_id("bob")

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig(nodes=[alice_node_cfg, bob_node_cfg], links=[])
    network = build_network(network_cfg, global_env)
    alice_procnode = network.nodes["alice"]
    bob_procnode = network.nodes["bob"]

    alice_program = load_program(alice_file)
    alice_tasks = create_alice_tasks(alice_program)
    alice_inputs = [ProgramInput({"bob_id": bob_id}) for _ in range(num_iterations)]

    alice_unit_module = UnitModule.from_full_ehi(alice_procnode.memmgr.get_ehi())
    alice_batch = create_batch(
        alice_program, alice_unit_module, alice_inputs, num_iterations, alice_tasks
    )
    alice_procnode.submit_batch(alice_batch)
    alice_procnode.initialize_processes()
    alice_procnode.initialize_schedule(NoTimeSolver)

    bob_program = load_program(bob_file)
    bob_tasks = create_bob_tasks(bob_program)
    bob_inputs = [ProgramInput({"alice_id": alice_id}) for _ in range(num_iterations)]

    bob_unit_module = UnitModule.from_full_ehi(bob_procnode.memmgr.get_ehi())
    bob_batch = create_batch(
        bob_program, bob_unit_module, bob_inputs, num_iterations, bob_tasks
    )
    bob_procnode.submit_batch(bob_batch)
    bob_procnode.initialize_processes()
    bob_procnode.initialize_schedule(NoTimeSolver)

    network.start()
    ns.sim_run()

    # only one batch (ID = 0), so get value at index 0
    alice_result = alice_procnode.scheduler.get_batch_results()[0]
    bob_result = bob_procnode.scheduler.get_batch_results()[0]

    return QkdResult(alice_result, bob_result)


def test_qkd_md_1pair():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_md_1pair_alice.iqoala"
    bob_file = "qkd_md_1pair_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)
    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]


def test_qkd_md_2pairs():
    ns.sim_reset()

    num_iterations = 10
    alice_file = "qkd_md_2pairs_alice.iqoala"
    bob_file = "qkd_md_2pairs_bob.iqoala"

    qkd_result = run_qkd(num_iterations, alice_file, bob_file)

    alice_results = qkd_result.alice_result.results
    bob_results = qkd_result.bob_result.results

    assert len(alice_results) == num_iterations
    assert len(bob_results) == num_iterations

    alice_outcomes = [alice_results[i].values for i in range(num_iterations)]
    bob_outcomes = [bob_results[i].values for i in range(num_iterations)]

    for alice, bob in zip(alice_outcomes, bob_outcomes):
        assert alice["m0"] == bob["m0"]
        assert alice["m1"] == bob["m1"]


if __name__ == "__main__":
    test_qkd_md_1pair()
    test_qkd_md_2pairs()
