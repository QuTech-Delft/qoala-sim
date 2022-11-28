from __future__ import annotations

import os
from typing import List

import netsquid as ns

from qoala.lang.iqoala import IqoalaParser, IqoalaProgram
from qoala.runtime.config import (
    GenericQDeviceConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
)
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.program import BatchInfo, ProgramInput
from qoala.runtime.schedule import NaiveSolver, ProgramTaskList, TaskBuilder
from qoala.sim.build import build_network
from qoala.sim.network import ProcNodeNetwork


def create_global_env() -> GlobalEnvironment:

    env = GlobalEnvironment()
    env.add_node(0, GlobalNodeInfo("alice", 0))

    env.set_global_schedule([0])
    env.set_timeslot_len(1e6)

    return env


def get_config() -> ProcNodeConfig:
    qdevice_cfg = GenericQDeviceConfig.perfect_config(1)
    qdevice_cfg.T1 = 1
    qdevice_cfg.T2 = 1
    # qdevice_cfg.single_qubit_gate_depolar_prob = 1
    return ProcNodeConfig(
        name="alice",
        node_id=0,
        qdevice_typ="generic",
        qdevice_cfg=qdevice_cfg,
        instr_latency=1000,
    )


def create_network(
    node_cfg: ProcNodeConfig,
) -> ProcNodeNetwork:
    global_env = create_global_env()

    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    return build_network(network_cfg, global_env)


def load_program() -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "simple_program.iqoala")
    with open(path) as file:
        text = file.read()
    program = IqoalaParser(text).parse()

    return program


def create_tasks(program: IqoalaProgram) -> ProgramTaskList:
    tasks = []

    cl_dur = 1000
    ql_dur = 1e6

    # vec<m> = run_subroutine(vec<>) : subrt0
    tasks.append(TaskBuilder.CL(cl_dur, 0))
    for i in range(4):
        tasks.append(TaskBuilder.QL(ql_dur, "subrt0", i))

    # x = assign_cval() : 0
    tasks.append(TaskBuilder.CL(1e4, 1))

    # vec<m> = run_subroutine(vec<>) : subrt1
    tasks.append(TaskBuilder.CL(cl_dur, 2))
    for i in range(2):
        tasks.append(TaskBuilder.QL(ql_dur, "subrt1", i))

    # return_result(x)
    tasks.append(TaskBuilder.CL(cl_dur, 3))

    return ProgramTaskList(program, {i: task for i, task in enumerate(tasks)})


def create_batch(
    inputs: List[ProgramInput],
    num_iterations: int,
    deadline: int,
    num_qubits: int,
) -> BatchInfo:
    program = load_program()
    tasks = create_tasks(program)

    return BatchInfo(
        program=program,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=deadline,
        tasks=tasks,
        num_qubits=num_qubits,
    )


def run_program():
    ns.sim_reset()

    node_config = get_config()
    network = create_network(node_config)
    procnode = network.nodes["alice"]

    num_iterations = 100
    inputs = [ProgramInput({}) for i in range(num_iterations)]

    batch_info = create_batch(
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
        num_qubits=1,
    )

    procnode.submit_batch(batch_info)
    procnode.initialize_processes()
    procnode.initialize_schedule(NaiveSolver)

    network.start_all_nodes()
    ns.sim_run()

    all_results = procnode.scheduler.get_batch_results()
    batch0_result = all_results[0]
    results = [result.values["m"] for result in batch0_result.results]
    print(results)


def test_simple_program():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("logs/simple_program.log")

    run_program()


if __name__ == "__main__":
    test_simple_program()
