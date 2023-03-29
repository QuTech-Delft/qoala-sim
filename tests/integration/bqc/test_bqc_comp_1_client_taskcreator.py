from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import netsquid as ns

from qoala.lang.ehi import ExposedHardwareInfo, UnitModule
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput, ProgramInstance
from qoala.runtime.schedule import NoTimeSolver, ProgramTaskList, TaskBuilder
from qoala.runtime.taskcreator import (
    CpuSchedule,
    QpuSchedule,
    TaskCreator,
    TaskExecutionMode,
)
from qoala.sim.build import build_network


def create_network_info(names: List[str]) -> NetworkInfo:
    env = NetworkInfo.with_nodes({i: name for i, name in enumerate(names)})
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


def create_server_tasks(
    server_program: IqoalaProgram, cfg: ProcNodeConfig
) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    # ql_dur = 1e4
    qc_dur = 1e6

    topology_cfg: TopologyConfig = cfg.topology

    single_qubit_gate_time = topology_cfg.get_single_gate_configs()[0][0].to_duration()
    two_qubit_gate_time = list(topology_cfg.get_multi_gate_configs().values())[0][
        0
    ].to_duration()

    set_dur = cfg.latencies.qnos_instr_time
    rot_dur = single_qubit_gate_time
    h_dur = single_qubit_gate_time
    meas_dur = single_qubit_gate_time
    free_dur = cfg.latencies.qnos_instr_time
    cphase_dur = two_qubit_gate_time

    tasks.append(TaskBuilder.CL(cl_dur, 0))
    tasks.append(TaskBuilder.QC(qc_dur, 1, "req0"))
    tasks.append(TaskBuilder.QC(qc_dur, 2, "req1"))
    dur = cl_dur + 2 * set_dur + cphase_dur
    tasks.append(TaskBuilder.QL(dur, 3, "local_cphase"))
    tasks.append(TaskBuilder.CC(cc_dur, 4))
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 5, "meas_qubit_1"))
    tasks.append(TaskBuilder.CC(cc_dur, 6))
    tasks.append(TaskBuilder.CC(cc_dur, 7))
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 8, "meas_qubit_0"))
    tasks.append(TaskBuilder.CL(cl_dur, 9))
    tasks.append(TaskBuilder.CL(cl_dur, 10))

    return ProgramTaskList(server_program, {i: task for i, task in enumerate(tasks)})


def create_client_tasks(
    client_program: IqoalaProgram, cfg: ProcNodeConfig
) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    # ql_dur = 1e3
    qc_dur = 1e6

    topology_cfg: TopologyConfig = cfg.topology

    single_qubit_gate_time = topology_cfg.get_single_gate_configs()[0][0].to_duration()

    set_dur = cfg.latencies.qnos_instr_time
    rot_dur = single_qubit_gate_time
    h_dur = single_qubit_gate_time
    meas_dur = single_qubit_gate_time
    free_dur = cfg.latencies.qnos_instr_time

    tasks.append(TaskBuilder.CL(cl_dur, 0))
    tasks.append(TaskBuilder.QC(qc_dur, 1, "req0"))
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 2, "post_epr_0"))
    tasks.append(TaskBuilder.QC(qc_dur, 3, "req1"))
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 4, "post_epr_1"))
    tasks.append(TaskBuilder.CL(cl_dur, 5))
    tasks.append(TaskBuilder.CL(cl_dur, 6))
    tasks.append(TaskBuilder.CL(cl_dur, 7))
    tasks.append(TaskBuilder.CL(cl_dur, 8))
    tasks.append(TaskBuilder.CC(cc_dur, 9))
    tasks.append(TaskBuilder.CC(cc_dur, 10))
    tasks.append(TaskBuilder.CL(cl_dur, 11))
    tasks.append(TaskBuilder.CL(cl_dur, 12))
    tasks.append(TaskBuilder.CL(cl_dur, 13))
    tasks.append(TaskBuilder.CL(cl_dur, 14))
    tasks.append(TaskBuilder.CL(cl_dur, 15))
    tasks.append(TaskBuilder.CC(cc_dur, 16))
    tasks.append(TaskBuilder.CL(cl_dur, 17))
    tasks.append(TaskBuilder.CL(cl_dur, 18))

    return ProgramTaskList(client_program, {i: task for i, task in enumerate(tasks)})


@dataclass
class BqcResult:
    client_results: Dict[int, BatchResult]
    server_results: Dict[int, BatchResult]


def instantiate(
    program: IqoalaProgram,
    ehi: ExposedHardwareInfo,
    pid: int = 0,
    inputs: Optional[ProgramInput] = None,
) -> ProgramInstance:
    unit_module = UnitModule.from_full_ehi(ehi)

    if inputs is None:
        inputs = ProgramInput.empty()

    return ProgramInstance(
        pid,
        program,
        inputs,
        tasks=ProgramTaskList.empty(program),
        unit_module=unit_module,
    )


def run_bqc(alpha, beta, theta1, theta2, num_iterations: int):
    ns.sim_reset()

    num_qubits = 3
    network_info = create_network_info(names=["client", "server"])
    server_id = network_info.get_node_id("server")
    client_id = network_info.get_node_id("client")

    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[server_node_cfg, client_node_cfg], link_duration=1000
    )
    network = build_network(network_cfg, network_info)
    server_procnode = network.nodes["server"]
    client_procnode = network.nodes["client"]

    server_program = load_program("bqc_server.iqoala")
    server_input = ProgramInput({"client_id": client_id})
    server_instance = instantiate(
        server_program, server_procnode.local_ehi, 0, server_input
    )
    server_procnode.scheduler.submit_program_instance(server_instance)

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
    client_instance = instantiate(
        client_program, client_procnode.local_ehi, 0, client_input
    )
    client_procnode.scheduler.submit_program_instance(client_instance)

    task_creator = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC)
    tasks_server = task_creator.from_program(
        server_program, 0, server_procnode.local_ehi, server_procnode.network_ehi
    )
    tasks_client = task_creator.from_program(
        client_program, 0, client_procnode.local_ehi, client_procnode.network_ehi
    )
    cpu_server = CpuSchedule.consecutive(tasks_server)
    qpu_server = QpuSchedule.consecutive(tasks_server)
    cpu_client = CpuSchedule.consecutive(tasks_client)
    qpu_client = QpuSchedule.consecutive(tasks_client)

    server_procnode.scheduler.upload_cpu_schedule(cpu_server)
    server_procnode.scheduler.upload_qpu_schedule(qpu_server)
    client_procnode.scheduler.upload_cpu_schedule(cpu_client)
    client_procnode.scheduler.upload_qpu_schedule(qpu_client)

    network.start()
    ns.sim_run()

    client_results = client_procnode.scheduler.get_batch_results()
    server_results = server_procnode.scheduler.get_batch_results()

    return BqcResult(client_results, server_results)


def test_bqc():
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("test_run.log")

    def check(alpha, beta, theta1, theta2, expected, num_iterations):
        ns.sim_reset()
        bqc_result = run_bqc(
            alpha=alpha,
            beta=beta,
            theta1=theta1,
            theta2=theta2,
            num_iterations=num_iterations,
        )
        assert len(bqc_result.client_results) > 0
        assert len(bqc_result.server_results) > 0

        server_batch_results = bqc_result.server_results
        for _, batch_results in server_batch_results.items():
            program_results = batch_results.results
            m2s = [result.values["m2"] for result in program_results]
            assert all(m2 == expected for m2 in m2s)

    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0, num_iterations=10)
    # check(alpha=8, beta=24, theta1=0, theta2=0, expected=1, num_iterations=10)
    # check(alpha=8, beta=8, theta1=13, theta2=27, expected=0, num_iterations=10)
    # check(alpha=8, beta=24, theta1=2, theta2=22, expected=1, num_iterations=10)


if __name__ == "__main__":
    test_bqc()
