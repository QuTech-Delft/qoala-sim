from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.iqoala import IqoalaParser, IqoalaProgram
from qoala.runtime.config import (
    DepolariseLinkConfig,
    GenericQDeviceConfig,
    LinkConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
)
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.schedule import (
    NaiveSolver,
    NoTimeSolver,
    ProgramTaskList,
    TaskBuilder,
)
from qoala.sim.build import build_network
from qoala.sim.network import ProcNodeNetwork

INSTR_LATENCY = 1e5
CC_LATENCY = 1e6
QC_EXPECTATION = 30e6


def create_global_env(
    num_clients: int, global_schedule: List[int], timeslot_len: int
) -> GlobalEnvironment:

    env = GlobalEnvironment()
    env.add_node(0, GlobalNodeInfo("server", 0))
    for i in range(1, num_clients + 1):
        env.add_node(i, GlobalNodeInfo(f"client_{i}", i))

    env.set_global_schedule(global_schedule)
    env.set_timeslot_len(timeslot_len)
    return env


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        name=f"client_{id}",
        node_id=id,
        qdevice_typ="generic",
        qdevice_cfg=GenericQDeviceConfig.perfect_config(1),
        instr_latency=INSTR_LATENCY,
        receive_latency=CC_LATENCY,
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    config_file = relative_to_cwd("node_config.yaml")
    qdevice_cfg = GenericQDeviceConfig.from_file(config_file)
    qdevice_cfg.num_qubits = num_qubits
    return ProcNodeConfig(
        name="server",
        node_id=id,
        qdevice_typ="generic",
        # qdevice_cfg=GenericQDeviceConfig.perfect_config(num_qubits),
        qdevice_cfg=qdevice_cfg,
        instr_latency=INSTR_LATENCY,
        receive_latency=CC_LATENCY,
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
    global_schedule: List[int],
    timeslot_len: int,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    global_env = create_global_env(num_clients, global_schedule, timeslot_len)

    link_cfg_file = relative_to_cwd("link_config.yaml")
    depolarise_config = DepolariseLinkConfig.from_file(link_cfg_file)
    link_cfgs = [
        LinkConfig(
            node1="server", node2=cfg.name, typ="depolarise", cfg=depolarise_config
        )
        for cfg in client_configs
    ]

    node_cfgs = [server_cfg] + client_configs

    network_cfg = ProcNodeNetworkConfig(nodes=node_cfgs, links=link_cfgs)
    return build_network(network_cfg, global_env)


@dataclass
class TaskDurations:
    instr_latency: int
    cc_latency: int
    single_gate: int
    two_gate: int
    meas: int


def create_server_tasks(
    server_program: IqoalaProgram, task_durations: TaskDurations
) -> ProgramTaskList:
    tasks = []

    cl_dur = 2 * task_durations.instr_latency
    cc_dur = 2 * task_durations.cc_latency
    qc_dur = QC_EXPECTATION

    set_dur = task_durations.instr_latency
    rot_dur = task_durations.single_gate
    h_dur = task_durations.single_gate
    meas_dur = task_durations.meas
    free_dur = task_durations.instr_latency
    cphase_dur = task_durations.two_gate

    # run_subroutine(vec<client_id>) : create_epr_0
    task0 = TaskBuilder.CL(cl_dur, 0)
    task1 = TaskBuilder.QC(qc_dur, "create_epr_0")
    # run_subroutine(vec<client_id>) : create_epr_1
    task2 = TaskBuilder.CL(cl_dur, 1)
    task3 = TaskBuilder.QC(qc_dur, "create_epr_1")

    # run_subroutine(vec<client_id>) : local_cphase
    task4 = TaskBuilder.CL(cl_dur, 2)
    task5 = TaskBuilder.QL(set_dur, "local_cphase", 0)
    task6 = TaskBuilder.QL(set_dur, "local_cphase", 1)
    task7 = TaskBuilder.QL(cphase_dur, "local_cphase", 2)

    # csocket = assign_cval() : 0
    task8 = TaskBuilder.CL(cl_dur, 3)
    # delta1 = recv_cmsg(client_id)
    task9 = TaskBuilder.CC(cc_dur, 4)

    # vec<m1> = run_subroutine(vec<delta1>) : meas_qubit_1
    task10 = TaskBuilder.CL(cl_dur, 5)
    task11 = TaskBuilder.QL(set_dur, "meas_qubit_1", 0)
    task12 = TaskBuilder.QL(rot_dur, "meas_qubit_1", 1)
    task13 = TaskBuilder.QL(h_dur, "meas_qubit_1", 2)
    task14 = TaskBuilder.QL(meas_dur, "meas_qubit_1", 3)
    task15 = TaskBuilder.QL(free_dur, "meas_qubit_1", 4)

    # send_cmsg(csocket, m1)
    task16 = TaskBuilder.CC(cc_dur, 6)

    # delta2 = recv_cmsg(csocket)
    task17 = TaskBuilder.CC(cc_dur, 7)

    # vec<m2> = run_subroutine(vec<delta2>) : meas_qubit_0
    task18 = TaskBuilder.CL(cl_dur, 8)
    task19 = TaskBuilder.QL(set_dur, "meas_qubit_0", 0)
    task20 = TaskBuilder.QL(rot_dur, "meas_qubit_0", 1)
    task21 = TaskBuilder.QL(h_dur, "meas_qubit_0", 2)
    task22 = TaskBuilder.QL(meas_dur, "meas_qubit_0", 3)
    task23 = TaskBuilder.QL(free_dur, "meas_qubit_0", 4)

    # send_cmsg(csocket, m2)
    task24 = TaskBuilder.CC(cc_dur, 9)

    tasks.append(TaskBuilder.QC_group([task0, task1]))
    tasks.append(TaskBuilder.QC_group([task2, task3]))
    tasks.append(task4)
    tasks.append(TaskBuilder.QL_group([task5, task6, task7]))
    tasks.append(task8)
    tasks.append(task9)
    tasks.append(task10)
    tasks.append(TaskBuilder.QL_group([task11, task12, task13, task14, task15]))
    tasks.append(task16)
    tasks.append(task17)
    tasks.append(task18)
    tasks.append(TaskBuilder.QL_group([task19, task20, task21, task22, task23]))
    tasks.append(task24)

    return ProgramTaskList(server_program, {i: task for i, task in enumerate(tasks)})


def create_client_tasks(
    client_program: IqoalaProgram, task_durations: TaskDurations
) -> ProgramTaskList:
    tasks = []

    cl_dur = task_durations.instr_latency
    cc_dur = task_durations.cc_latency
    qc_dur = QC_EXPECTATION

    set_dur = task_durations.instr_latency
    rot_dur = task_durations.single_gate
    h_dur = task_durations.single_gate
    meas_dur = task_durations.meas
    free_dur = task_durations.instr_latency

    class Counter:
        def __init__(self):
            self.index = 0

        def next(self):
            index = self.index
            self.index += 1
            return index

    c = Counter()

    # csocket = assign_cval() : 0
    tasks.append(TaskBuilder.CL(cl_dur, c.next()))
    # const_1 = assign_cval() : 1
    tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # compute epr0_rot_y etc
    for _ in range(10):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # run_subroutine(vec<>) : create_epr_0
    tasks.append(TaskBuilder.CL(cl_dur, c.next()))
    # vec<p2> = run_subroutine(vec<theta2>) : post_epr_0
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_0"))

    tasks.append(TaskBuilder.CL(cl_dur, c.next()))
    tasks.append(TaskBuilder.QL(set_dur, "post_epr_0", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "post_epr_0", 1))
    tasks.append(TaskBuilder.QL(h_dur, "post_epr_0", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "post_epr_0", 3))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_0", 4))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_0", 5))

    tasks.append(TaskBuilder.CL(cl_dur, c.next()))
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_1"))
    tasks.append(TaskBuilder.CL(cl_dur, c.next()))
    tasks.append(TaskBuilder.QL(set_dur, "post_epr_1", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "post_epr_1", 1))
    tasks.append(TaskBuilder.QL(h_dur, "post_epr_1", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "post_epr_1", 3))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_1", 4))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_1", 5))

    # x = mult_const(p1) : 16
    # minus_theta1 = mult_const(theta1) : -1
    # delta1 = add_cval_c(minus_theta1, x)
    # delta1 = add_cval_c(delta1, alpha)
    for _ in range(4):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # minus_dummy0 = mult_const(dummy0) : -1
    # should_correct_0 = add_cval_c(const_1, minus_dummy0)
    # delta1_correction = bcond_mult_const(alpha, should_correct_0) : 0
    # delta1_correction = mult_const(delta1_correction) : -1
    # delta1 = add_cval_c(delta1, delta1_correction)
    for _ in range(5):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # send_cmsg(csocket, delta1)
    # m1 = recv_cmsg(csocket)
    tasks.append(TaskBuilder.CC(cl_dur, c.next()))
    tasks.append(TaskBuilder.CC(cc_dur, c.next()))

    # y = mult_const(p2) : 16
    # minus_theta2 = mult_const(theta2) : -1
    # beta = bcond_mult_const(beta, m1) : -1
    # delta2 = add_cval_c(beta, minus_theta2)
    # delta2 = add_cval_c(delta2, y)
    for _ in range(5):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # minus_dummy1 = mult_const(dummy1) : -1
    # should_correct_1 = add_cval_c(const_1, minus_dummy1)
    # delta2_correction = bcond_mult_const(beta, should_correct_1) : 0
    # delta2_correction = mult_const(delta2_correction) : -1
    # delta2 = add_cval_c(delta2, delta2_correction)
    for _ in range(5):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    # send_cmsg(csocket, delta2)
    # m2 = recv_cmsg(csocket)
    tasks.append(TaskBuilder.CC(cl_dur, c.next()))
    tasks.append(TaskBuilder.CC(cl_dur, c.next()))

    # return results
    for _ in range(4):
        tasks.append(TaskBuilder.CL(cl_dur, c.next()))

    return ProgramTaskList(client_program, {i: task for i, task in enumerate(tasks)})


@dataclass
class BqcResult:
    client_batches: List[Dict[int, ProgramBatch]]
    client_results: List[Dict[int, BatchResult]]


def create_durations() -> TaskDurations:
    config_file = relative_to_cwd("node_config.yaml")
    qdevice_cfg = GenericQDeviceConfig.from_file(config_file)

    return TaskDurations(
        instr_latency=INSTR_LATENCY,
        cc_latency=CC_LATENCY,
        single_gate=qdevice_cfg.single_qubit_gate_time,
        two_gate=qdevice_cfg.two_qubit_gate_time,
        meas=qdevice_cfg.measure_time,
    )


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def load_server_program(remote_name: str) -> IqoalaProgram:
    path = relative_to_cwd("bqc_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = IqoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> IqoalaProgram:
    path = relative_to_cwd("bqc_client.iqoala")
    with open(path) as file:
        client_text = file.read()
    return IqoalaParser(client_text).parse()


def create_server_batch(
    client_id: int,
    inputs: List[ProgramInput],
    num_iterations: int,
    deadline: int,
    num_qubits: int,
) -> BatchInfo:
    durations = create_durations()
    server_program = load_server_program(remote_name=f"client_{client_id}")
    server_tasks = create_server_tasks(server_program, durations)
    return BatchInfo(
        program=server_program,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=deadline,
        tasks=server_tasks,
        num_qubits=num_qubits,
    )


def create_client_batch(
    inputs: List[ProgramInput], num_iterations: int, deadline: int
) -> BatchInfo:
    durations = create_durations()
    client_program = load_client_program()
    client_tasks = create_client_tasks(client_program, durations)
    return BatchInfo(
        program=client_program,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=deadline,
        tasks=client_tasks,
        num_qubits=1,
    )


def run_bqc(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    num_iterations: List[int],
    deadlines: List[int],
    num_clients: int,
    global_schedule: List[int],
    timeslot_len: int,
):
    # server needs to have 2 qubits per client
    server_num_qubits = num_clients * 2
    server_config = get_server_config(id=0, num_qubits=server_num_qubits)
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(
        server_config, client_configs, num_clients, global_schedule, timeslot_len
    )

    server_procnode = network.nodes["server"]

    for client_id in range(1, num_clients + 1):
        # index in num_iterations and deadlines list
        index = client_id - 1

        server_inputs = [
            ProgramInput({"client_id": client_id}) for _ in range(num_iterations[index])
        ]

        server_batch_info = create_server_batch(
            client_id=client_id,
            inputs=server_inputs,
            num_iterations=num_iterations[index],
            deadline=deadlines[index],
            num_qubits=server_num_qubits,
        )

        server_procnode.submit_batch(server_batch_info)
    server_procnode.initialize_processes()
    server_procnode.initialize_schedule(NaiveSolver)

    for client_id in range(1, num_clients + 1):
        # index in num_iterations and deadlines list
        index = client_id - 1

        client_inputs = [
            ProgramInput(
                {
                    "server_id": 0,
                    "alpha": alpha,
                    "beta": beta,
                    "theta1": theta1,
                    "theta2": theta2,
                    "dummy0": dummy0,
                    "dummy1": dummy1,
                }
            )
            for _ in range(num_iterations[index])
        ]

        client_batch_info = create_client_batch(
            client_inputs, num_iterations[index], deadlines[index]
        )

        client_procnode = network.nodes[f"client_{client_id}"]
        client_procnode.submit_batch(client_batch_info)
        client_procnode.initialize_processes()
        client_procnode.initialize_schedule(NoTimeSolver)

    network.start_all_nodes()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    client_procnodes = [network.nodes[f"client_{i}"] for i in range(1, num_clients + 1)]
    client_batches = [node.get_batches() for node in client_procnodes]

    client_results: List[Dict[int, BatchResult]]
    client_results = [node.scheduler.get_batch_results() for node in client_procnodes]

    return BqcResult(client_batches, client_results), makespan


def check(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
    expected,
    num_iterations,
    deadlines,
    num_clients,
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.sim_reset()
    bqc_result, makespan = run_bqc(
        alpha=alpha,
        beta=beta,
        theta1=theta1,
        theta2=theta2,
        dummy0=dummy0,
        dummy1=dummy1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )

    batch_success_probabilities: List[float] = []

    for i in range(num_clients):
        assert len(bqc_result.client_results[i]) == 1
        batch_result = bqc_result.client_results[i][0]
        assert len(bqc_result.client_batches[i]) == 1
        program_batch = bqc_result.client_batches[i][0]
        batch_iterations = program_batch.info.num_iterations

        p1s = [result.values["p1"] for result in batch_result.results]
        p2s = [result.values["p2"] for result in batch_result.results]
        m1s = [result.values["m1"] for result in batch_result.results]
        m2s = [result.values["m2"] for result in batch_result.results]

        if dummy0 == 0:
            num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
        else:  # dummy0 = 1
            num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

        frac_fail = round(num_fails / batch_iterations, 2)
        batch_success_probabilities.append(1 - frac_fail)

    return batch_success_probabilities, makespan


def compute_succ_prob(
    num_clients: int,
    num_iterations: List[int],
    deadlines: List[int],
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    return check(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=1,
        expected=1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )


def test_bqc():
    num_clients = 10
    succ_probs, makespan = compute_succ_prob(
        num_clients=num_clients,
        num_iterations=[30] * num_clients,
        deadlines=[1e8] * num_clients,
        global_schedule=[i for i in range(num_clients)],
        timeslot_len=50e6,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


if __name__ == "__main__":
    test_bqc()
