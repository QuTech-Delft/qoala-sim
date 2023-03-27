from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.config import (
    GenericQDeviceConfig,
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
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


def topology_config(num_qubits: int) -> TopologyConfig:
    return TopologyConfig.perfect_config_uniform(
        num_qubits,
        single_instructions=[
            "INSTR_INIT",
            "INSTR_ROT_X",
            "INSTR_ROT_Y",
            "INSTR_ROT_Z",
            "INSTR_X",
            "INSTR_Y",
            "INSTR_Z",
            "INSTR_H",
            "INSTR_MEASURE",
        ],
        single_duration=1e3,
        two_instructions=["INSTR_CNOT", "INSTR_CZ"],
        two_duration=100e3,
    )


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client_{id}",
        node_id=id,
        topology=topology_config(1),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=topology_config(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
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

    node_cfgs = [server_cfg] + client_configs

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )
    return build_network(network_cfg, global_env)


@dataclass
class TaskDurations:
    instr_latency: int
    rot_dur: int
    h_dur: int
    meas_dur: int
    free_dur: int
    cphase_dur: int


def create_server_tasks(
    server_program: IqoalaProgram, task_durations: TaskDurations
) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    # ql_dur = 1e4
    qc_dur = 1e6

    set_dur = task_durations.instr_latency
    rot_dur = task_durations.rot_dur
    h_dur = task_durations.h_dur
    meas_dur = task_durations.meas_dur
    free_dur = task_durations.free_dur
    cphase_dur = task_durations.cphase_dur

    # csocket = assign_cval() : 0
    tasks.append(TaskBuilder.CL(cl_dur, 0))

    # run_request(vec<>) : req0
    tasks.append(TaskBuilder.QC(qc_dur, 1, "req0"))

    # run_request(vec<>) : req1
    tasks.append(TaskBuilder.QC(qc_dur, 2, "req1"))

    # run_subroutine(vec<client_id>) : local_cphase
    dur = cl_dur + 2 * set_dur + cphase_dur
    tasks.append(TaskBuilder.QL(dur, 3, "local_cphase"))

    # delta1 = recv_cmsg(client_id)
    tasks.append(TaskBuilder.CC(cc_dur, 4))

    # vec<m1> = run_subroutine(vec<delta1>) : meas_qubit_1
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 5, "meas_qubit_1"))

    # send_cmsg(csocket, m1)
    tasks.append(TaskBuilder.CC(cc_dur, 6))
    # delta2 = recv_cmsg(csocket)
    tasks.append(TaskBuilder.CC(cc_dur, 7))

    # vec<m2> = run_subroutine(vec<delta2>) : meas_qubit_0
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, 8, "meas_qubit_0"))

    # send_cmsg(csocket, m2)
    tasks.append(TaskBuilder.CC(cc_dur, 9))

    return ProgramTaskList(server_program, {i: task for i, task in enumerate(tasks)})


def create_client_tasks(
    client_program: IqoalaProgram, task_durations: TaskDurations
) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    # ql_dur = 1e3
    qc_dur = 1e6

    set_dur = task_durations.instr_latency
    rot_dur = task_durations.rot_dur
    h_dur = task_durations.h_dur
    meas_dur = task_durations.meas_dur
    free_dur = task_durations.free_dur

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

    tasks.append(TaskBuilder.QC(qc_dur, c.next(), "req0"))

    # vec<p2> = run_subroutine(vec<theta2>) : post_epr_0
    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, c.next(), "post_epr_0"))

    tasks.append(TaskBuilder.QC(qc_dur, c.next(), "req1"))

    dur = cl_dur + set_dur + rot_dur + h_dur + meas_dur + free_dur
    tasks.append(TaskBuilder.QL(dur, c.next(), "post_epr_1"))

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
    perfect_qdevice_cfg = GenericQDeviceConfig.perfect_config(1)
    instr_latency = 1000

    return TaskDurations(
        instr_latency=instr_latency,
        rot_dur=perfect_qdevice_cfg.single_qubit_gate_time,
        h_dur=perfect_qdevice_cfg.single_qubit_gate_time,
        meas_dur=perfect_qdevice_cfg.measure_time,
        free_dur=instr_latency,
        cphase_dur=perfect_qdevice_cfg.two_qubit_gate_time,
    )


def load_server_program(remote_name: str) -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = IqoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "vbqc_client.iqoala")
    with open(path) as file:
        client_text = file.read()
    return IqoalaParser(client_text).parse()


def create_server_batch(
    client_id: int,
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
    deadline: int,
) -> BatchInfo:
    durations = create_durations()
    server_program = load_server_program(remote_name=f"client_{client_id}")
    server_tasks = create_server_tasks(server_program, durations)
    return BatchInfo(
        program=server_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=deadline,
        tasks=server_tasks,
    )


def create_client_batch(
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
    deadline: int,
) -> BatchInfo:
    durations = create_durations()
    client_program = load_client_program()
    client_tasks = create_client_tasks(client_program, durations)
    return BatchInfo(
        program=client_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=deadline,
        tasks=client_tasks,
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
    ns.sim_reset()

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

        server_unit_module = UnitModule.from_full_ehi(server_procnode.memmgr.get_ehi())
        server_batch_info = create_server_batch(
            client_id=client_id,
            inputs=server_inputs,
            unit_module=server_unit_module,
            num_iterations=num_iterations[index],
            deadline=deadlines[index],
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

        client_procnode = network.nodes[f"client_{client_id}"]

        client_unit_module = UnitModule.from_full_ehi(client_procnode.memmgr.get_ehi())
        client_batch_info = create_client_batch(
            client_inputs, client_unit_module, num_iterations[index], deadlines[index]
        )

        client_procnode.submit_batch(client_batch_info)
        client_procnode.initialize_processes()
        client_procnode.initialize_schedule(NoTimeSolver)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    client_procnodes = [network.nodes[f"client_{i}"] for i in range(1, num_clients + 1)]
    client_batches = [node.get_batches() for node in client_procnodes]

    client_results: List[Dict[int, BatchResult]]
    client_results = [node.scheduler.get_batch_results() for node in client_procnodes]

    return BqcResult(client_batches, client_results), makespan


def check_computation(
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

        m2s = [result.values["m2"] for result in batch_result.results]
        correct_outcomes = len([m2 for m2 in m2s if m2 == expected])
        succ_prob = round(correct_outcomes / batch_iterations, 2)
        batch_success_probabilities.append(succ_prob)

    return batch_success_probabilities, makespan


def compute_succ_prob_computation(
    num_clients: int,
    num_iterations: List[int],
    deadlines: List[int],
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    return check_computation(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=0,
        expected=1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )


def check_trap(
    alpha,
    beta,
    theta1,
    theta2,
    dummy0,
    dummy1,
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
            # corresponds to "dummy = 1"
            # do normal rotations on qubit 0
            # no rotations on qubit 1
            num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
        else:  # dummy0 = 1
            # corresponds to "dummy = 2"
            # no rotations on qubit 0
            # do normal rotations on qubit 1
            num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

        frac_fail = round(num_fails / batch_iterations, 2)
        batch_success_probabilities.append(1 - frac_fail)

    return batch_success_probabilities, makespan


def compute_succ_prob_trap(
    num_clients: int,
    num_iterations: List[int],
    deadlines: List[int],
    global_schedule: List[int],
    timeslot_len: int,
):
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    return check_trap(
        alpha=8,
        beta=24,
        theta1=2,
        theta2=22,
        dummy0=0,
        dummy1=1,
        num_iterations=num_iterations,
        deadlines=deadlines,
        num_clients=num_clients,
        global_schedule=global_schedule,
        timeslot_len=timeslot_len,
    )


def test_bqc_computation():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("logs/test_computation.log")

    num_clients = 3

    succ_probs, makespan = compute_succ_prob_computation(
        num_clients=num_clients,
        num_iterations=[30] * num_clients,
        deadlines=[1e9] * num_clients,
        global_schedule=[i for i in range(num_clients)],
        timeslot_len=1e6,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


def test_bqc_trap():
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("logs/test_trap.log")

    num_clients = 4

    succ_probs, makespan = compute_succ_prob_trap(
        num_clients=num_clients,
        num_iterations=[20] * num_clients,
        deadlines=[1e9] * num_clients,
        global_schedule=[i for i in range(num_clients)],
        timeslot_len=1e6,
    )
    print(f"success probabilities: {succ_probs}")
    print(f"makespan: {makespan}")


if __name__ == "__main__":
    test_bqc_computation()
    test_bqc_trap()
