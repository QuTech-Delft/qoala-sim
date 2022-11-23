from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import netsquid as ns
from netsquid.nodes import Node
from netsquid.qubits import ketstates, qubitapi
from netsquid_magic.link_layer import (
    MagicLinkLayerProtocolWithSignaling,
    SingleClickTranslationUnit,
)
from netsquid_magic.magic_distributor import PerfectStateMagicDistributor

from qoala.lang.iqoala import IqoalaParser, IqoalaProgram
from qoala.runtime.config import (
    GenericQDeviceConfig,
    LinkConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
)
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.schedule import (
    NaiveSolver,
    NoTimeSolver,
    ProgramTaskList,
    TaskBuilder,
)
from qoala.sim.build import build_network
from qoala.sim.egp import EgpProtocol
from qoala.sim.logging import LogManager
from qoala.sim.network import ProcNodeNetwork


def create_global_env(num_clients: int) -> GlobalEnvironment:

    env = GlobalEnvironment()
    env.add_node(0, GlobalNodeInfo("server", 0))
    for i in range(1, num_clients + 1):
        env.add_node(i, GlobalNodeInfo(f"client_{i}", i))
    return env


def create_egp_protocols(node1: Node, node2: Node) -> Tuple[EgpProtocol, EgpProtocol]:
    link_dist = PerfectStateMagicDistributor(nodes=[node1, node2], state_delay=0)
    link_prot = MagicLinkLayerProtocolWithSignaling(
        nodes=[node1, node2],
        magic_distributor=link_dist,
        translation_unit=SingleClickTranslationUnit(),
    )
    return EgpProtocol(node1, link_prot), EgpProtocol(node2, link_prot)


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        name=f"client_{id}",
        node_id=id,
        qdevice_typ="generic",
        qdevice_cfg=GenericQDeviceConfig.perfect_config(1),
        instr_latency=1000,
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        name="server",
        node_id=id,
        qdevice_typ="generic",
        qdevice_cfg=GenericQDeviceConfig.perfect_config(num_qubits),
        instr_latency=1000,
    )


def create_network(
    server_cfg: ProcNodeConfig, client_configs: List[ProcNodeConfig], num_clients: int
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    global_env = create_global_env(num_clients)

    link_cfgs = [
        LinkConfig.perfect_config("server", cfg.name) for cfg in client_configs
    ]

    node_cfgs = [server_cfg] + client_configs

    network_cfg = ProcNodeNetworkConfig(nodes=node_cfgs, links=link_cfgs)
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
    # run_subroutine(vec<client_id>) : create_epr_0
    tasks.append(TaskBuilder.CL(cl_dur, 1))
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_0"))
    # run_subroutine(vec<client_id>) : create_epr_1
    tasks.append(TaskBuilder.CL(cl_dur, 2))
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_1"))
    # run_subroutine(vec<client_id>) : local_cphase
    tasks.append(TaskBuilder.CL(cl_dur, 3))
    tasks.append(TaskBuilder.QL(set_dur, "local_cphase", 0))
    tasks.append(TaskBuilder.QL(set_dur, "local_cphase", 1))
    tasks.append(TaskBuilder.QL(cphase_dur, "local_cphase", 2))
    # delta1 = recv_cmsg(client_id)
    tasks.append(TaskBuilder.CC(cc_dur, 4))
    # vec<m1> = run_subroutine(vec<delta1>) : meas_qubit_1
    tasks.append(TaskBuilder.CL(cl_dur, 5))
    tasks.append(TaskBuilder.QL(set_dur, "meas_qubit_1", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "meas_qubit_1", 1))
    tasks.append(TaskBuilder.QL(h_dur, "meas_qubit_1", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "meas_qubit_1", 3))
    tasks.append(TaskBuilder.QL(free_dur, "meas_qubit_1", 4))
    # send_cmsg(csocket, m1)
    tasks.append(TaskBuilder.CC(cc_dur, 6))
    # delta2 = recv_cmsg(csocket)
    tasks.append(TaskBuilder.CC(cc_dur, 7))
    # vec<m2> = run_subroutine(vec<delta2>) : meas_qubit_0
    tasks.append(TaskBuilder.CL(cl_dur, 8))
    tasks.append(TaskBuilder.QL(set_dur, "meas_qubit_0", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "meas_qubit_0", 1))
    tasks.append(TaskBuilder.QL(h_dur, "meas_qubit_0", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "meas_qubit_0", 3))
    tasks.append(TaskBuilder.QL(free_dur, "meas_qubit_0", 4))
    # return_result(m1)
    tasks.append(TaskBuilder.CL(cl_dur, 9))
    # return_result(m2)
    tasks.append(TaskBuilder.CL(cl_dur, 10))

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

    tasks.append(TaskBuilder.CL(cl_dur, 0))
    tasks.append(TaskBuilder.CL(cl_dur, 1))
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_0"))
    tasks.append(TaskBuilder.CL(cl_dur, 2))
    tasks.append(TaskBuilder.QL(set_dur, "post_epr_0", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "post_epr_0", 1))
    tasks.append(TaskBuilder.QL(h_dur, "post_epr_0", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "post_epr_0", 3))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_0", 4))

    tasks.append(TaskBuilder.CL(cl_dur, 3))
    tasks.append(TaskBuilder.QC(qc_dur, "create_epr_1"))
    tasks.append(TaskBuilder.CL(cl_dur, 4))
    tasks.append(TaskBuilder.QL(set_dur, "post_epr_1", 0))
    tasks.append(TaskBuilder.QL(rot_dur, "post_epr_1", 1))
    tasks.append(TaskBuilder.QL(h_dur, "post_epr_1", 2))
    tasks.append(TaskBuilder.QL(meas_dur, "post_epr_1", 3))
    tasks.append(TaskBuilder.QL(free_dur, "post_epr_1", 4))

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
    client_results: List[Dict[int, BatchResult]]
    server_results: Dict[int, BatchResult]


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
    path = os.path.join(os.path.dirname(__file__), "bqc_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    program = IqoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program() -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), "bqc_client.iqoala")
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


def run_bqc(alpha, beta, theta1, theta2, num_iterations: int, num_clients: int):
    # server needs to have 2 qubits per client
    server_num_qubits = num_clients * 2
    server_config = get_server_config(id=0, num_qubits=server_num_qubits)
    server_config.qdevice_cfg.T1 = 1000
    server_config.qdevice_cfg.T2 = 1000
    server_config.qdevice_cfg.two_qubit_gate_time = 1e5
    client_configs = [get_client_config(i) for i in range(1, num_clients + 1)]

    network = create_network(server_config, client_configs, num_clients)

    server_procnode = network.nodes["server"]

    for client_id in range(1, num_clients + 1):
        server_inputs = [
            ProgramInput({"client_id": client_id}) for _ in range(num_iterations)
        ]

        server_batch_info = create_server_batch(
            client_id=client_id,
            inputs=server_inputs,
            num_iterations=num_iterations,
            deadline=0,
            num_qubits=server_num_qubits,
        )

        server_procnode.submit_batch(server_batch_info)
    server_procnode.initialize_processes()
    # server_procnode.initialize_schedule(NoTimeSolver)
    server_procnode.initialize_schedule(NaiveSolver)

    for client_id in range(1, num_clients + 1):
        client_inputs = [
            ProgramInput(
                {
                    "server_id": 0,
                    "alpha": alpha,
                    "beta": beta,
                    "theta1": theta1,
                    "theta2": theta2,
                }
            )
            for _ in range(num_iterations)
        ]

        client_batch_info = create_client_batch(
            client_inputs, num_iterations, deadline=0
        )

        client_procnode = network.nodes[f"client_{client_id}"]
        client_procnode.submit_batch(client_batch_info)
        client_procnode.initialize_processes()
        client_procnode.initialize_schedule(NoTimeSolver)

    network.start_all_nodes()
    ns.sim_run()

    client_results: List[Dict[int, BatchResult]]
    client_results = client_procnode.scheduler.get_batch_results()

    server_results: Dict[int, BatchResult]
    server_results = server_procnode.scheduler.get_batch_results()

    return BqcResult(client_results, server_results)


def expected_rsp_qubit(theta: int, p: int, dummy: bool):
    expected = qubitapi.create_qubits(1)[0]

    if dummy:
        if p == 0:
            qubitapi.assign_qstate(expected, ketstates.s0)
        elif p == 1:
            qubitapi.assign_qstate(expected, ketstates.s1)
    else:
        if (theta, p) == (0, 0):
            qubitapi.assign_qstate(expected, ketstates.h0)
        elif (theta, p) == (0, 1):
            qubitapi.assign_qstate(expected, ketstates.h1)
        if (theta, p) == (8, 0):
            qubitapi.assign_qstate(expected, ketstates.y0)
        elif (theta, p) == (8, 1):
            qubitapi.assign_qstate(expected, ketstates.y1)
        if (theta, p) == (16, 0):
            qubitapi.assign_qstate(expected, ketstates.h1)
        elif (theta, p) == (16, 1):
            qubitapi.assign_qstate(expected, ketstates.h0)
        if (theta, p) == (-8, 0):
            qubitapi.assign_qstate(expected, ketstates.y1)
        elif (theta, p) == (-8, 1):
            qubitapi.assign_qstate(expected, ketstates.y0)

    return expected


def expected_rsp_state(theta: int, p: int, dummy: bool):
    expected = qubitapi.create_qubits(1)[0]

    if dummy:
        if p == 0:
            return ketstates.s0
        elif p == 1:
            return ketstates.s1
    else:
        if (theta, p) == (0, 0):
            return ketstates.h0
        elif (theta, p) == (0, 1):
            return ketstates.h1
        if (theta, p) == (8, 0):
            return ketstates.y0
        elif (theta, p) == (8, 1):
            return ketstates.y1
        if (theta, p) == (16, 0):
            return ketstates.h1
        elif (theta, p) == (16, 1):
            return ketstates.h0
        if (theta, p) == (-8, 0):
            return ketstates.y1
        elif (theta, p) == (-8, 1):
            return ketstates.y0

    return expected.qstate


def test_bqc():

    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    LogManager.set_log_level("DEBUG")
    LogManager.log_to_file("example_bqc.log")

    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    num_clients = 1

    def check(alpha, beta, theta1, theta2, expected):
        ns.sim_reset()
        bqc_result = run_bqc(
            alpha=alpha,
            beta=beta,
            theta1=theta1,
            theta2=theta2,
            num_iterations=5,
            num_clients=num_clients,
        )

        server_batch_results = bqc_result.server_results
        for batch_id, batch_results in server_batch_results.items():
            program_results = batch_results.results
            m2s = [result.values["m2"] for result in program_results]
            print(f"checking batch_id {batch_id}...")
            print(f"# correct outcomes: {len([m2 for m2 in m2s if m2==expected])}")
            assert all(m2 == expected for m2 in m2s)
            print("OK!")

    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1)


if __name__ == "__main__":
    test_bqc()
