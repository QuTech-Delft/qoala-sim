from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.lhi import LhiLatencies, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import (
    ProgramTaskList,
    Schedule,
    ScheduleEntry,
    ScheduleTime,
    TaskBuilder,
)
from qoala.sim.build import build_network, build_qprocessor_from_topology
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.process import IqoalaProcess
from qoala.sim.procnode import ProcNode


def create_process(
    pid: int,
    program: IqoalaProgram,
    unit_module: UnitModule,
    host_interface: HostInterface,
    inputs: Dict[str, Any],
    tasks: ProgramTaskList,
) -> IqoalaProcess:
    prog_input = ProgramInput(values=inputs)
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=prog_input,
        tasks=tasks,
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=0)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={
            id: ClassicalSocket(host_interface, name)
            for (id, name) in program.meta.csockets.items()
        },
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def create_global_env(names: List[str]) -> GlobalEnvironment:
    env = GlobalEnvironment()
    for i, name in enumerate(names):
        env.add_node(i, GlobalNodeInfo(name, i))
    return env


def create_server_tasks(server_program: IqoalaProgram) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    ql_dur = 1e3
    qc_dur = 1e6

    tasks.append(TaskBuilder.CL(cl_dur, 0))
    tasks.append(TaskBuilder.QC(qc_dur, 1, "req0"))
    tasks.append(TaskBuilder.QC(qc_dur, 2, "req1"))
    dur = cl_dur + 3 * ql_dur
    tasks.append(TaskBuilder.QL(dur, 3, "local_cphase"))
    tasks.append(TaskBuilder.CC(cc_dur, 4))
    dur = cl_dur + 5 * ql_dur
    tasks.append(TaskBuilder.QL(dur, 5, "meas_qubit_1"))
    tasks.append(TaskBuilder.CC(cc_dur, 6))
    tasks.append(TaskBuilder.CC(cc_dur, 7))
    dur = cl_dur + 5 * ql_dur
    tasks.append(TaskBuilder.QL(dur, 8, "meas_qubit_0"))
    tasks.append(TaskBuilder.CL(cl_dur, 9))
    tasks.append(TaskBuilder.CL(cl_dur, 10))

    return ProgramTaskList(server_program, {i: task for i, task in enumerate(tasks)})


def create_client_tasks(client_program: IqoalaProgram) -> ProgramTaskList:
    tasks = []

    cl_dur = 1e3
    cc_dur = 10e6
    ql_dur = 1e3
    qc_dur = 1e6

    tasks.append(TaskBuilder.CL(cl_dur, 0))
    tasks.append(TaskBuilder.QC(qc_dur, 1, "req0"))
    dur = cl_dur + 5 * ql_dur
    tasks.append(TaskBuilder.QL(dur, 2, "post_epr_0"))
    tasks.append(TaskBuilder.QC(qc_dur, 3, "req1"))
    dur = cl_dur + 5 * ql_dur
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


def create_server_schedule(num_tasks: int) -> Schedule:
    entries = [
        (ScheduleTime(int(i * 1e8)), ScheduleEntry(pid=0, task_index=i))
        for i in range(num_tasks)
    ]
    return Schedule(entries)


def create_client_schedule(num_tasks: int) -> Schedule:
    entries = [
        (ScheduleTime(None), ScheduleEntry(pid=0, task_index=i))
        for i in range(num_tasks)
    ]
    return Schedule(entries)


def load_program(path: str) -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return IqoalaParser(text).parse()


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
    )


@dataclass
class BqcResult:
    client_process: IqoalaProcess
    server_process: IqoalaProcess
    client_procnode: ProcNode
    server_procnode: ProcNode


def run_bqc(alpha, beta, theta1, theta2):
    ns.sim_reset()

    num_qubits = 3
    global_env = create_global_env(names=["client", "server"])
    server_id = global_env.get_node_id("server")
    client_id = global_env.get_node_id("client")

    server_program = load_program("test_bqc_server.iqoala")
    server_tasks = create_server_tasks(server_program)

    server_node_cfg = create_procnode_cfg("server", server_id, num_qubits)
    client_node_cfg = create_procnode_cfg("client", client_id, num_qubits)
    network_cfg = ProcNodeNetworkConfig(
        nodes=[server_node_cfg, client_node_cfg], links=[]
    )
    network = build_network(network_cfg, global_env)
    server_procnode = network.nodes["server"]
    client_procnode = network.nodes["client"]

    server_ehi = server_procnode.memmgr.get_ehi()
    server_process = create_process(
        pid=0,
        program=server_program,
        unit_module=UnitModule.from_full_ehi(server_ehi),
        host_interface=server_procnode.host._interface,
        inputs={"client_id": client_id},
        tasks=server_tasks,
    )
    server_procnode.add_process(server_process)
    server_procnode.scheduler.initialize_process(server_process)

    server_schedule = create_server_schedule(len(server_tasks.tasks))
    server_procnode.install_schedule(server_schedule)

    client_program = load_program("test_bqc_client.iqoala")
    client_tasks = create_client_tasks(client_program)

    client_ehi = client_procnode.memmgr.get_ehi()
    client_process = create_process(
        pid=0,
        program=client_program,
        unit_module=UnitModule.from_full_ehi(client_ehi),
        host_interface=client_procnode.host._interface,
        inputs={
            "server_id": server_id,
            "alpha": alpha,
            "beta": beta,
            "theta1": theta1,
            "theta2": theta2,
        },
        tasks=client_tasks,
    )
    client_procnode.add_process(client_process)
    client_procnode.scheduler.initialize_process(client_process)

    client_schedule = create_client_schedule(len(client_tasks.tasks))
    client_procnode.install_schedule(client_schedule)

    network.start()
    ns.sim_run()

    return BqcResult(
        client_process=client_process,
        server_process=server_process,
        client_procnode=client_procnode,
        server_procnode=server_procnode,
    )


def test_bqc():
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    def check(alpha, beta, theta1, theta2, expected):
        ns.sim_reset()
        results = [
            run_bqc(
                alpha=alpha,
                beta=beta,
                theta1=theta1,
                theta2=theta2,
            )
            for _ in range(1)
        ]
        assert all(len(result.client_process.result.values) > 0 for result in results)
        assert all(len(result.server_process.result.values) > 0 for result in results)
        m2s = [result.server_process.result.values["m2"] for result in results]
        assert all(m2 == expected for m2 in m2s)

    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1)


if __name__ == "__main__":
    test_bqc()
