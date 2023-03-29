import os

from netqasm.lang.instr import core

from qoala.lang.parse import IqoalaParser
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
    NetworkLhi,
)
from qoala.runtime.taskcreator import (
    CpuQpuTaskList,
    CpuSchedule,
    CpuTask,
    QpuSchedule,
    QpuTask,
    RoutineType,
    TaskCreator,
    TaskExecutionMode,
)
from qoala.sim.build import build_network_from_lhi
from qoala.sim.network import ProcNodeNetwork


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


def test1():
    path = relative_path("integration/bqc/vbqc_client.iqoala")
    with open(path) as file:
        text = file.read()
    program = IqoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    tasks = creator.from_program(program, pid)

    assert tasks.cpu_tasks() == [CpuTask(pid, "b0"), CpuTask(pid, "b5")]
    assert tasks.qpu_tasks() == [
        QpuTask(pid, RoutineType.REQUEST, "b1"),
        QpuTask(pid, RoutineType.LOCAL, "b2"),
        QpuTask(pid, RoutineType.REQUEST, "b3"),
        QpuTask(pid, RoutineType.LOCAL, "b4"),
    ]


def setup_network() -> ProcNodeNetwork:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=3)
    latencies = LhiLatencies(
        host_instr_time=1000, qnos_instr_time=2000, host_peer_latency=3000
    )
    link_info = LhiLinkInfo.perfect(duration=20_000)

    alice_lhi = LhiProcNodeInfo(
        name="alice", id=0, topology=topology, latencies=latencies
    )
    network_lhi = NetworkLhi.fully_connected([0, 1], link_info)
    network_info = NetworkInfo.with_nodes({0: "alice", 1: "bob"})
    bob_lhi = LhiProcNodeInfo(name="bob", id=1, topology=topology, latencies=latencies)
    return build_network_from_lhi([alice_lhi, bob_lhi], network_info, network_lhi)


def test2():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_scheduling_alice.iqoala")
    with open(path) as file:
        text = file.read()
    program = IqoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    tasks = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    cpu_time = alice.local_ehi.latencies.host_instr_time

    assert tasks.cpu_tasks() == [
        CpuTask(pid, "blk_host0", 2 * cpu_time),
        CpuTask(pid, "blk_host1", 1 * cpu_time),
        CpuTask(pid, "blk_host2", 1 * cpu_time),
    ]

    qpu_time = alice.local_ehi.latencies.qnos_instr_time
    meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    epr_time = alice.network_ehi.get_link(0, 1).duration
    assert tasks.qpu_tasks() == [
        QpuTask(pid, RoutineType.LOCAL, "blk_add_one", 5 * qpu_time),
        QpuTask(pid, RoutineType.REQUEST, "blk_epr_md_1", 1 * epr_time),
        QpuTask(pid, RoutineType.REQUEST, "blk_epr_md_2", 2 * epr_time),
        QpuTask(pid, RoutineType.REQUEST, "blk_epr_ck_1", 1 * epr_time),
        QpuTask(pid, RoutineType.LOCAL, "blk_meas_q0", 3 * qpu_time + 1 * meas_time),
        QpuTask(pid, RoutineType.REQUEST, "blk_epr_ck_2", 2 * epr_time),
        QpuTask(pid, RoutineType.LOCAL, "blk_meas_q0_q1", 6 * qpu_time + 2 * meas_time),
    ]


def test_consecutive():
    pid = 0

    tasks = [
        CpuTask(pid, "blk_host0", 1000),
        QpuTask(pid, RoutineType.LOCAL, "blk_lr0", 5000),
        CpuTask(pid, "blk_host1", 200),
        QpuTask(pid, RoutineType.REQUEST, "blk_rr0", 30_000),
        CpuTask(pid, "blk_host2", 4000),
    ]

    cpu_schedule = CpuSchedule.consecutive(CpuQpuTaskList(tasks))
    qpu_schedule = QpuSchedule.consecutive(CpuQpuTaskList(tasks))

    assert cpu_schedule.tasks == [
        (0, tasks[0]),
        (1000 + 5000, tasks[2]),
        (1000 + 5000 + 200 + 30_000, tasks[4]),
    ]
    assert qpu_schedule.tasks == [
        (1000, tasks[1]),
        (1000 + 5000 + 200, tasks[3]),
    ]


if __name__ == "__main__":
    test1()
    test2()
    test_consecutive()
