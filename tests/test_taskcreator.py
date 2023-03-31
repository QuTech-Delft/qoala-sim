import os

from netqasm.lang.instr import core

from qoala.lang.hostlang import BasicBlockType
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
    BlockTask,
    LinkSlotInfo,
    QcSlotInfo,
    TaskCreator,
    TaskExecutionMode,
    TaskSchedule,
    TaskScheduleEntry,
)
from qoala.sim.build import build_network_from_lhi
from qoala.sim.network import ProcNodeNetwork


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


CL = BasicBlockType.CL
CC = BasicBlockType.CC
QL = BasicBlockType.QL
QC = BasicBlockType.QC


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


def test_from_program_1():
    path = relative_path("integration/bqc/vbqc_client.iqoala")
    with open(path) as file:
        text = file.read()
    program = IqoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    tasks = creator.from_program(program, pid)

    assert tasks == [
        BlockTask(pid, "b0", CL),
        BlockTask(pid, "b1", QC),
        BlockTask(pid, "b2", QL),
        BlockTask(pid, "b3", QC),
        BlockTask(pid, "b4", QL),
        BlockTask(pid, "b5", CL),
        BlockTask(pid, "b6", CC),
        BlockTask(pid, "b7", CL),
        BlockTask(pid, "b8", CC),
        BlockTask(pid, "b9", CL),
    ]


def test_from_program_2():
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
    recv_time = alice.local_ehi.latencies.host_peer_latency
    qpu_time = alice.local_ehi.latencies.qnos_instr_time
    meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    epr_time = alice.network_ehi.get_link(0, 1).duration

    assert tasks == [
        BlockTask(pid, "blk_host0", CL, 2 * cpu_time),
        BlockTask(pid, "blk_host1", CL, 1 * cpu_time),
        BlockTask(pid, "blk_host2", CL, 1 * cpu_time),
        BlockTask(pid, "blk_prep_cc", CL, 2 * cpu_time),
        BlockTask(pid, "blk_send", CL, 1 * cpu_time),
        BlockTask(pid, "blk_recv", CC, 1 * recv_time),
        BlockTask(pid, "blk_add_one", QL, 5 * qpu_time),
        BlockTask(pid, "blk_epr_md_1", QC, 1 * epr_time),
        BlockTask(pid, "blk_epr_md_2", QC, 2 * epr_time),
        BlockTask(pid, "blk_epr_ck_1", QC, 1 * epr_time),
        BlockTask(pid, "blk_meas_q0", QL, 3 * qpu_time + 1 * meas_time),
        BlockTask(pid, "blk_epr_ck_2", QC, 2 * epr_time),
        BlockTask(pid, "blk_meas_q0_q1", QL, 6 * qpu_time + 2 * meas_time),
    ]


def test_consecutive():
    pid = 0
    tasks = [
        BlockTask(pid, "blk_host0", CL, 1000),
        BlockTask(pid, "blk_recv", CC, 1000),
        BlockTask(pid, "blk_add_one", QL, 1000),
        BlockTask(pid, "blk_epr_md_1", QC, 1000),
        BlockTask(pid, "blk_host1", CL, 1000),
    ]

    schedule = TaskSchedule.consecutive(tasks)

    assert schedule.entries == [
        TaskScheduleEntry(tasks[0], None, prev=None),
        TaskScheduleEntry(tasks[1], None, prev=None),
        TaskScheduleEntry(tasks[2], None, prev=tasks[1]),  # CPU -> QPU
        TaskScheduleEntry(tasks[3], None, prev=None),
        TaskScheduleEntry(tasks[4], None, prev=tasks[3]),  # QPU -> CPU
    ]


def test_consecutive_qc_slots():
    pid = 0
    tasks = [
        BlockTask(pid, "blk_host0", CL, 1000),
        BlockTask(pid, "blk_recv", CC, 10000),
        BlockTask(pid, "blk_add_one", QL, 1000),
        BlockTask(pid, "blk_epr_md_1", QC, 1000, remote_id=0),
        BlockTask(pid, "blk_host1", CL, 1000),
    ]

    schedule = TaskSchedule.consecutive(
        tasks, qc_slot_info=QcSlotInfo({0: LinkSlotInfo(0, 100, 50_000)})
    )

    assert schedule.entries == [
        TaskScheduleEntry(tasks[0], timestamp=None, prev=None),
        TaskScheduleEntry(tasks[1], timestamp=None, prev=None),
        TaskScheduleEntry(tasks[2], timestamp=None, prev=tasks[1]),  # CPU -> QPU
        TaskScheduleEntry(tasks[3], timestamp=50_000, prev=None),  # QC task
        TaskScheduleEntry(tasks[4], timestamp=None, prev=tasks[3]),  # QPU -> CPU
    ]


def test_consecutive_timestamps():
    pid = 0

    tasks = [
        BlockTask(pid, "blk_host0", CL, 1000),
        BlockTask(pid, "blk_lr0", QL, 5000),
        BlockTask(pid, "blk_host1", CL, 200),
        BlockTask(pid, "blk_rr0", QC, 30_000, remote_id=0),
        BlockTask(pid, "blk_host2", CL, 4000),
    ]

    schedule = TaskSchedule.consecutive_timestamps(tasks)

    assert schedule.entries == [
        TaskScheduleEntry(tasks[0], 0),
        TaskScheduleEntry(tasks[1], 1000),
        TaskScheduleEntry(tasks[2], 1000 + 5000),
        TaskScheduleEntry(tasks[3], 1000 + 5000 + 200),
        TaskScheduleEntry(tasks[4], 1000 + 5000 + 200 + 30_000),
    ]

    assert schedule.cpu_schedule.entries == [
        TaskScheduleEntry(tasks[0], 0),
        TaskScheduleEntry(tasks[2], 1000 + 5000),
        TaskScheduleEntry(tasks[4], 1000 + 5000 + 200 + 30_000),
    ]
    assert schedule.qpu_schedule.entries == [
        TaskScheduleEntry(tasks[1], 1000),
        TaskScheduleEntry(tasks[3], 1000 + 5000 + 200),
    ]


if __name__ == "__main__":
    test_from_program_1()
    test_from_program_2()
    test_consecutive()
    test_consecutive_qc_slots()
    test_consecutive_timestamps()
