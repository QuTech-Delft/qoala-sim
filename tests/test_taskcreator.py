import os

from netqasm.lang.instr import core

from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.task import (
    BlockTask,
    MultiPairCallbackTask,
    MultiPairTask,
    PostCallTask,
    PreCallTask,
    SinglePairCallbackTask,
    SinglePairTask,
    TaskCreator,
    TaskExecutionMode,
    TaskGraph,
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
    network_lhi = LhiNetworkInfo.fully_connected([0, 1], link_info)
    network_info = NetworkInfo.with_nodes({0: "alice", 1: "bob"})
    bob_lhi = LhiProcNodeInfo(name="bob", id=1, topology=topology, latencies=latencies)
    return build_network_from_lhi([alice_lhi, bob_lhi], network_info, network_lhi)


def test_from_program_1():
    path = relative_path("integration/bqc/vbqc_client.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    task_graph = creator.from_program(program, pid)

    expected_tasks = [
        BlockTask(0, pid, "b0", CL),
        BlockTask(1, pid, "b1", QC),
        BlockTask(2, pid, "b2", QL),
        BlockTask(3, pid, "b3", QC),
        BlockTask(4, pid, "b4", QL),
        BlockTask(5, pid, "b5", CL),
        BlockTask(6, pid, "b6", CC),
        BlockTask(7, pid, "b7", CL),
        BlockTask(8, pid, "b8", CC),
        BlockTask(9, pid, "b9", CL),
    ]
    expected_precedences = [(i - 1, i) for i in range(1, 10)]

    expected_graph = TaskGraph()
    expected_graph.add_tasks(expected_tasks)
    expected_graph.add_precedences(expected_precedences)

    assert task_graph == expected_graph


def test_from_program_2():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_scheduling_alice.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    task_graph = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    cpu_time = alice.local_ehi.latencies.host_instr_time
    recv_time = alice.local_ehi.latencies.host_peer_latency
    qpu_time = alice.local_ehi.latencies.qnos_instr_time
    meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    epr_time = alice.network_ehi.get_link(0, 1).duration

    expected_tasks = [
        BlockTask(0, pid, "blk_host0", CL, 2 * cpu_time),
        BlockTask(1, pid, "blk_host1", CL, 1 * cpu_time),
        BlockTask(2, pid, "blk_host2", CL, 1 * cpu_time),
        BlockTask(3, pid, "blk_prep_cc", CL, 2 * cpu_time),
        BlockTask(4, pid, "blk_send", CL, 1 * cpu_time),
        BlockTask(5, pid, "blk_recv", CC, 1 * recv_time),
        BlockTask(6, pid, "blk_add_one", QL, 5 * qpu_time),
        BlockTask(7, pid, "blk_epr_md_1", QC, 1 * epr_time),
        BlockTask(8, pid, "blk_epr_md_2", QC, 2 * epr_time),
        BlockTask(9, pid, "blk_epr_ck_1", QC, 1 * epr_time),
        BlockTask(10, pid, "blk_meas_q0", QL, 3 * qpu_time + 1 * meas_time),
        BlockTask(11, pid, "blk_epr_ck_2", QC, 2 * epr_time),
        BlockTask(12, pid, "blk_meas_q0_q1", QL, 6 * qpu_time + 2 * meas_time),
    ]
    expected_precedences = [(i - 1, i) for i in range(1, 13)]

    expected_graph = TaskGraph()
    expected_graph.add_tasks(expected_tasks)
    expected_graph.add_precedences(expected_precedences)

    assert task_graph == expected_graph


def test_routine_split_1_pair_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_1_pair.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    cpu_time = alice.local_ehi.latencies.host_instr_time
    cb_time = alice.local_ehi.latencies.qnos_instr_time
    pair_time = alice.network_ehi.get_link(0, 1).duration

    creator = TaskCreator(TaskExecutionMode.ROUTINE_SPLIT)
    pid = 3
    task_graph = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    expected_tasks = [
        # blk_1_pair_wait_all
        PreCallTask(0, pid, "blk_1_pair_wait_all", 0, cpu_time),
        PostCallTask(1, pid, "blk_1_pair_wait_all", 0, cpu_time),
        MultiPairTask(2, pid, 0, pair_time),
        MultiPairCallbackTask(3, pid, "meas_1_pair", 0, cb_time),
        # blk_1_pair_sequential
        PreCallTask(4, pid, "blk_1_pair_sequential", 4, cpu_time),
        PostCallTask(5, pid, "blk_1_pair_sequential", 4, cpu_time),
        SinglePairTask(6, pid, 0, 4, pair_time),
        SinglePairCallbackTask(7, pid, "meas_1_pair", 0, 4, cb_time),
    ]

    expected_precedences = [
        (0, 2),  # rr after precall
        (2, 3),  # callback after rr
        (3, 1),  # postcall after callback
        (1, 4),  # second block after first block
        (4, 6),  # rr after precall
        (6, 7),  # callback after rr
        (7, 5),  # postcall after callback
    ]

    expected_graph = TaskGraph()
    expected_graph.add_tasks(expected_tasks)
    expected_graph.add_precedences(expected_precedences)

    assert task_graph == expected_graph


def test_routine_split_2_pairs_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_2_pairs.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    cpu_time = alice.local_ehi.latencies.host_instr_time
    cb_time = alice.local_ehi.latencies.qnos_instr_time
    pair_time = alice.network_ehi.get_link(0, 1).duration

    creator = TaskCreator(TaskExecutionMode.ROUTINE_SPLIT)
    pid = 3
    task_graph = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    expected_tasks = [
        # blk_2_pairs_wait_all
        PreCallTask(0, pid, "blk_2_pairs_wait_all", 0, cpu_time),
        PostCallTask(1, pid, "blk_2_pairs_wait_all", 0, cpu_time),
        MultiPairTask(2, pid, 0, 2 * pair_time),
        MultiPairCallbackTask(3, pid, "meas_2_pairs", 0, cb_time),
        # blk_2_pairs_sequential
        PreCallTask(4, pid, "blk_2_pairs_sequential", 4, cpu_time),
        PostCallTask(5, pid, "blk_2_pairs_sequential", 4, cpu_time),
        SinglePairTask(6, pid, 0, 4, pair_time),
        SinglePairCallbackTask(7, pid, "meas_1_pair", 0, 4, cb_time),
        SinglePairTask(8, pid, 1, 4, pair_time),
        SinglePairCallbackTask(9, pid, "meas_1_pair", 1, 4, cb_time),
    ]

    expected_precedences = [
        (0, 2),  # rr after precall
        (2, 3),  # callback after rr
        (3, 1),  # postcall after callback
        (1, 4),  # second block after first block
        (4, 6),  # 1st pair after precall
        (6, 7),  # 1st pair callback after 1st pair rr
        (4, 8),  # 2nd pair after precall
        (8, 9),  # 2nd pair callback after 2nd pair rr
        (7, 5),  # postcall after 1st pair callback
        (9, 5),  # postcall after 2nd pair callback
    ]

    expected_graph = TaskGraph()
    expected_graph.add_tasks(expected_tasks)
    expected_graph.add_precedences(expected_precedences)

    assert task_graph == expected_graph


if __name__ == "__main__":
    test_from_program_1()
    test_from_program_2()
    test_routine_split_1_pair_callback()
    test_routine_split_2_pairs_callback()
