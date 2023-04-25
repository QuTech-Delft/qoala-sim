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

    assert task_graph.tasks == {
        0: BlockTask(0, pid, "b0", CL),
        1: BlockTask(1, pid, "b1", QC),
        2: BlockTask(2, pid, "b2", QL),
        3: BlockTask(3, pid, "b3", QC),
        4: BlockTask(4, pid, "b4", QL),
        5: BlockTask(5, pid, "b5", CL),
        6: BlockTask(6, pid, "b6", CC),
        7: BlockTask(7, pid, "b7", CL),
        8: BlockTask(8, pid, "b8", CC),
        9: BlockTask(9, pid, "b9", CL),
    }

    assert task_graph.precedences == [(i - 1, i) for i in range(1, 10)]


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

    assert task_graph.tasks == {
        0: BlockTask(0, pid, "blk_host0", CL, 2 * cpu_time),
        1: BlockTask(1, pid, "blk_host1", CL, 1 * cpu_time),
        2: BlockTask(2, pid, "blk_host2", CL, 1 * cpu_time),
        3: BlockTask(3, pid, "blk_prep_cc", CL, 2 * cpu_time),
        4: BlockTask(4, pid, "blk_send", CL, 1 * cpu_time),
        5: BlockTask(5, pid, "blk_recv", CC, 1 * recv_time),
        6: BlockTask(6, pid, "blk_add_one", QL, 5 * qpu_time),
        7: BlockTask(7, pid, "blk_epr_md_1", QC, 1 * epr_time),
        8: BlockTask(8, pid, "blk_epr_md_2", QC, 2 * epr_time),
        9: BlockTask(9, pid, "blk_epr_ck_1", QC, 1 * epr_time),
        10: BlockTask(10, pid, "blk_meas_q0", QL, 3 * qpu_time + 1 * meas_time),
        11: BlockTask(11, pid, "blk_epr_ck_2", QC, 2 * epr_time),
        12: BlockTask(12, pid, "blk_meas_q0_q1", QL, 6 * qpu_time + 2 * meas_time),
    }

    assert task_graph.precedences == [(i - 1, i) for i in range(1, 13)]


def test_routine_split_1_pair_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_1_pair.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_SPLIT)
    pid = 3
    task_graph = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    assert task_graph.tasks == {
        # blk_1_pair_wait_all
        0: PreCallTask(0, pid, "blk_1_pair_wait_all"),
        1: PostCallTask(1, pid, "blk_1_pair_wait_all", None),
        2: MultiPairTask(2, pid, None),
        3: MultiPairCallbackTask(3, pid, "meas_1_pair", None),
        # blk_1_pair_sequential
        4: PreCallTask(4, pid, "blk_1_pair_sequential"),
        5: PostCallTask(5, pid, "blk_1_pair_sequential", None),
        6: SinglePairTask(6, pid, 0, None),
        7: SinglePairCallbackTask(7, pid, "meas_1_pair", 0, None),
    }

    assert (0, 2) in task_graph.precedences  # rr after precall
    assert (2, 3) in task_graph.precedences  # callback after rr
    assert (3, 1) in task_graph.precedences  # postcall after callback

    assert (1, 4) in task_graph.precedences  # second block after first block
    assert (4, 6) in task_graph.precedences  # rr after precall
    assert (6, 7) in task_graph.precedences  # callback after rr
    assert (7, 5) in task_graph.precedences  # postcall after callback


def test_routine_split_2_pairs_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_2_pairs.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_SPLIT)
    pid = 3
    task_graph = creator.from_program(program, pid, alice.local_ehi, alice.network_ehi)

    assert task_graph.tasks == {
        # blk_2_pairs_wait_all
        0: PreCallTask(0, pid, "blk_2_pairs_wait_all"),
        1: PostCallTask(1, pid, "blk_2_pairs_wait_all", None),
        2: MultiPairTask(2, pid, None),
        3: MultiPairCallbackTask(3, pid, "meas_2_pairs", None),
        # blk_2_pairs_sequential
        4: PreCallTask(4, pid, "blk_2_pairs_sequential"),
        5: PostCallTask(5, pid, "blk_2_pairs_sequential", None),
        6: SinglePairTask(6, pid, 0, None),
        7: SinglePairCallbackTask(7, pid, "meas_1_pair", 0, None),
        8: SinglePairTask(8, pid, 1, None),
        9: SinglePairCallbackTask(9, pid, "meas_1_pair", 1, None),
    }

    assert (0, 2) in task_graph.precedences  # rr after precall
    assert (2, 3) in task_graph.precedences  # callback after rr
    assert (3, 1) in task_graph.precedences  # postcall after callback

    assert (1, 4) in task_graph.precedences  # second block after first block
    assert (4, 6) in task_graph.precedences  # 1st pair after precall
    assert (6, 7) in task_graph.precedences  # 1st pair callback after 1st pair rr
    assert (4, 8) in task_graph.precedences  # 2nd pair after precall
    assert (8, 9) in task_graph.precedences  # 2nd pair callback after 2nd pair rr
    assert (7, 5) in task_graph.precedences  # postcall after 1st pair callback
    assert (9, 5) in task_graph.precedences  # postcall after 2nd pair callback

    assert task_graph.roots() == [0]


if __name__ == "__main__":
    test_from_program_1()
    test_from_program_2()
    test_routine_split_1_pair_callback()
    test_routine_split_2_pairs_callback()
