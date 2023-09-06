import os

from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.ntf import GenericNtf
from qoala.runtime.task import (
    HostLocalTask,
    MultiPairCallbackTask,
    MultiPairTask,
    PostCallTask,
    PreCallTask,
    SinglePairCallbackTask,
    SinglePairTask,
    TaskDurationEstimator,
    TaskGraph,
    TaskGraphBuilder,
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
    nodes = {0: "alice", 1: "bob"}
    network_lhi = LhiNetworkInfo.fully_connected(nodes, link_info)
    bob_lhi = LhiProcNodeInfo(name="bob", id=1, topology=topology, latencies=latencies)
    ntfs = [GenericNtf(), GenericNtf()]
    return build_network_from_lhi([alice_lhi, bob_lhi], ntfs, network_lhi)


def test_qoala_tasks_1_pair_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_1_pair.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    cb = program.local_routines["meas_1_pair"]

    cpu_time = alice.local_ehi.latencies.host_instr_time
    cb_time = TaskDurationEstimator.lr_duration(alice.local_ehi, cb)
    pair_time = alice.network_ehi.get_link(0, 1).duration

    pid = 3
    task_graph = TaskGraphBuilder.from_program(
        program, pid, alice.local_ehi, alice.network_ehi
    )

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


def test_qoala_tasks_2_pairs_callback():
    network = setup_network()
    alice = network.nodes["alice"]

    path = relative_path("test_callbacks_2_pairs.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    cb1 = program.local_routines["meas_1_pair"]
    cb2 = program.local_routines["meas_2_pairs"]
    cpu_time = alice.local_ehi.latencies.host_instr_time

    cb1_time = TaskDurationEstimator.lr_duration(alice.local_ehi, cb1)
    cb2_time = TaskDurationEstimator.lr_duration(alice.local_ehi, cb2)
    pair_time = alice.network_ehi.get_link(0, 1).duration

    pid = 3
    task_graph = TaskGraphBuilder.from_program(
        program, pid, alice.local_ehi, alice.network_ehi
    )

    expected_tasks = [
        # blk_2_pairs_wait_all
        PreCallTask(0, pid, "blk_2_pairs_wait_all", 0, cpu_time),
        PostCallTask(1, pid, "blk_2_pairs_wait_all", 0, cpu_time),
        MultiPairTask(2, pid, 0, 2 * pair_time),
        MultiPairCallbackTask(3, pid, "meas_2_pairs", 0, cb2_time),
        # blk_2_pairs_sequential
        PreCallTask(4, pid, "blk_2_pairs_sequential", 4, cpu_time),
        PostCallTask(5, pid, "blk_2_pairs_sequential", 4, cpu_time),
        SinglePairTask(6, pid, 0, 4, pair_time),
        SinglePairCallbackTask(7, pid, "meas_1_pair", 0, 4, cb1_time),
        SinglePairTask(8, pid, 1, 4, pair_time),
        SinglePairCallbackTask(9, pid, "meas_1_pair", 1, 4, cb1_time),
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


def test_deadlines():
    path = relative_path("test_deadlines.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    pid = 3
    task_graph = TaskGraphBuilder.from_program(program, pid)

    expected_tasks = [
        HostLocalTask(0, pid, "b0"),
        HostLocalTask(1, pid, "b1"),
    ]
    expected_precedences = [(0, 1)]
    expected_deadlines = [((0, 1), 100)]

    expected_graph = TaskGraph()
    expected_graph.add_tasks(expected_tasks)
    expected_graph.add_precedences(expected_precedences)
    expected_graph.add_rel_deadlines(expected_deadlines)

    assert task_graph == expected_graph


if __name__ == "__main__":
    test_qoala_tasks_1_pair_callback()
    test_qoala_tasks_2_pairs_callback()
    test_deadlines()
