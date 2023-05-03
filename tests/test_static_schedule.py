import os
from typing import List, Optional

import netsquid as ns
import pytest
from netqasm.lang.instr.flavour import core

from qoala.lang.ehi import EhiNodeInfo, UnitModule
from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.schedule import (
    LinkSlotInfo,
    QcSlotInfo,
    StaticSchedule,
    StaticScheduleEntry,
)
from qoala.runtime.task import BlockTask, TaskCreator, TaskExecutionMode, TaskGraph
from qoala.sim.build import build_network_from_lhi
from qoala.sim.driver import CpuDriver, QpuDriver, SharedSchedulerMemory
from qoala.sim.network import ProcNodeNetwork
from qoala.sim.scheduler import TimeTriggeredScheduler
from qoala.util.builder import ObjectBuilder


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


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


def instantiate(
    program: QoalaProgram,
    ehi: EhiNodeInfo,
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
        unit_module=unit_module,
        task_graph=TaskGraph(),
    )


def get_pure_host_program() -> QoalaProgram:
    program_text = """
META_START
    name: alice
    parameters:
    csockets:
    epr_sockets:
META_END

^b0 {type = CL}:
    var_x = assign_cval() : 3
    var_y = assign_cval() : 5
^b1 {type = CL}:
    var_z = assign_cval() : 9
    """

    return QoalaParser(program_text).parse()


def get_lr_program() -> QoalaProgram:
    program_text = """
META_START
    name: alice
    parameters:
    csockets:
    epr_sockets:
META_END

^b0 {type = CL}:
    x = assign_cval() : 3
^b1 {type = QL}:
    vec<y> = run_subroutine(vec<x>) : add_one

SUBROUTINE add_one
    params: x
    returns: y
    uses: 
    keeps:
    request:
  NETQASM_START
    load C0 @input[0]
    set C1 1
    add R0 C0 C1
    store R0 @output[0]
  NETQASM_END
    """

    return QoalaParser(program_text).parse()


CL = BasicBlockType.CL
CC = BasicBlockType.CC
QL = BasicBlockType.QL
QC = BasicBlockType.QC


def task_list_to_graph(task_list: List[BlockTask]) -> TaskGraph:
    tasks = {t.task_id: t for t in task_list}
    return TaskGraph.only_tasks(tasks)


def test_consecutive():
    pid = 0
    tasks = [
        BlockTask(0, pid, "blk_host0", CL, 1000),
        BlockTask(1, pid, "blk_recv", CC, 1000),
        BlockTask(2, pid, "blk_add_one", QL, 1000),
        BlockTask(3, pid, "blk_epr_md_1", QC, 1000),
        BlockTask(4, pid, "blk_host1", CL, 1000),
    ]
    graph = task_list_to_graph(tasks)
    schedule = StaticSchedule.consecutive_block_tasks([graph])

    assert schedule.entries == [
        StaticScheduleEntry(tasks[0], None, prev=None),
        StaticScheduleEntry(tasks[1], None, prev=None),
        StaticScheduleEntry(tasks[2], None, prev=[tasks[1]]),  # CPU -> QPU
        StaticScheduleEntry(tasks[3], None, prev=None),
        StaticScheduleEntry(tasks[4], None, prev=[tasks[3]]),  # QPU -> CPU
    ]


def test_consecutive_qc_slots():
    pid = 0
    tasks = [
        BlockTask(0, pid, "blk_host0", CL, 1000),
        BlockTask(1, pid, "blk_recv", CC, 10000),
        BlockTask(2, pid, "blk_add_one", QL, 1000),
        BlockTask(3, pid, "blk_epr_md_1", QC, 1000, remote_id=0),
        BlockTask(4, pid, "blk_host1", CL, 1000),
    ]
    graph = task_list_to_graph(tasks)
    schedule = StaticSchedule.consecutive_block_tasks(
        [graph], qc_slot_info=QcSlotInfo({0: LinkSlotInfo(0, 100, 50_000)})
    )

    assert schedule.entries == [
        StaticScheduleEntry(tasks[0], timestamp=None, prev=None),
        StaticScheduleEntry(tasks[1], timestamp=None, prev=None),
        StaticScheduleEntry(tasks[2], timestamp=None, prev=[tasks[1]]),  # CPU -> QPU
        StaticScheduleEntry(tasks[3], timestamp=50_000, prev=None),  # QC task
        StaticScheduleEntry(tasks[4], timestamp=None, prev=[tasks[3]]),  # QPU -> CPU
    ]


def test_consecutive_block_tasks_with_timestamps():
    pid = 0

    tasks = [
        BlockTask(0, pid, "blk_host0", CL, 1000),
        BlockTask(1, pid, "blk_lr0", QL, 5000),
        BlockTask(2, pid, "blk_host1", CL, 200),
        BlockTask(3, pid, "blk_rr0", QC, 30_000, remote_id=0),
        BlockTask(4, pid, "blk_host2", CL, 4000),
    ]
    graph = task_list_to_graph(tasks)
    schedule = StaticSchedule.consecutive_block_tasks_with_timestamps([graph])

    assert schedule.entries == [
        StaticScheduleEntry(tasks[0], 0),
        StaticScheduleEntry(tasks[1], 1000),
        StaticScheduleEntry(tasks[2], 1000 + 5000),
        StaticScheduleEntry(tasks[3], 1000 + 5000 + 200),
        StaticScheduleEntry(tasks[4], 1000 + 5000 + 200 + 30_000),
    ]

    assert schedule.cpu_schedule.entries == [
        StaticScheduleEntry(tasks[0], 0),
        StaticScheduleEntry(tasks[2], 1000 + 5000),
        StaticScheduleEntry(tasks[4], 1000 + 5000 + 200 + 30_000),
    ]
    assert schedule.qpu_schedule.entries == [
        StaticScheduleEntry(tasks[1], 1000),
        StaticScheduleEntry(tasks[3], 1000 + 5000 + 200),
    ]


def test_cpu_driver():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    cpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, 0, "b0", CL), 0),
            StaticScheduleEntry(BlockTask(1, 0, "b1", CL), 1000),
        ]
    )

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
    scheduler.upload_schedule(cpu_schedule)

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid).host_mem.read("var_y") == 5
    assert procnode.memmgr.get_process(pid).host_mem.read("var_z") == 9

    assert ns.sim_time() == 1000


def test_cpu_driver_no_time():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    task_graph = TaskGraph.only_tasks(
        {0: BlockTask(0, 0, "b0", CL), 1: BlockTask(1, 0, "b1", CL)}
    )
    cpu_schedule = StaticSchedule.consecutive_block_tasks([task_graph])

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
    scheduler.upload_schedule(cpu_schedule)

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid).host_mem.read("var_y") == 5
    assert procnode.memmgr.get_process(pid).host_mem.read("var_z") == 9

    assert ns.sim_time() == 0


def test_cpu_driver_2_processes():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid0 = 0
    pid1 = 1
    instance0 = ObjectBuilder.simple_program_instance(program, pid0)
    instance1 = ObjectBuilder.simple_program_instance(program, pid1)

    procnode.scheduler.submit_program_instance(instance0)
    procnode.scheduler.submit_program_instance(instance1)

    cpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid0, "b0", CL), 0),
            StaticScheduleEntry(BlockTask(1, pid1, "b0", CL), 500),
            StaticScheduleEntry(BlockTask(2, pid0, "b1", CL), 1000),
            StaticScheduleEntry(BlockTask(3, pid1, "b1", CL), 1500),
        ]
    )

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
    scheduler.upload_schedule(cpu_schedule)

    ns.sim_reset()
    scheduler.start()
    ns.sim_run(duration=1000)

    assert procnode.memmgr.get_process(pid0).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid0).host_mem.read("var_y") == 5
    with pytest.raises(KeyError):
        procnode.memmgr.get_process(pid0).host_mem.read("var_z")
        procnode.memmgr.get_process(pid1).host_mem.read("var_z")

    ns.sim_run()
    assert procnode.memmgr.get_process(pid0).host_mem.read("var_z") == 9
    assert procnode.memmgr.get_process(pid1).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid1).host_mem.read("var_y") == 5
    assert procnode.memmgr.get_process(pid1).host_mem.read("var_z") == 9

    assert ns.sim_time() == 1500


def test_qpu_driver():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_lr_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    cpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, 0, "b0", CL), 0),
        ]
    )
    qpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(1, 0, "b1", QL), 1000),
        ]
    )

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = TimeTriggeredScheduler("alice", cpu_driver)
    cpu_scheduler.upload_schedule(cpu_schedule)

    mem = SharedSchedulerMemory()
    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = TimeTriggeredScheduler("alice", qpu_driver)
    qpu_scheduler.upload_schedule(qpu_schedule)

    ns.sim_reset()
    cpu_scheduler.start()
    qpu_scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("y") == 4

    assert ns.sim_time() == 1000


def test_qpu_driver_2_processes():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_lr_program()

    pid0 = 0
    pid1 = 1
    instance0 = ObjectBuilder.simple_program_instance(program, pid0)
    instance1 = ObjectBuilder.simple_program_instance(program, pid1)

    procnode.scheduler.submit_program_instance(instance0)
    procnode.scheduler.submit_program_instance(instance1)

    cpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid0, "b0", CL), 0),
            StaticScheduleEntry(BlockTask(1, pid1, "b0", CL), 500),
        ]
    )
    qpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid0, "b1", QL), 500),
            StaticScheduleEntry(BlockTask(1, pid1, "b1", QL), 1000),
        ]
    )

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = TimeTriggeredScheduler("alice", cpu_driver)
    cpu_scheduler.upload_schedule(cpu_schedule)

    mem = SharedSchedulerMemory()
    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = TimeTriggeredScheduler("alice", qpu_driver)
    qpu_scheduler.upload_schedule(qpu_schedule)

    ns.sim_reset()
    cpu_scheduler.start()
    qpu_scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid0).host_mem.read("y") == 4
    assert procnode.memmgr.get_process(pid1).host_mem.read("y") == 4

    assert ns.sim_time() == 1000


def test_host_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = load_program("test_scheduling_alice.iqoala")
    pid = 0
    instance = instantiate(program, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance)
    bob.scheduler.submit_program_instance(instance)

    tasks = [
        BlockTask(0, pid, "blk_host0", CL),
        BlockTask(1, pid, "blk_host1", CL),
    ]
    graph = task_list_to_graph(tasks)

    cpu_schedule = StaticSchedule.consecutive_block_tasks([graph])
    alice.scheduler.upload_cpu_schedule(cpu_schedule)
    bob.scheduler.upload_cpu_schedule(cpu_schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 3 * alice.local_ehi.latencies.host_instr_time
    alice.memmgr.get_process(pid).host_mem.read("var_z") == 9
    bob.memmgr.get_process(pid).host_mem.read("var_z") == 9


def test_lr_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = load_program("test_scheduling_alice.iqoala")
    pid = 0
    instance = instantiate(program, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance)
    bob.scheduler.submit_program_instance(instance)

    host_instr_time = alice.local_ehi.latencies.host_instr_time

    tasks = [
        BlockTask(0, pid, "blk_host2", CL),
        BlockTask(1, pid, "blk_add_one", QL),
    ]
    graph = task_list_to_graph(tasks)

    schedule = StaticSchedule.consecutive_block_tasks([graph])
    alice.scheduler.upload_schedule(schedule)
    bob.scheduler.upload_schedule(schedule)
    print(schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    qnos_instr_time = alice.local_ehi.latencies.qnos_instr_time
    expected_duration = host_instr_time + 5 * qnos_instr_time
    assert ns.sim_time() == expected_duration
    alice.memmgr.get_process(pid).host_mem.read("y") == 4
    bob.memmgr.get_process(pid).host_mem.read("y") == 4


def test_epr_md_1():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    qpu_schedule = StaticSchedule(
        [StaticScheduleEntry(BlockTask(0, pid, "blk_epr_md_1", QC))]
    )
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    expected_duration = alice.network_ehi.get_link(0, 1).duration
    assert ns.sim_time() == expected_duration
    alice_outcome = alice.memmgr.get_process(pid).host_mem.read("m")
    bob_outcome = bob.memmgr.get_process(pid).host_mem.read("m")
    assert alice_outcome == bob_outcome


def test_epr_md_2():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    qpu_schedule = StaticSchedule(
        [StaticScheduleEntry(BlockTask(0, pid, "blk_epr_md_2", QC))]
    )
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    expected_duration = alice.network_ehi.get_link(0, 1).duration * 2
    assert ns.sim_time() == expected_duration
    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    alice_outcomes = [alice_mem.read("m0"), alice_mem.read("m1")]
    bob_outcomes = [bob_mem.read("m0"), bob_mem.read("m1")]
    assert alice_outcomes == bob_outcomes


def test_epr_ck_1():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    qpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid, "blk_epr_ck_1", QC)),
            StaticScheduleEntry(BlockTask(1, pid, "blk_meas_q0", QL)),
        ]
    )
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    # subrt meas_q0 has 3 classical instructions + 1 meas instruction
    subrt_class_time = 3 * alice.local_ehi.latencies.qnos_instr_time
    subrt_meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    subrt_time = subrt_class_time + subrt_meas_time
    expected_duration = alice.network_ehi.get_link(0, 1).duration + subrt_time
    assert ns.sim_time() == expected_duration
    alice_outcome = alice.memmgr.get_process(pid).host_mem.read("p")
    bob_outcome = bob.memmgr.get_process(pid).host_mem.read("p")
    assert alice_outcome == bob_outcome


def test_epr_ck_2():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    qpu_schedule = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid, "blk_epr_ck_2", QC)),
            StaticScheduleEntry(BlockTask(1, pid, "blk_meas_q0_q1", QL)),
        ]
    )
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    # subrt meas_q0_q1 has 6 classical instructions + 2 meas instruction
    subrt_class_time = 6 * alice.local_ehi.latencies.qnos_instr_time
    meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    subrt_time = subrt_class_time + 2 * meas_time
    epr_time = alice.network_ehi.get_link(0, 1).duration
    expected_duration = 2 * epr_time + subrt_time
    assert ns.sim_time() == expected_duration
    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    alice_outcomes = [alice_mem.read("p0"), alice_mem.read("p1")]
    bob_outcomes = [bob_mem.read("p0"), bob_mem.read("p1")]
    assert alice_outcomes == bob_outcomes


def test_cc():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    assert alice.local_ehi.latencies.host_peer_latency == 3000
    assert alice.local_ehi.latencies.host_instr_time == 1000

    schedule_alice = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(0, pid, "blk_prep_cc", CL), 0),
            StaticScheduleEntry(BlockTask(1, pid, "blk_send", CL), 2000),
            StaticScheduleEntry(BlockTask(2, pid, "blk_host1", CL), 10000),
        ]
    )
    schedule_bob = StaticSchedule(
        [
            StaticScheduleEntry(BlockTask(4, pid, "blk_prep_cc", CL), 0),
            StaticScheduleEntry(BlockTask(5, pid, "blk_recv", CC), 3000),
            StaticScheduleEntry(BlockTask(6, pid, "blk_host1", CL), 10000),
        ]
    )
    alice.scheduler.upload_cpu_schedule(schedule_alice)
    bob.scheduler.upload_cpu_schedule(schedule_bob)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    # assert ns.sim_time() == expected_duration
    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    assert bob_mem.read("msg") == 25
    assert alice_mem.read("var_z") == 9
    assert bob_mem.read("var_z") == 9


def test_full_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program_alice = load_program("test_scheduling_alice.iqoala")
    program_bob = load_program("test_scheduling_bob.iqoala")
    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    inputs_bob = ProgramInput({"alice_id": 0})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)
    instance_bob = instantiate(program_bob, bob.local_ehi, pid, inputs_bob)

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    tasks_alice = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC).from_program(
        program_alice, pid, alice.local_ehi, alice.network_ehi
    )
    tasks_bob = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC).from_program(
        program_bob, pid, bob.local_ehi, bob.network_ehi
    )
    schedule_alice = StaticSchedule.consecutive_block_tasks([tasks_alice])
    schedule_bob = StaticSchedule.consecutive_block_tasks([tasks_bob])

    alice.scheduler.upload_schedule(schedule_alice)
    bob.scheduler.upload_schedule(schedule_bob)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    alice_outcomes = [alice_mem.read("p0"), alice_mem.read("p1")]
    bob_outcomes = [bob_mem.read("p0"), bob_mem.read("p1")]
    assert alice_outcomes == bob_outcomes


if __name__ == "__main__":
    test_consecutive()
    test_consecutive_qc_slots()
    test_consecutive_block_tasks_with_timestamps()
    test_cpu_driver()
    test_cpu_driver_no_time()
    test_cpu_driver_2_processes()
    test_qpu_driver()
    test_qpu_driver_2_processes()
    test_host_program()
    test_lr_program()
    test_epr_md_1()
    test_epr_md_2()
    test_epr_ck_1()
    test_epr_ck_2()
    test_cc()
    test_full_program()
