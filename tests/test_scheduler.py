import os
from typing import Dict, List, Optional

import netsquid as ns
import pytest
from netqasm.lang.instr import TrappedIonFlavour, core

from qoala.lang.ehi import EhiNodeInfo, UnitModule
from qoala.lang.hostlang import BasicBlock, BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.ntf import GenericNtf, TrappedIonNtf
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.task import (
    HostLocalTask,
    LocalRoutineTask,
    PostCallTask,
    PreCallTask,
    TaskGraphBuilder,
    TaskInfo,
)
from qoala.sim.build import build_network_from_lhi
from qoala.sim.driver import CpuDriver, QpuDriver, SharedSchedulerMemory
from qoala.sim.network import ProcNodeNetwork
from qoala.sim.scheduler import CpuEdfScheduler, QpuEdfScheduler
from qoala.util.builder import ObjectBuilder
from qoala.util.logging import LogManager

CL = BasicBlockType.CL
CC = BasicBlockType.CC
QL = BasicBlockType.QL
QC = BasicBlockType.QC


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
    tuple<y> = run_subroutine(tuple<x>) : add_one

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


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def load_program_trapped_ion(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text, flavour=TrappedIonFlavour()).parse()


def setup_network(internal_sched_latency: float = 0) -> ProcNodeNetwork:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=3)
    latencies = LhiLatencies(
        host_instr_time=1000,
        qnos_instr_time=2000,
        host_peer_latency=3000,
        internal_sched_latency=internal_sched_latency,
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


def setup_network_trapped_ion() -> ProcNodeNetwork:
    num_qubits = 3
    topology = LhiTopologyBuilder.trapped_ion_default_perfect_gates(
        num_qubits=num_qubits
    )
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
    ntfs = [TrappedIonNtf(), TrappedIonNtf()]
    return build_network_from_lhi([alice_lhi, bob_lhi], ntfs, network_lhi)


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
    )


def test_cpu_scheduler():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    tasks_with_start_times = [
        (HostLocalTask(0, 0, "b0"), 0),
        (HostLocalTask(1, 0, "b1"), 1000),
    ]
    graph = TaskGraphBuilder.linear_tasks_with_start_times(tasks_with_start_times)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = CpuEdfScheduler(
        "alice", 0, driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.add_tasks(graph.get_tasks())

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid).host_mem.read("var_y") == 5
    assert procnode.memmgr.get_process(pid).host_mem.read("var_z") == 9

    assert ns.sim_time() == 1000


def test_cpu_scheduler_no_time():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    tasks = [HostLocalTask(0, 0, "b0"), HostLocalTask(1, 0, "b1")]
    tinfos: Dict[int, TaskInfo] = {
        task.task_id: TaskInfo.only_task(task) for task in tasks
    }

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = CpuEdfScheduler(
        "alice", 0, driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.add_tasks(tinfos)

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("var_x") == 3
    assert procnode.memmgr.get_process(pid).host_mem.read("var_y") == 5
    assert procnode.memmgr.get_process(pid).host_mem.read("var_z") == 9

    assert ns.sim_time() == 0


def test_cpu_scheduler_2_processes():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid0 = 0
    pid1 = 1
    instance0 = ObjectBuilder.simple_program_instance(program, pid0)
    instance1 = ObjectBuilder.simple_program_instance(program, pid1)

    procnode.scheduler.submit_program_instance(instance0)
    procnode.scheduler.submit_program_instance(instance1)

    tasks_with_start_times = [
        (HostLocalTask(0, pid0, "b0"), 0),
        (HostLocalTask(1, pid1, "b0"), 500),
        (HostLocalTask(2, pid0, "b1"), 1000),
        (HostLocalTask(3, pid1, "b1"), 1500),
    ]
    graph = TaskGraphBuilder.linear_tasks_with_start_times(tasks_with_start_times)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = CpuEdfScheduler(
        "alice", 0, driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.add_tasks(graph.get_tasks())

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


def test_qpu_scheduler():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_lr_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    shared_ptr = 0

    cpu_tasks_with_start_times = [
        (HostLocalTask(0, 0, "b0"), 0),
        (PreCallTask(1, 0, "b1", shared_ptr), 1000),
        (PostCallTask(2, 0, "b1", shared_ptr), 5000),
    ]
    cpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(
        cpu_tasks_with_start_times
    )
    qpu_tasks_with_start_times = [
        (LocalRoutineTask(3, 0, "b1", shared_ptr), 2000),
    ]
    qpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(
        qpu_tasks_with_start_times
    )
    qpu_graph.get_tinfo(3).ext_predecessors.add(1)
    cpu_graph.get_tinfo(2).ext_predecessors.add(3)

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = CpuEdfScheduler(
        "alice", 0, cpu_driver, procnode.memmgr, procnode.host.interface
    )
    cpu_scheduler.add_tasks(cpu_graph.get_tasks())

    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = QpuEdfScheduler("alice", 0, qpu_driver, procnode.memmgr, None)
    qpu_scheduler.add_tasks(qpu_graph.get_tasks())

    cpu_scheduler.set_other_scheduler(qpu_scheduler)
    qpu_scheduler.set_other_scheduler(cpu_scheduler)

    LogManager.set_log_level("INFO")
    ns.sim_reset()
    cpu_scheduler.start()
    qpu_scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("y") == 4

    assert ns.sim_time() == 5000


def test_qpu_scheduler_2_processes():
    LogManager.enable_task_logger(True)

    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_lr_program()

    pid0 = 0
    pid1 = 1
    instance0 = ObjectBuilder.simple_program_instance(program, pid0)
    instance1 = ObjectBuilder.simple_program_instance(program, pid1)

    procnode.scheduler.submit_program_instance(instance0)
    procnode.scheduler.submit_program_instance(instance1)

    shared_ptr_pid0 = 0
    shared_ptr_pid1 = 1

    cpu_tasks = [
        (HostLocalTask(0, pid0, "b0", CL), 0),
        (HostLocalTask(1, pid1, "b0", CL), 500),
        (PreCallTask(2, pid0, "b1", shared_ptr_pid0), 1000),
        (PreCallTask(3, pid1, "b1", shared_ptr_pid1), 1000),
        (PostCallTask(4, pid0, "b1", shared_ptr_pid0), 1000),
        (PostCallTask(5, pid1, "b1", shared_ptr_pid1), 1000),
    ]
    cpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(cpu_tasks)
    qpu_tasks = [
        (LocalRoutineTask(6, pid0, "b1", shared_ptr_pid0), 1000),
        (LocalRoutineTask(7, pid1, "b1", shared_ptr_pid1), 1000),
    ]
    qpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(qpu_tasks)

    cpu_graph.get_tinfo(4).ext_predecessors.add(6)
    cpu_graph.get_tinfo(5).ext_predecessors.add(7)
    qpu_graph.get_tinfo(6).ext_predecessors.add(2)
    qpu_graph.get_tinfo(7).ext_predecessors.add(3)

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = CpuEdfScheduler(
        "alice", 0, cpu_driver, procnode.memmgr, procnode.host.interface
    )
    cpu_scheduler.add_tasks(cpu_graph.get_tasks())

    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = QpuEdfScheduler("alice", 0, qpu_driver, procnode.memmgr, None)
    qpu_scheduler.add_tasks(qpu_graph.get_tasks())

    cpu_scheduler.set_other_scheduler(qpu_scheduler)
    qpu_scheduler.set_other_scheduler(cpu_scheduler)

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
    instance = instantiate(program, alice.local_ehi, pid, ProgramInput({"bob_id": 1}))

    used_blocks = {"blk_host0", "blk_host1"}

    new_blocks: List[BasicBlock] = []
    for block in instance.program.blocks:
        if block.name in used_blocks:
            new_blocks.append(block)

    instance.program.blocks = new_blocks

    alice.scheduler.submit_program_instance(instance, remote_pid=0)
    bob.scheduler.submit_program_instance(instance, remote_pid=0)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 3 * alice.local_ehi.latencies.host_instr_time
    assert alice.memmgr.get_process(pid).host_mem.read("var_z") == 9
    assert bob.memmgr.get_process(pid).host_mem.read("var_z") == 9


def test_lr_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = load_program("test_scheduling_alice.iqoala")
    pid = 0
    instance = instantiate(program, alice.local_ehi, pid, ProgramInput({"bob_id": 1}))

    used_blocks = {"blk_host2", "blk_add_one"}

    new_blocks = []
    for block in instance.program.blocks:
        if block.name in used_blocks:
            new_blocks.append(block)

    instance.program.blocks = new_blocks

    alice.scheduler.submit_program_instance(instance, remote_pid=0)
    bob.scheduler.submit_program_instance(instance, remote_pid=0)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    qnos_instr_time = alice.local_ehi.latencies.qnos_instr_time
    expected_duration = 3 * host_instr_time + 5 * qnos_instr_time
    assert ns.sim_time() == expected_duration
    assert alice.memmgr.get_process(pid).host_mem.read("y") == 4
    assert bob.memmgr.get_process(pid).host_mem.read("y") == 4


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

    used_blocks_alice = {"blk_epr_md_1"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    used_blocks_bob = {"blk_epr_md_1"}
    new_blocks_bob = []
    for block in instance_bob.program.blocks:
        if block.name in used_blocks_bob:
            new_blocks_bob.append(block)

    instance_bob.program.blocks = new_blocks_bob

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    expected_duration = alice.network_ehi.get_link(0, 1).duration
    assert ns.sim_time() == 2 * host_instr_time + expected_duration
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

    used_blocks_alice = {"blk_epr_md_2"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    used_blocks_bob = {"blk_epr_md_2"}
    new_blocks_bob = []
    for block in instance_bob.program.blocks:
        if block.name in used_blocks_bob:
            new_blocks_bob.append(block)

    instance_bob.program.blocks = new_blocks_bob

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    expected_duration = alice.network_ehi.get_link(0, 1).duration * 2
    assert ns.sim_time() == 2 * host_instr_time + expected_duration
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

    used_blocks_alice = {"blk_epr_ck_1", "blk_meas_q0"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    used_blocks_bob = {"blk_epr_ck_1", "blk_meas_q0"}
    new_blocks_bob = []
    for block in instance_bob.program.blocks:
        if block.name in used_blocks_bob:
            new_blocks_bob.append(block)

    instance_bob.program.blocks = new_blocks_bob

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    # subrt meas_q0 has 3 classical instructions + 1 meas instruction
    subrt_class_time = 3 * alice.local_ehi.latencies.qnos_instr_time
    subrt_meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    subrt_time = subrt_class_time + subrt_meas_time
    host_instr_time = alice.local_ehi.latencies.host_instr_time
    epr_time = alice.network_ehi.get_link(0, 1).duration
    expected_duration = 4 * host_instr_time + epr_time + subrt_time
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

    used_blocks_alice = {"blk_epr_ck_2", "blk_meas_q0_q1"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    used_blocks_bob = {"blk_epr_ck_2", "blk_meas_q0_q1"}
    new_blocks_bob = []
    for block in instance_bob.program.blocks:
        if block.name in used_blocks_bob:
            new_blocks_bob.append(block)

    instance_bob.program.blocks = new_blocks_bob

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    # subrt meas_q0_q1 has 6 classical instructions + 2 meas instruction
    subrt_class_time = 6 * alice.local_ehi.latencies.qnos_instr_time
    meas_time = alice.local_ehi.find_single_gate(0, core.MeasInstruction).duration
    subrt_time = subrt_class_time + 2 * meas_time
    host_instr_time = alice.local_ehi.latencies.host_instr_time
    epr_time = alice.network_ehi.get_link(0, 1).duration
    expected_duration = 4 * host_instr_time + 2 * epr_time + subrt_time
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

    # TODO: add start times ?

    used_blocks_alice = {"blk_prep_cc", "blk_send", "blk_host1"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    used_blocks_bob = {"blk_prep_cc", "blk_recv", "blk_host1"}
    new_blocks_bob = []
    for block in instance_bob.program.blocks:
        if block.name in used_blocks_bob:
            new_blocks_bob.append(block)

    instance_bob.program.blocks = new_blocks_bob

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    assert alice.local_ehi.latencies.host_peer_latency == 3000
    assert alice.local_ehi.latencies.host_instr_time == 1000

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    ns.sim_reset()
    network.start()
    ns.sim_run()


def test_jump_instruction():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_jump", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 5000  # 5 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 0
    assert alice_mem.read("var_y") == 1


def test_beq_instruction_1():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_beq_1", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 5000  # 5 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 1
    assert alice_mem.read("var_y") == 1


def test_beq_instruction_2():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_beq_2", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 7000  # 7 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 9
    assert alice_mem.read("var_y") == 9


def test_bne_instruction_1():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_bne_1", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 5000  # 5 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 2
    assert alice_mem.read("var_y") == 3


def test_bne_instruction_2():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_bne_2", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 7000  # 7 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 9
    assert alice_mem.read("var_y") == 9


def test_bgt_instruction_1():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_bgt_1", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 5000  # 5 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 5
    assert alice_mem.read("var_y") == 4


def test_bgt_instruction_2():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_bgt_2", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 7000  # 7 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 9
    assert alice_mem.read("var_y") == 9


def test_blt_instruction_1():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_blt_1", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 5000  # 5 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 6
    assert alice_mem.read("var_y") == 7


def test_blt_instruction_2():
    network = setup_network()
    alice = network.nodes["alice"]

    program_alice = load_program("test_jumping_and_branching.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    used_blocks_alice = {"blk_blt_2", "blk_temp", "blk_last"}
    new_blocks_alice = []
    for block in instance_alice.program.blocks:
        if block.name in used_blocks_alice:
            new_blocks_alice.append(block)

    instance_alice.program.blocks = new_blocks_alice

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()

    assert ns.sim_time() == 7000  # 7 * 1000

    alice_mem = alice.memmgr.get_process(pid).host_mem
    assert alice_mem.read("var_x") == 9
    assert alice_mem.read("var_y") == 9


def test_measure_all():
    network = setup_network_trapped_ion()
    alice = network.nodes["alice"]

    program_alice = load_program_trapped_ion("test_measure_all.iqoala")

    pid = 0
    inputs_alice = ProgramInput({"bob_id": 1})
    instance_alice = instantiate(program_alice, alice.local_ehi, pid, inputs_alice)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()
    ns.sim_run()
    alice_outcome = alice.memmgr.get_process(pid).host_mem.read("m0")
    alice_outcome2 = alice.memmgr.get_process(pid).host_mem.read("m1")
    alice_outcome3 = alice.memmgr.get_process(pid).host_mem.read("m2")
    assert alice_outcome == alice_outcome2 == alice_outcome3 == 0


def test_internal_sched_latency():
    network = setup_network(internal_sched_latency=500)
    alice = network.nodes["alice"]

    program_alice = load_program("test_internal_sched_latency.iqoala")

    pid = 0
    instance_alice = instantiate(program_alice, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance_alice, 0)

    ns.sim_reset()
    assert ns.sim_time() == 0
    network.start()

    network.nodes["bob"].stop()
    ns.sim_run()

    total_host_instr_time = 4 * 1000
    total_internal_sched_latency = 2 * 500  # 2 CL blocks => 2 * 500
    total_time = total_host_instr_time + total_internal_sched_latency
    assert ns.sim_time() == total_time


if __name__ == "__main__":
    test_cpu_scheduler()
    test_cpu_scheduler_no_time()
    test_cpu_scheduler_2_processes()
    test_qpu_scheduler()
    test_qpu_scheduler_2_processes()
    test_host_program()
    test_lr_program()
    test_epr_md_1()
    test_epr_md_2()
    test_epr_ck_1()
    test_epr_ck_2()
    test_cc()
    test_full_program()
    test_jump_instruction()
    test_beq_instruction_1()
    test_beq_instruction_2()
    test_bne_instruction_1()
    test_bne_instruction_2()
    test_bgt_instruction_1()
    test_bgt_instruction_2()
    test_blt_instruction_1()
    test_blt_instruction_2()
    test_measure_all()
    test_internal_sched_latency()
