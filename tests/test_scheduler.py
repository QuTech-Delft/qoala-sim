import os
from typing import Optional

import netsquid as ns
import pytest
from netqasm.lang.instr import core

from qoala.lang.ehi import EhiNodeInfo, UnitModule
from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.ntf import GenericNtf
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.task import (
    HostEventTask,
    HostLocalTask,
    LocalRoutineTask,
    MultiPairTask,
    PostCallTask,
    PreCallTask,
    TaskGraph,
    TaskGraphBuilder,
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
        "alice", driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.upload_task_graph(graph)

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
    graph = TaskGraphBuilder.linear_tasks(tasks)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = CpuEdfScheduler(
        "alice", driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.upload_task_graph(graph)

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
        "alice", driver, procnode.memmgr, procnode.host.interface
    )
    scheduler.upload_task_graph(graph)

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
        "alice", cpu_driver, procnode.memmgr, procnode.host.interface
    )
    cpu_scheduler.upload_task_graph(cpu_graph)

    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = QpuEdfScheduler("alice", qpu_driver, procnode.memmgr, None)
    qpu_scheduler.upload_task_graph(qpu_graph)

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
        (LocalRoutineTask(6, pid0, "b1", shared_ptr_pid0), 2000),
        (LocalRoutineTask(7, pid1, "b1", shared_ptr_pid1), 2000),
    ]
    qpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(qpu_tasks)

    cpu_graph.get_tinfo(4).ext_predecessors.add(6)
    cpu_graph.get_tinfo(5).ext_predecessors.add(7)
    qpu_graph.get_tinfo(6).ext_predecessors.add(2)
    qpu_graph.get_tinfo(7).ext_predecessors.add(3)

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = CpuEdfScheduler(
        "alice", cpu_driver, procnode.memmgr, procnode.host.interface
    )
    cpu_scheduler.upload_task_graph(cpu_graph)

    qpu_driver = QpuDriver(
        "alice",
        mem,
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.memmgr,
        procnode.memmgr,
    )
    qpu_scheduler = QpuEdfScheduler("alice", qpu_driver, procnode.memmgr, None)
    qpu_scheduler.upload_task_graph(qpu_graph)

    cpu_scheduler.set_other_scheduler(qpu_scheduler)
    qpu_scheduler.set_other_scheduler(cpu_scheduler)

    ns.sim_reset()
    cpu_scheduler.start()
    qpu_scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid0).host_mem.read("y") == 4
    assert procnode.memmgr.get_process(pid1).host_mem.read("y") == 4

    assert ns.sim_time() == 2000


def test_host_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = load_program("test_scheduling_alice.iqoala")
    pid = 0
    instance = instantiate(program, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance, remote_pid=0)
    bob.scheduler.submit_program_instance(instance, remote_pid=0)

    tasks = [
        HostLocalTask(0, pid, "blk_host0"),
        HostLocalTask(1, pid, "blk_host1"),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)

    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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
    instance = instantiate(program, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance, remote_pid=0)
    bob.scheduler.submit_program_instance(instance, remote_pid=0)

    host_instr_time = alice.local_ehi.latencies.host_instr_time

    shared_ptr = 0
    tasks = [
        HostLocalTask(0, pid, "blk_host2"),
        PreCallTask(1, pid, "blk_add_one", shared_ptr),
        LocalRoutineTask(2, pid, "blk_add_one", shared_ptr),
        PostCallTask(3, pid, "blk_add_one", shared_ptr),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)

    LogManager.set_log_level("DEBUG")
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)
    print(graph)

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    shared_ptr = 0
    tasks = [
        PreCallTask(0, pid, "blk_epr_md_1", shared_ptr),
        MultiPairTask(1, pid, shared_ptr),
        PostCallTask(2, pid, "blk_epr_md_1", shared_ptr),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    shared_ptr = 0
    tasks = [
        PreCallTask(0, pid, "blk_epr_md_2", shared_ptr),
        MultiPairTask(1, pid, shared_ptr),
        PostCallTask(2, pid, "blk_epr_md_2", shared_ptr),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    shared_ptr0 = 0
    shared_ptr1 = 1
    tasks = [
        PreCallTask(0, pid, "blk_epr_ck_1", shared_ptr0),
        MultiPairTask(1, pid, shared_ptr0),
        PostCallTask(2, pid, "blk_epr_ck_1", shared_ptr0),
        PreCallTask(3, pid, "blk_meas_q0", shared_ptr1),
        LocalRoutineTask(4, pid, "blk_meas_q0", shared_ptr1),
        PostCallTask(5, pid, "blk_meas_q0", shared_ptr1),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    shared_ptr0 = 0
    shared_ptr1 = 1
    tasks = [
        PreCallTask(0, pid, "blk_epr_ck_2", shared_ptr0),
        MultiPairTask(1, pid, shared_ptr0),
        PostCallTask(2, pid, "blk_epr_ck_2", shared_ptr0),
        PreCallTask(3, pid, "blk_meas_q0_q1", shared_ptr1),
        LocalRoutineTask(4, pid, "blk_meas_q0_q1", shared_ptr1),
        PostCallTask(5, pid, "blk_meas_q0_q1", shared_ptr1),
    ]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    alice.scheduler.submit_program_instance(instance_alice, instance_bob.pid)
    bob.scheduler.submit_program_instance(instance_bob, instance_alice.pid)

    assert alice.local_ehi.latencies.host_peer_latency == 3000
    assert alice.local_ehi.latencies.host_instr_time == 1000

    tasks_alice = [
        (HostLocalTask(0, pid, "blk_prep_cc"), 0),
        (HostLocalTask(1, pid, "blk_send"), 2000),
        (HostLocalTask(2, pid, "blk_host1"), 10000),
    ]
    graph_alice = TaskGraphBuilder.linear_tasks_with_start_times(tasks_alice)
    tasks_bob = [
        (HostLocalTask(4, pid, "blk_prep_cc"), 0),
        (HostEventTask(5, pid, "blk_recv"), 3000),
        (HostLocalTask(6, pid, "blk_host1"), 10000),
    ]
    graph_bob = TaskGraphBuilder.linear_tasks_with_start_times(tasks_bob)
    alice.scheduler.upload_cpu_task_graph(graph_alice)
    bob.scheduler.upload_cpu_task_graph(graph_bob)

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

    tasks_alice = TaskGraphBuilder.from_program(
        program_alice, pid, alice.local_ehi, alice.network_ehi
    )
    tasks_bob = TaskGraphBuilder.from_program(
        program_bob, pid, bob.local_ehi, bob.network_ehi
    )

    alice.scheduler.upload_task_graph(tasks_alice)
    bob.scheduler.upload_task_graph(tasks_bob)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    alice_outcomes = [alice_mem.read("p0"), alice_mem.read("p1")]
    bob_outcomes = [bob_mem.read("p0"), bob_mem.read("p1")]
    assert alice_outcomes == bob_outcomes


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
