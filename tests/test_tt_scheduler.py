import os
from typing import Optional

import netsquid as ns
import pytest
from netqasm.lang.instr import core

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
from qoala.runtime.task import (
    BlockTask,
    TaskCreator,
    TaskExecutionMode,
    TaskGraph,
    TaskGraphBuilder,
)
from qoala.sim.build import build_network_from_lhi
from qoala.sim.driver import CpuDriver, QpuDriver, SharedSchedulerMemory
from qoala.sim.network import ProcNodeNetwork
from qoala.sim.scheduler import TimeTriggeredScheduler
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


def test_cpu_scheduler():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    tasks_with_start_times = [
        (BlockTask(0, 0, "b0", CL), 0),
        (BlockTask(1, 0, "b1", CL), 1000),
    ]
    graph = TaskGraphBuilder.linear_tasks_with_start_times(tasks_with_start_times)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
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

    tasks = [BlockTask(0, 0, "b0", CL), BlockTask(1, 0, "b1", CL)]
    graph = TaskGraphBuilder.linear_tasks(tasks)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
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
        (BlockTask(0, pid0, "b0", CL), 0),
        (BlockTask(1, pid1, "b0", CL), 500),
        (BlockTask(2, pid0, "b1", CL), 1000),
        (BlockTask(3, pid1, "b1", CL), 1500),
    ]
    graph = TaskGraphBuilder.linear_tasks_with_start_times(tasks_with_start_times)

    mem = SharedSchedulerMemory()
    driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    scheduler = TimeTriggeredScheduler("alice", driver)
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

    cpu_tasks_with_start_times = [
        (BlockTask(0, 0, "b0", CL), 0),
    ]
    cpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(
        cpu_tasks_with_start_times
    )
    qpu_tasks_with_start_times = [
        (BlockTask(1, 0, "b1", QL), 1000),
    ]
    qpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(
        qpu_tasks_with_start_times
    )

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = TimeTriggeredScheduler("alice", cpu_driver)
    cpu_scheduler.upload_task_graph(cpu_graph)

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
    qpu_scheduler.upload_task_graph(qpu_graph)

    ns.sim_reset()
    cpu_scheduler.start()
    qpu_scheduler.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid).host_mem.read("y") == 4

    assert ns.sim_time() == 1000


def test_qpu_scheduler_2_processes():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_lr_program()

    pid0 = 0
    pid1 = 1
    instance0 = ObjectBuilder.simple_program_instance(program, pid0)
    instance1 = ObjectBuilder.simple_program_instance(program, pid1)

    procnode.scheduler.submit_program_instance(instance0)
    procnode.scheduler.submit_program_instance(instance1)

    cpu_tasks = [
        (BlockTask(0, pid0, "b0", CL), 0),
        (BlockTask(1, pid1, "b0", CL), 500),
    ]
    cpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(cpu_tasks)
    qpu_tasks = [
        (BlockTask(0, pid0, "b1", QL), 500),
        (BlockTask(1, pid1, "b1", QL), 1000),
    ]
    qpu_graph = TaskGraphBuilder.linear_tasks_with_start_times(qpu_tasks)

    mem = SharedSchedulerMemory()
    cpu_driver = CpuDriver("alice", mem, procnode.host.processor, procnode.memmgr)
    cpu_scheduler = TimeTriggeredScheduler("alice", cpu_driver)
    cpu_scheduler.upload_task_graph(cpu_graph)

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
    qpu_scheduler.upload_task_graph(qpu_graph)

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
    graph = TaskGraphBuilder.linear_tasks(tasks)

    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    tasks = [BlockTask(0, pid, "blk_epr_md_1", QC)]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    tasks = [BlockTask(0, pid, "blk_epr_md_2", QC)]
    graph = TaskGraphBuilder.linear_tasks(tasks)
    alice.scheduler.upload_task_graph(graph)
    bob.scheduler.upload_task_graph(graph)

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

    tasks = [
        BlockTask(0, pid, "blk_epr_ck_1", QC),
        BlockTask(1, pid, "blk_meas_q0", QL),
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

    tasks = [
        BlockTask(0, pid, "blk_epr_ck_2", QC),
        BlockTask(1, pid, "blk_meas_q0_q1", QL),
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

    tasks_alice = [
        (BlockTask(0, pid, "blk_prep_cc", CL), 0),
        (BlockTask(1, pid, "blk_send", CL), 2000),
        (BlockTask(2, pid, "blk_host1", CL), 10000),
    ]
    graph_alice = TaskGraphBuilder.linear_tasks_with_start_times(tasks_alice)
    tasks_bob = [
        (BlockTask(4, pid, "blk_prep_cc", CL), 0),
        (BlockTask(5, pid, "blk_recv", CC), 3000),
        (BlockTask(6, pid, "blk_host1", CL), 10000),
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

    alice.scheduler.submit_program_instance(instance_alice)
    bob.scheduler.submit_program_instance(instance_bob)

    tasks_alice = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC).from_program(
        program_alice, pid, alice.local_ehi, alice.network_ehi
    )
    tasks_bob = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC).from_program(
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
