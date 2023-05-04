import netsquid as ns
import pytest

from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.schedule import StaticSchedule, StaticScheduleEntry
from qoala.runtime.task import BlockTask, TaskGraph
from qoala.sim.driver import CpuDriver, QpuDriver, SharedSchedulerMemory
from qoala.sim.scheduler import TimeTriggeredScheduler
from qoala.util.builder import ObjectBuilder

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


def test_cpu_scheduler():
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


def test_cpu_scheduler_no_time():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    task_graph = TaskGraph()
    task_graph.add_tasks([BlockTask(0, 0, "b0", CL), BlockTask(1, 0, "b1", CL)])
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


def test_cpu_scheduler_2_processes():
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


def test_qpu_scheduler():
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


def test_qpu_scheduler_2_processes():
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


if __name__ == "__main__":
    test_cpu_scheduler()
    test_cpu_scheduler_no_time()
    test_cpu_scheduler_2_processes()
    test_qpu_scheduler()
    test_qpu_scheduler_2_processes()
