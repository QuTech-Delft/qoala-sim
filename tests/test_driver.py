import netsquid as ns
import pytest

from qoala.lang.hostlang import BasicBlockType
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.schedule import BlockTask, TaskSchedule, TaskScheduleEntry
from qoala.sim.driver import CpuDriver, QpuDriver
from qoala.util.builder import ObjectBuilder


def get_pure_host_program() -> IqoalaProgram:
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

    return IqoalaParser(program_text).parse()


def get_lr_program() -> IqoalaProgram:
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

    return IqoalaParser(program_text).parse()


CL = BasicBlockType.CL
QL = BasicBlockType.QL
QC = BasicBlockType.QC


def test_cpu_driver():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    cpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(0, "b0", CL), 0),
            TaskScheduleEntry(BlockTask(0, "b1", CL), 1000),
        ]
    )

    driver = CpuDriver("alice", procnode.host.processor, procnode.memmgr)
    driver.upload_schedule(cpu_schedule)

    ns.sim_reset()
    driver.start()
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

    cpu_schedule = TaskSchedule.consecutive(
        [BlockTask(0, "b0", CL), BlockTask(0, "b1", CL)]
    )

    driver = CpuDriver("alice", procnode.host.processor, procnode.memmgr)
    driver.upload_schedule(cpu_schedule)

    ns.sim_reset()
    driver.start()
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

    cpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(pid0, "b0", CL), 0),
            TaskScheduleEntry(BlockTask(pid1, "b0", CL), 500),
            TaskScheduleEntry(BlockTask(pid0, "b1", CL), 1000),
            TaskScheduleEntry(BlockTask(pid1, "b1", CL), 1500),
        ]
    )

    driver = CpuDriver("alice", procnode.host.processor, procnode.memmgr)
    driver.upload_schedule(cpu_schedule)

    ns.sim_reset()
    driver.start()
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

    cpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(0, "b0", CL), 0),
        ]
    )
    qpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(0, "b1", QL), 1000),
        ]
    )

    cpudriver = CpuDriver("alice", procnode.host.processor, procnode.memmgr)
    cpudriver.upload_schedule(cpu_schedule)

    qpudriver = QpuDriver(
        "alice",
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.netstack.processor,
        procnode.memmgr,
    )
    qpudriver.upload_schedule(qpu_schedule)

    ns.sim_reset()
    cpudriver.start()
    qpudriver.start()
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

    cpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(pid0, "b0", CL), 0),
            TaskScheduleEntry(BlockTask(pid1, "b0", CL), 500),
        ]
    )
    qpu_schedule = TaskSchedule(
        [
            TaskScheduleEntry(BlockTask(pid0, "b1", QL), 500),
            TaskScheduleEntry(BlockTask(pid1, "b1", QL), 1000),
        ]
    )

    cpudriver = CpuDriver("alice", procnode.host.processor, procnode.memmgr)
    cpudriver.upload_schedule(cpu_schedule)

    qpudriver = QpuDriver(
        "alice",
        procnode.host.processor,
        procnode.qnos.processor,
        procnode.netstack.processor,
        procnode.memmgr,
    )
    qpudriver.upload_schedule(qpu_schedule)

    ns.sim_reset()
    cpudriver.start()
    qpudriver.start()
    ns.sim_run()

    assert procnode.memmgr.get_process(pid0).host_mem.read("y") == 4
    assert procnode.memmgr.get_process(pid1).host_mem.read("y") == 4

    assert ns.sim_time() == 1000


if __name__ == "__main__":
    test_cpu_driver()
    test_cpu_driver_no_time()
    test_cpu_driver_2_processes()
    test_qpu_driver()
    test_qpu_driver_2_processes()
