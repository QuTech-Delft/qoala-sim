import netsquid as ns
import pytest

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import IqoalaParseError, IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.lhi import LhiLatencies, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.schedule import ProgramTaskList
from qoala.runtime.taskcreator import (
    CpuSchedule,
    CpuTask,
    QpuSchedule,
    QpuTask,
    RoutineType,
    TaskExecutionMode,
)
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.driver import CpuDriver, QpuDriver
from qoala.sim.procnode import ProcNode
from qoala.util.tests import ObjectBuilder, netsquid_run


def get_pure_host_program() -> IqoalaProgram:
    program_text = """
META_START
    name: alice
    parameters:
    csockets:
    epr_sockets:
META_END

^b0 {type = host}:
    var_x = assign_cval() : 3
    var_y = assign_cval() : 5
^b1 {type = host}:
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

^b0 {type = host}:
    x = assign_cval() : 3
^b1 {type = LR}:
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


def test_cpu_driver():
    procnode = ObjectBuilder.simple_procnode("alice", 1)
    program = get_pure_host_program()

    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    procnode.scheduler.submit_program_instance(instance)

    cpu_schedule = CpuSchedule(
        [
            (0, CpuTask(0, "b0")),
            (1000, CpuTask(0, "b1")),
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

    cpu_schedule = CpuSchedule.no_constraints([CpuTask(0, "b0"), CpuTask(0, "b1")])

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

    cpu_schedule = CpuSchedule(
        [
            (0, CpuTask(pid0, "b0")),
            (500, CpuTask(pid1, "b0")),
            (1000, CpuTask(pid0, "b1")),
            (1500, CpuTask(pid1, "b1")),
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

    cpu_schedule = CpuSchedule(
        [
            (0, CpuTask(0, "b0")),
        ]
    )
    qpu_schedule = QpuSchedule(
        [
            (1000, QpuTask(0, RoutineType.LOCAL, "b1")),
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

    cpu_schedule = CpuSchedule(
        [
            (0, CpuTask(pid0, "b0")),
            (500, CpuTask(pid1, "b0")),
        ]
    )
    qpu_schedule = QpuSchedule(
        [
            (500, QpuTask(pid0, RoutineType.LOCAL, "b1")),
            (1000, QpuTask(pid1, RoutineType.LOCAL, "b1")),
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
