import os
from typing import Optional

import netsquid as ns
from netqasm.lang.instr.flavour import core

from qoala.lang.ehi import ExposedHardwareInfo, UnitModule
from qoala.lang.parse import IqoalaParser
from qoala.lang.program import IqoalaProgram
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiProcNodeInfo,
    LhiTopologyBuilder,
    NetworkLhi,
)
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.runtime.schedule import ProgramTaskList
from qoala.runtime.taskcreator import (
    CpuSchedule,
    CpuTask,
    QpuSchedule,
    QpuTask,
    RoutineType,
    TaskCreator,
    TaskExecutionMode,
)
from qoala.sim.build import build_network_from_lhi
from qoala.sim.network import ProcNodeNetwork


def load_program(path: str) -> IqoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return IqoalaParser(text).parse()


def setup_network() -> ProcNodeNetwork:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=3)
    latencies = LhiLatencies(
        host_instr_time=1000, qnos_instr_time=2000, host_peer_latency=3000
    )
    link_info = LhiLinkInfo.perfect(duration=20_000)

    alice_lhi = LhiProcNodeInfo(
        name="alice", id=0, topology=topology, latencies=latencies
    )
    network_lhi = NetworkLhi.fully_connected([0, 1], link_info)
    network_info = NetworkInfo.with_nodes({0: "alice", 1: "bob"})
    bob_lhi = LhiProcNodeInfo(name="bob", id=1, topology=topology, latencies=latencies)
    return build_network_from_lhi([alice_lhi, bob_lhi], network_info, network_lhi)


def instantiate(
    program: IqoalaProgram,
    ehi: ExposedHardwareInfo,
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
        tasks=ProgramTaskList.empty(program),
        unit_module=unit_module,
    )


def test_host_program():
    network = setup_network()
    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = load_program("test_scheduling_alice.iqoala")
    pid = 0
    instance = instantiate(program, alice.local_ehi, pid)

    alice.scheduler.submit_program_instance(instance)
    bob.scheduler.submit_program_instance(instance)

    cpu_schedule = CpuSchedule.no_constraints(
        [
            CpuTask(pid, "blk_host0"),
            CpuTask(pid, "blk_host1"),
        ]
    )
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
    cpu_schedule = CpuSchedule.no_constraints([CpuTask(pid, "blk_host2")])
    qpu_schedule = QpuSchedule(
        [(host_instr_time, QpuTask(pid, RoutineType.LOCAL, "blk_add_one"))]
    )
    alice.scheduler.upload_cpu_schedule(cpu_schedule)
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_cpu_schedule(cpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

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

    qpu_schedule = QpuSchedule(
        [(None, QpuTask(pid, RoutineType.REQUEST, "blk_epr_md_1"))]
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

    qpu_schedule = QpuSchedule(
        [(None, QpuTask(pid, RoutineType.REQUEST, "blk_epr_md_2"))]
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

    qpu_schedule = QpuSchedule(
        [
            (None, QpuTask(pid, RoutineType.REQUEST, "blk_epr_ck_1")),
            (None, QpuTask(pid, RoutineType.LOCAL, "blk_meas_q0")),
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

    qpu_schedule = QpuSchedule(
        [
            (None, QpuTask(pid, RoutineType.REQUEST, "blk_epr_ck_2")),
            (None, QpuTask(pid, RoutineType.LOCAL, "blk_meas_q0_q1")),
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

    task_creator = TaskCreator(mode=TaskExecutionMode.ROUTINE_ATOMIC)
    tasks_alice = task_creator.from_program(
        program_alice, pid, alice.local_ehi, alice.network_ehi
    )
    tasks_bob = task_creator.from_program(
        program_bob, pid, bob.local_ehi, bob.network_ehi
    )

    cpu_alice = CpuSchedule.consecutive(tasks_alice)
    qpu_alice = QpuSchedule.consecutive(tasks_alice)
    cpu_bob = CpuSchedule.consecutive(tasks_bob)
    qpu_bob = QpuSchedule.consecutive(tasks_bob)

    alice.scheduler.upload_cpu_schedule(cpu_alice)
    alice.scheduler.upload_qpu_schedule(qpu_alice)
    bob.scheduler.upload_cpu_schedule(cpu_bob)
    bob.scheduler.upload_qpu_schedule(qpu_bob)

    ns.sim_reset()
    network.start()
    ns.sim_run()

    alice_mem = alice.memmgr.get_process(pid).host_mem
    bob_mem = bob.memmgr.get_process(pid).host_mem
    alice_outcomes = [alice_mem.read("p0"), alice_mem.read("p1")]
    bob_outcomes = [bob_mem.read("p0"), bob_mem.read("p1")]
    assert alice_outcomes == bob_outcomes


if __name__ == "__main__":
    test_host_program()
    test_lr_program()
    test_epr_md_1()
    test_epr_md_2()
    test_epr_ck_1()
    test_epr_ck_2()
    test_full_program()
