import netsquid as ns

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
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface, LhiConverter
from qoala.runtime.taskcreator import (
    CpuSchedule,
    CpuTask,
    QpuSchedule,
    QpuTask,
    RoutineType,
)
from qoala.sim import host
from qoala.sim.build import build_network_from_lhi, build_qprocessor_from_topology
from qoala.sim.driver import CpuDriver
from qoala.sim.procnode import ProcNode
from qoala.util.builder import ObjectBuilder
from qoala.util.tests import netsquid_run


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


def test_host_program():
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
    network = build_network_from_lhi([alice_lhi, bob_lhi], network_info, network_lhi)

    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = get_pure_host_program()
    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    alice.scheduler.submit_program_instance(instance)
    bob.scheduler.submit_program_instance(instance)

    cpu_schedule = CpuSchedule.no_constraints([CpuTask(pid, "b0"), CpuTask(pid, "b1")])
    alice.scheduler.upload_cpu_schedule(cpu_schedule)
    bob.scheduler.upload_cpu_schedule(cpu_schedule)

    ns.sim_reset()
    alice.start()
    bob.start()
    ns.sim_run()

    assert ns.sim_time() == 3 * alice.local_ehi.latencies.host_instr_time
    alice.memmgr.get_process(pid).host_mem.read("var_z") == 9
    bob.memmgr.get_process(pid).host_mem.read("var_z") == 9


def test_lr_program():
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
    network = build_network_from_lhi([alice_lhi, bob_lhi], network_info, network_lhi)

    alice = network.nodes["alice"]
    bob = network.nodes["bob"]

    program = get_lr_program()
    pid = 0
    instance = ObjectBuilder.simple_program_instance(program, pid)

    alice.scheduler.submit_program_instance(instance)
    bob.scheduler.submit_program_instance(instance)

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    cpu_schedule = CpuSchedule.no_constraints([CpuTask(pid, "b0")])
    qpu_schedule = QpuSchedule(
        [(host_instr_time, QpuTask(pid, RoutineType.LOCAL, "b1"))]
    )
    alice.scheduler.upload_cpu_schedule(cpu_schedule)
    alice.scheduler.upload_qpu_schedule(qpu_schedule)
    bob.scheduler.upload_cpu_schedule(cpu_schedule)
    bob.scheduler.upload_qpu_schedule(qpu_schedule)

    ns.sim_reset()
    alice.start()
    bob.start()
    ns.sim_run()

    host_instr_time = alice.local_ehi.latencies.host_instr_time
    qnos_instr_time = alice.local_ehi.latencies.qnos_instr_time
    expected_duration = host_instr_time + 4 * qnos_instr_time
    assert ns.sim_time() == expected_duration
    alice.memmgr.get_process(pid).host_mem.read("y") == 4
    bob.memmgr.get_process(pid).host_mem.read("y") == 4


if __name__ == "__main__":
    # test_host_program()
    test_lr_program()
