from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Type

import netsquid as ns
from netsquid.components import QuantumProcessor
from netsquid.qubits import ketstates, qubitapi

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, UnitModule
from qoala.lang.hostlang import RunSubroutineOp
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.lhi import LhiLatencies, LhiLinkInfo, LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NtfInterface
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.ntf import GenericNtf
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.driver import QpuDriver
from qoala.sim.entdist.entdist import EntDist
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.eprsocket import EprSocket
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.process import QoalaProcess
from qoala.sim.procnode import ProcNode
from qoala.sim.scheduler import NodeScheduler
from qoala.util.math import has_state


class MockScheduler(NodeScheduler):
    def __init__(self, scheduler: NodeScheduler) -> None:
        self._cpu_scheduler = scheduler.cpu_scheduler
        self._qpu_scheduler = scheduler.qpu_scheduler
        self._host = scheduler.host
        self._last_cpu_task_pid = -1
        self._last_qpu_task_pid = -1
        self._is_predictable = True
        pass

    def schedule_next_for(self, pid: int) -> None:
        pass

    def schedule_all(self) -> None:
        pass


def create_process(
    pid: int,
    remote_pid: int,
    program: QoalaProgram,
    unit_module: UnitModule,
    host_interface: HostInterface,
    network_ehi: EhiNetworkInfo,
    inputs: Optional[Dict[str, Any]] = None,
) -> QoalaProcess:
    if inputs is None:
        inputs = {}
    prog_input = ProgramInput(values=inputs)
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=prog_input,
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=0)

    process = QoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={
            id: ClassicalSocket(host_interface, name, pid, remote_pid)
            for (id, name) in program.meta.csockets.items()
        },
        epr_sockets={
            id: EprSocket(id, network_ehi.get_node_id(name), pid, remote_pid, 1.0)
            for (id, name) in program.meta.epr_sockets.items()
        },
        result=ProgramResult(values={}),
    )
    return process


def create_procnode(
    part: str,
    name: str,
    num_qubits: int,
    network_ehi: EhiNetworkInfo,
    procnode_cls: Type[ProcNode] = ProcNode,
    asynchronous: bool = False,
    pid: int = 0,
) -> ProcNode:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
    qprocessor = build_qprocessor_from_topology(f"{name}_processor", topology)

    node_id = network_ehi.get_node_id(name)
    procnode = procnode_cls(
        part=part,
        name=name,
        qprocessor=qprocessor,
        qdevice_topology=topology,
        latencies=LhiLatencies(qnos_instr_time=1000),
        ntf_interface=GenericNtf(),
        network_ehi=network_ehi,
        node_id=node_id,
        asynchronous=asynchronous,
        pid=pid,
    )

    return procnode


class BqcProcNode(ProcNode):
    def __init__(
        self,
        name: str,
        qprocessor: QuantumProcessor,
        qdevice_topology: LhiTopology,
        latencies: LhiLatencies,
        ntf_interface: NtfInterface,
        network_ehi: EhiNetworkInfo,
        node_id: Optional[int] = None,
        asynchronous: bool = False,
        pid: int = 0,
    ) -> None:
        super().__init__(
            name=name,
            qprocessor=qprocessor,
            qdevice_topology=qdevice_topology,
            latencies=latencies,
            ntf_interface=ntf_interface,
            network_ehi=network_ehi,
            node_id=node_id,
            asynchronous=asynchronous,
        )
        self.pid = pid

    def run_subroutine(
        self, process: QoalaProcess, host_instr_index: int, subrt_name: str
    ) -> Generator[EventExpression, None, None]:
        host_instr = process.program.instructions[host_instr_index]
        assert isinstance(host_instr, RunSubroutineOp)
        lrcall = self.host.processor.prepare_lr_call(process, host_instr)
        qpudriver: QpuDriver = self.scheduler.qpu_scheduler.driver
        qpudriver.allocate_qubits_for_routine(process, lrcall.routine_name)
        yield from self.qnos.processor.assign_local_routine(
            process, lrcall.routine_name, lrcall.input_addr, lrcall.result_addr
        )
        qpudriver.free_qubits_after_routine(process, lrcall.routine_name)
        self.host.processor.post_lr_call(process, host_instr, lrcall)

    def run_request(
        self, process: QoalaProcess, host_instr_index: int, req_name: str
    ) -> Generator[EventExpression, None, None]:
        host_instr = process.program.instructions[host_instr_index]
        rrcall = self.host.processor.prepare_rr_call(process, host_instr)
        assert rrcall.routine_name == req_name
        yield from self.netstack.processor.assign_request_routine(process, rrcall)


class ServerProcNode(BqcProcNode):
    def __init__(self, part: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._part = part

    def run(self) -> Generator[EventExpression, None, None]:
        self.finished = False

        process = self.memmgr.get_process(self.pid)
        self.scheduler.initialize_process(process)

        # csocket = assign_cval() : 0
        yield from self.host.processor.assign_instr_index(process, 0)
        # run_request() : req0
        yield from self.run_request(process, 1, "req0")
        # run_request() : req1
        yield from self.run_request(process, 2, "req1")

        if self._part == "only_rsp":
            self.finished = True
            return

        # run_subroutine(tuple<client_id>) : local_cphase
        yield from self.run_subroutine(process, 3, "local_cphase")
        # delta1 = recv_cmsg(client_id)
        yield from self.host.processor.assign_instr_index(process, 4)
        # tuple<m1> = run_subroutine(tuple<delta1>) : meas_qubit_1
        yield from self.run_subroutine(process, 5, "meas_qubit_1")
        # send_cmsg(csocket, m1)
        yield from self.host.processor.assign_instr_index(process, 6)
        # delta2 = recv_cmsg(csocket)
        yield from self.host.processor.assign_instr_index(process, 7)

        if self._part == "meas_first_epr":
            self.finished = True
            return

        # tuple<m2> = run_subroutine(tuple<delta2>) : meas_qubit_0
        yield from self.run_subroutine(process, 8, "meas_qubit_0")
        # return_result(m1)
        yield from self.host.processor.assign_instr_index(process, 9)
        # return_result(m2)
        yield from self.host.processor.assign_instr_index(process, 10)

        assert self._part == "full"
        self.finished = True


class ClientProcNode(BqcProcNode):
    def __init__(self, part: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._part = part

    def run(self) -> Generator[EventExpression, None, None]:
        self.finished = False

        process = self.memmgr.get_process(self.pid)
        self.scheduler.initialize_process(process)

        # csocket = assign_cval() : 0
        yield from self.host.processor.assign_instr_index(process, 0)
        # run_request() : req0
        yield from self.run_request(process, 1, "req0")
        # run_subroutine(tuple<theta2>) : post_epr_0
        yield from self.run_subroutine(process, 2, "post_epr_0")
        # run_request() : req1
        yield from self.run_request(process, 3, "req1")
        # run_subroutine(tuple<theta1>) : post_epr_1
        yield from self.run_subroutine(process, 4, "post_epr_1")
        # x = mult_const(p1) : 16
        # minus_theta1 = mult_const(theta1) : -1
        # delta1 = add_cval_c(minus_theta1, x)
        # delta1 = add_cval_c(delta1, alpha)
        # send_cmsg(server_id, delta1)
        for i in range(5, 10):
            yield from self.host.processor.assign_instr_index(process, i)

        if self._part == "only_rsp":
            self.finished = True
            return

        # m1 = recv_cmsg(csocket)
        yield from self.host.processor.assign_instr_index(process, 10)
        # y = mult_const(p2) : 16
        # minus_theta2 = mult_const(theta2) : -1
        # beta = bcond_mult_const(beta, m1) : -1
        # delta2 = add_cval_c(beta, minus_theta2)
        # delta2 = add_cval_c(delta2, y)
        # send_cmsg(csocket, delta2)
        for i in range(11, 17):
            yield from self.host.processor.assign_instr_index(process, i)

        if self._part == "meas_first_epr":
            self.finished = True
            return

        # return_result(p1)
        yield from self.host.processor.assign_instr_index(process, 17)
        # return_result(p2)
        yield from self.host.processor.assign_instr_index(process, 18)

        assert self._part == "full"
        self.finished = True


@dataclass
class BqcResult:
    client_process: QoalaProcess
    server_process: QoalaProcess
    client_procnode: BqcProcNode
    server_procnode: BqcProcNode


def run_bqc(
    part: str,
    alpha,
    beta,
    theta1,
    theta2,
    client_pid: int = 0,
    server_pid: int = 0,
):
    ns.sim_reset()

    num_qubits = 3
    nodes = {0: "client", 1: "server"}

    link_info = LhiLinkInfo.perfect(1000)
    ehi_link = LhiConverter.link_info_to_ehi(link_info)
    network_ehi = EhiNetworkInfo.fully_connected(nodes, ehi_link)
    server_id = network_ehi.get_node_id("server")
    client_id = network_ehi.get_node_id("client")

    path = os.path.join(os.path.dirname(__file__), "bqc_server.iqoala")
    with open(path) as file:
        server_text = file.read()
    server_program = QoalaParser(server_text).parse()

    server_procnode = create_procnode(
        part,
        "server",
        num_qubits,
        network_ehi,
        ServerProcNode,
        pid=server_pid,
    )
    server_ehi = server_procnode.memmgr.get_ehi()
    server_process = create_process(
        pid=server_pid,
        remote_pid=client_pid,
        program=server_program,
        unit_module=UnitModule.from_full_ehi(server_ehi),
        host_interface=server_procnode.host._interface,
        network_ehi=network_ehi,
        inputs={"client_id": client_id},
    )
    server_procnode.add_process(server_process)

    path = os.path.join(os.path.dirname(__file__), "bqc_client.iqoala")
    with open(path) as file:
        client_text = file.read()
    client_program = QoalaParser(client_text).parse()

    client_procnode = create_procnode(
        part,
        "client",
        num_qubits,
        network_ehi,
        ClientProcNode,
        pid=client_pid,
    )
    client_ehi = client_procnode.memmgr.get_ehi()
    client_process = create_process(
        pid=client_pid,
        remote_pid=server_pid,
        program=client_program,
        unit_module=UnitModule.from_full_ehi(client_ehi),
        host_interface=client_procnode.host._interface,
        network_ehi=network_ehi,
        inputs={
            "server_id": server_id,
            "alpha": alpha,
            "beta": beta,
            "theta1": theta1,
            "theta2": theta2,
        },
    )
    client_procnode.add_process(client_process)

    client_procnode.connect_to(server_procnode)

    nodes = [client_procnode.node, server_procnode.node]
    entdistcomp = EntDistComponent(network_ehi)
    client_procnode.node.entdist_out_port.connect(entdistcomp.node_in_port("client"))
    client_procnode.node.entdist_in_port.connect(entdistcomp.node_out_port("client"))
    server_procnode.node.entdist_out_port.connect(entdistcomp.node_in_port("server"))
    server_procnode.node.entdist_in_port.connect(entdistcomp.node_out_port("server"))
    entdist = EntDist(nodes=nodes, ehi_network=network_ehi, comp=entdistcomp)
    entdist.add_sampler(client_procnode.node.ID, server_procnode.node.ID, link_info)

    # To prevent scheduler from running
    server_procnode.scheduler = MockScheduler(server_procnode.scheduler)
    client_procnode.scheduler = MockScheduler(client_procnode.scheduler)

    server_procnode.start()
    client_procnode.start()
    entdist.start()
    ns.sim_run()

    assert client_procnode.finished
    assert server_procnode.finished

    return BqcResult(
        client_process=client_process,
        server_process=server_process,
        client_procnode=client_procnode,
        server_procnode=server_procnode,
    )


def expected_rsp_state(theta: int, p: int, dummy: bool):
    expected = qubitapi.create_qubits(1)[0]

    if dummy:
        if p == 0:
            return ketstates.s0
        elif p == 1:
            return ketstates.s1
    else:
        if (theta, p) == (0, 0):
            return ketstates.h0
        elif (theta, p) == (0, 1):
            return ketstates.h1
        if (theta, p) == (8, 0):
            return ketstates.y0
        elif (theta, p) == (8, 1):
            return ketstates.y1
        if (theta, p) == (16, 0):
            return ketstates.h1
        elif (theta, p) == (16, 1):
            return ketstates.h0
        if (theta, p) == (-8, 0):
            return ketstates.y1
        elif (theta, p) == (-8, 1):
            return ketstates.y0

    return expected.qstate


def test_bqc_only_rsp():
    ns.sim_reset()

    alpha_beta_theta1_theta2 = [
        (0, 0, 0, 0),
        (0, 8, 0, 0),
        (8, 0, 0, 0),
        (8, 8, 0, 0),
    ]

    for alpha, beta, theta1, theta2 in alpha_beta_theta1_theta2:
        for _ in range(10):
            result = run_bqc("only_rsp", alpha, beta, theta1, theta2)

            p1 = result.client_process.host_mem.read("p1")
            p2 = result.client_process.host_mem.read("p2")
            delta1 = result.client_process.host_mem.read("delta1")
            q0 = result.server_procnode.qdevice.get_local_qubit(0)
            q1 = result.server_procnode.qdevice.get_local_qubit(1)

            assert delta1 == alpha - theta1 + p1 * 16

            # p2 and theta2 control state of q0
            expected_q0 = expected_rsp_state(theta2, p2, dummy=False)
            assert has_state(q0, expected_q0)

            # p1 and theta1 control state of q1
            expected_q1 = expected_rsp_state(theta1, p1, dummy=False)
            assert has_state(q1, expected_q1)


def test_bqc_only_meas_first_EPR():
    ns.sim_reset()

    alpha_beta_theta1_theta2 = [
        (0, 0, 0, 0),
        (0, 8, 0, 0),
        (8, 0, 0, 0),
        (8, 8, 0, 0),
    ]

    for alpha, beta, theta1, theta2 in alpha_beta_theta1_theta2:
        for _ in range(10):
            result = run_bqc("meas_first_epr", alpha, beta, theta1, theta2)

            p1 = result.client_process.host_mem.read("p1")
            p2 = result.client_process.host_mem.read("p2")
            delta1 = result.client_process.host_mem.read("delta1")
            m1 = result.client_process.host_mem.read("m1")
            delta2 = result.client_process.host_mem.read("delta2")

            q0 = result.server_procnode.qdevice.get_local_qubit(0)
            print(q0.qstate)

            assert delta1 == alpha - theta1 + p1 * 16
            assert delta2 == math.pow(-1, m1) * beta - theta2 + p2 * 16


def test_bqc_full():
    # Effective computation: measure in Z the following state:
    # H Rz(beta) H Rz(alpha) |+>
    # m2 should be this outcome

    # angles are in multiples of pi/16

    def check(alpha, beta, theta1, theta2, expected):
        ns.sim_reset()

        results = [
            run_bqc("full", alpha, beta, theta1, theta2, client_pid=i, server_pid=i)
            for i in range(2)
        ]
        m2s = [result.server_process.result.values["m2"] for result in results]
        assert all(m2 == expected for m2 in m2s)

    check(alpha=8, beta=8, theta1=0, theta2=0, expected=0)
    check(alpha=8, beta=24, theta1=0, theta2=0, expected=1)
    check(alpha=8, beta=8, theta1=13, theta2=27, expected=0)
    check(alpha=8, beta=24, theta1=2, theta2=22, expected=1)


if __name__ == "__main__":
    test_bqc_only_rsp()
    test_bqc_only_meas_first_EPR()
    test_bqc_full()
