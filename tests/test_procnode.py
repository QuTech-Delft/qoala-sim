from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Type

import netsquid as ns
from netqasm.lang.parsing import parse_text_subroutine
from netsquid.components import QuantumProcessor
from netsquid.qubits import ketstates

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, UnitModule
from qoala.lang.hostlang import (
    AssignCValueOp,
    BasicBlock,
    BasicBlockType,
    ClassicalIqoalaOp,
    IqoalaSingleton,
    ReceiveCMsgOp,
    SendCMsgOp,
)
from qoala.lang.parse import LocalRoutineParser, QoalaParser
from qoala.lang.program import LocalRoutine, ProgramMeta, QoalaProgram
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    QoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
)
from qoala.lang.routine import RoutineMetadata
from qoala.runtime.lhi import LhiLatencies, LhiLinkInfo, LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message, RrCallTuple
from qoala.runtime.ntf import GenericNtf, NtfInterface
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist.entdist import EntDist
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.eprsocket import EprSocket
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import NetstackInterface
from qoala.sim.process import QoalaProcess
from qoala.sim.procnode import ProcNode
from qoala.sim.qdevice import QDevice, QDeviceCommand
from qoala.sim.scheduler import NodeScheduler
from qoala.util.math import has_multi_state
from qoala.util.tests import netsquid_run

MOCK_MESSAGE = Message(src_pid=0, dst_pid=0, content=42)
MOCK_QNOS_RET_REG = "R0"
MOCK_QNOS_RET_VALUE = 7


@dataclass(frozen=True)
class InterfaceEvent:
    peer: str
    msg: Message


@dataclass(frozen=True)
class FlushEvent:
    pass


@dataclass(frozen=True)
class SignalEvent:
    pass


class MockNetstackInterface(NetstackInterface):
    def __init__(
        self,
        qdevice: QDevice,
        memmgr: MemoryManager,
    ) -> None:
        self._qdevice = qdevice
        self._memmgr = memmgr

    def send_qnos_msg(self, msg: Message) -> None:
        return None

    def send_peer_msg(self, peer: str, msg: Message) -> None:
        return None

    def receive_peer_msg(self, peer: str) -> Generator[EventExpression, None, Message]:
        return None
        yield  # to make this behave as a generator

    def send_entdist_msg(self, msg: Message) -> None:
        return None

    def receive_entdist_msg(self) -> Generator[EventExpression, None, Message]:
        return Message(-1, -1, 0)
        yield  # to make this behave as a generator

    @property
    def node_id(self) -> int:
        return 0


class MockQDevice(QDevice):
    def __init__(self, topology: LhiTopology) -> None:
        self._topology = topology

        self._executed_commands: List[QDeviceCommand] = []

    def set_mem_pos_in_use(self, id: int, in_use: bool) -> None:
        pass

    def execute_commands(
        self, commands: List[QDeviceCommand]
    ) -> Generator[EventExpression, None, Optional[int]]:
        self._executed_commands.extend(commands)
        return None
        yield

    def reset(self) -> None:
        self._executed_commands = []


class MockHostInterface(HostInterface):
    def __init__(self) -> None:
        self.send_events: List[InterfaceEvent] = []
        self.recv_events: List[InterfaceEvent] = []
        self._program_instance_jumps: Dict[int, int] = {}  # pid => block name

    @property
    def program_instance_jumps(self) -> Dict[int, int]:
        return self._program_instance_jumps

    def send_peer_msg(self, peer: str, msg: Message) -> None:
        self.send_events.append(InterfaceEvent(peer, msg))

    def receive_peer_msg(self, peer: str) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent(peer, MOCK_MESSAGE))
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    def send_qnos_msg(self, msg: Message) -> None:
        self.send_events.append(InterfaceEvent("qnos", msg))

    def receive_qnos_msg(self) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent("qnos", MOCK_MESSAGE))
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    @property
    def name(self) -> str:
        return "mock"


def create_program(
    instrs: Optional[List[ClassicalIqoalaOp]] = None,
    subroutines: Optional[Dict[str, LocalRoutine]] = None,
    req_routines: Optional[Dict[str, RequestRoutine]] = None,
    meta: Optional[ProgramMeta] = None,
) -> QoalaProgram:
    if instrs is None:
        instrs = []
    if subroutines is None:
        subroutines = {}

    if req_routines is None:
        req_routines = {}
    if meta is None:
        meta = ProgramMeta.empty("prog")
    # TODO: split into proper blocks
    block = BasicBlock("b0", BasicBlockType.CL, instrs)
    return QoalaProgram(
        blocks=[block],
        local_routines=subroutines,
        meta=meta,
        request_routines=req_routines,
    )


def create_process(
    pid: int,
    program: QoalaProgram,
    unit_module: UnitModule,
    host_interface: HostInterface,
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
            id: ClassicalSocket(host_interface, name, 0, 0)
            for (id, name) in program.meta.csockets.items()
        },
        epr_sockets={0: EprSocket(0, 0, 0, 0, 1.0)},
        result=ProgramResult(values={}),
    )
    return process


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


def create_qprocessor(name: str, num_qubits: int) -> QuantumProcessor:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits=num_qubits)
    return build_qprocessor_from_topology(name=f"{name}_processor", topology=topology)


def create_network_ehi(num_qubits: int, names: List[str] = None) -> EhiNetworkInfo:
    if names is None:
        names = ["alice", "bob", "charlie"]

    nodes = {i: name for i, name in enumerate(names)}
    return EhiNetworkInfo(nodes, links={})


def create_procnode(
    name: str,
    network_ehi: EhiNetworkInfo,
    num_qubits: int,
    topology: LhiTopology,
    latencies: LhiLatencies,
    ntf_interface: NtfInterface,
    procnode_cls: Type[ProcNode] = ProcNode,
    asynchronous: bool = False,
) -> ProcNode:
    alice_qprocessor = create_qprocessor(name, num_qubits)

    node_id = network_ehi.get_node_id(name)
    procnode = procnode_cls(
        name=name,
        qprocessor=alice_qprocessor,
        qdevice_topology=topology,
        latencies=latencies,
        ntf_interface=ntf_interface,
        network_ehi=network_ehi,
        node_id=node_id,
        asynchronous=asynchronous,
    )

    return procnode


def simple_subroutine(name: str, subrt_text: str) -> LocalRoutine:
    subrt = parse_text_subroutine(subrt_text)
    return LocalRoutine(
        name, subrt, return_vars=[], metadata=RoutineMetadata.use_none()
    )


def parse_iqoala_subroutines(subrt_text: str) -> LocalRoutine:
    return LocalRoutineParser(subrt_text).parse()


def generic_topology(num_qubits: int) -> LhiTopology:
    # Instructions and durations are not needed for these tests.
    return LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[],
        single_duration=0,
        two_instructions=[],
        two_duration=0,
    )


def star_topology(num_qubits: int) -> LhiTopology:
    # Instructions and durations are not needed for these tests.
    return LhiTopologyBuilder.perfect_star(
        num_qubits=num_qubits,
        comm_instructions=[],
        comm_duration=0,
        mem_instructions=[],
        mem_duration=0,
        two_instructions=[],
        two_duration=0,
    )


def test_initialize():
    num_qubits = 3
    topology = generic_topology(num_qubits)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()

    network_ehi = create_network_ehi(num_qubits)
    procnode = create_procnode(
        "alice", network_ehi, num_qubits, topology, latencies, ntf
    )
    procnode.qdevice = MockQDevice(topology)

    procnode.host.interface = MockHostInterface()

    procnode.netstack.interface = MockNetstackInterface(
        procnode.qdevice, procnode.memmgr
    )

    host_processor = procnode.host.processor
    qnos_processor = procnode.qnos.processor
    netstack_processor = procnode.netstack.processor

    ehi = LhiConverter.to_ehi(topology, ntf, latencies)
    unit_module = UnitModule.from_full_ehi(ehi)

    instrs = [AssignCValueOp(IqoalaSingleton("x"), 3)]
    subrt1 = simple_subroutine(
        "subrt1",
        """
    set R5 42
    """,
    )

    request_routine = RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=QoalaRequest(
            name="req1",
            remote_id=1,
            epr_socket_id=0,
            num_pairs=1,
            virt_ids=RequestVirtIdMapping.from_str("increment 0"),
            timeout=1000,
            fidelity=1.0,
            typ=EprType.CREATE_KEEP,
            role=EprRole.CREATE,
        ),
    )

    program = create_program(
        instrs=instrs,
        subroutines={"subrt1": subrt1},
        req_routines={"req1": request_routine},
    )
    process = create_process(
        pid=0,
        program=program,
        unit_module=unit_module,
        host_interface=procnode.host._interface,
        inputs={"x": 1, "theta": 3.14, "name": "alice"},
    )
    procnode.add_process(process)

    host_processor.initialize(process)
    host_mem = process.prog_memory.host_mem
    assert host_mem.read("x") == 1
    assert host_mem.read("theta") == 3.14
    assert host_mem.read("name") == "alice"

    netsquid_run(host_processor.assign_instr_index(process, instr_idx=0))
    routine = process.program.local_routines["subrt1"]
    qnos_processor.instantiate_routine(process, routine, {}, 0, 0)
    netsquid_run(qnos_processor.assign_routine_instr(process, "subrt1", 0))

    rrcall = RrCallTuple.no_alloc("req1")
    global_args = process.prog_instance.inputs.values
    netstack_processor.instantiate_routine(process, rrcall, global_args)
    netsquid_run(
        netstack_processor.handle_single_pair(
            process, "req1", 0
        )  # not s  ure about the 0
    )

    assert process.host_mem.read("x") == 3
    assert process.qnos_mem.get_reg_value("R5") == 42
    assert procnode.memmgr.phys_id_for(pid=process.pid, virt_id=0) == 0


def test_classical_comm():
    ns.sim_reset()

    num_qubits = 3

    topology = generic_topology(num_qubits)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()

    network_ehi = create_network_ehi(num_qubits)

    class TestProcNode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            process = self.memmgr.get_process(0)
            yield from self.host.processor.assign_instr_index(process, 0)

    alice_procnode = create_procnode(
        "alice",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )
    bob_procnode = create_procnode(
        "bob",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )

    alice_host_processor = alice_procnode.host.processor
    bob_host_processor = bob_procnode.host.processor

    ehi = LhiConverter.to_ehi(topology, ntf, latencies)
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_instrs = [
        SendCMsgOp(IqoalaSingleton("csocket_id"), IqoalaSingleton("message"))
    ]
    alice_meta = ProgramMeta(
        name="alice",
        parameters=["csocket_id", "message"],
        csockets={0: "bob"},
        epr_sockets={},
    )
    alice_program = create_program(instrs=alice_instrs, meta=alice_meta)
    alice_process = create_process(
        pid=0,
        program=alice_program,
        unit_module=unit_module,
        host_interface=alice_procnode.host._interface,
        inputs={"csocket_id": 0, "message": 1337},
    )
    alice_procnode.add_process(alice_process)
    alice_host_processor.initialize(alice_process)

    bob_instrs = [
        ReceiveCMsgOp(IqoalaSingleton("csocket_id"), IqoalaSingleton("result"))
    ]
    bob_meta = ProgramMeta(
        name="bob", parameters=["csocket_id"], csockets={0: "alice"}, epr_sockets={}
    )
    bob_program = create_program(instrs=bob_instrs, meta=bob_meta)
    bob_process = create_process(
        pid=0,
        program=bob_program,
        unit_module=unit_module,
        host_interface=bob_procnode.host._interface,
        inputs={"csocket_id": 0},
    )
    bob_procnode.add_process(bob_process)
    bob_host_processor.initialize(bob_process)

    alice_procnode.connect_to(bob_procnode)

    # To prevent scheduler from running
    alice_procnode.scheduler = MockScheduler(alice_procnode.scheduler)
    bob_procnode.scheduler = MockScheduler(bob_procnode.scheduler)

    # First start Bob, since Alice won't yield on anything (she only does a Send
    # instruction) and therefore calling 'start()' on alice completes her whole
    # protocol while Bob's interface has not even been started.

    bob_procnode.start()
    alice_procnode.start()

    ns.sim_run()

    assert bob_process.host_mem.read("result") == 1337


def test_classical_comm_three_nodes():
    ns.sim_reset()

    num_qubits = 3

    topology = generic_topology(num_qubits)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()

    network_ehi = create_network_ehi(num_qubits)

    class SenderProcNode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            process = self.memmgr.get_process(0)
            yield from self.host.processor.assign_instr_index(process, 0)

    class ReceiverProcNode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            process = self.memmgr.get_process(0)
            yield from self.host.processor.assign_instr_index(process, 0)
            yield from self.host.processor.assign_instr_index(process, 1)

    alice_procnode = create_procnode(
        "alice",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=SenderProcNode,
    )
    bob_procnode = create_procnode(
        "bob",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=SenderProcNode,
    )
    charlie_procnode = create_procnode(
        "charlie",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=ReceiverProcNode,
    )

    alice_host_processor = alice_procnode.host.processor
    bob_host_processor = bob_procnode.host.processor
    charlie_host_processor = charlie_procnode.host.processor

    ehi = LhiConverter.to_ehi(topology, ntf, latencies)
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_instrs = [
        SendCMsgOp(IqoalaSingleton("csocket_id"), IqoalaSingleton("message"))
    ]
    alice_meta = ProgramMeta(
        name="alice",
        parameters=["csocket_id", "message"],
        csockets={0: "charlie"},
        epr_sockets={},
    )
    alice_program = create_program(instrs=alice_instrs, meta=alice_meta)
    alice_process = create_process(
        pid=0,
        program=alice_program,
        unit_module=unit_module,
        host_interface=alice_procnode.host._interface,
        inputs={"csocket_id": 0, "message": 1337},
    )
    alice_procnode.add_process(alice_process)
    alice_host_processor.initialize(alice_process)

    bob_instrs = [SendCMsgOp(IqoalaSingleton("csocket_id"), IqoalaSingleton("message"))]
    bob_meta = ProgramMeta(
        name="bob",
        parameters=["csocket_id", "message"],
        csockets={0: "charlie"},
        epr_sockets={},
    )
    bob_program = create_program(instrs=bob_instrs, meta=bob_meta)
    bob_process = create_process(
        pid=0,
        program=bob_program,
        unit_module=unit_module,
        host_interface=bob_procnode.host._interface,
        inputs={"csocket_id": 0, "message": 42},
    )
    bob_procnode.add_process(bob_process)
    bob_host_processor.initialize(bob_process)

    charlie_instrs = [
        ReceiveCMsgOp(
            IqoalaSingleton("csocket_id_alice"), IqoalaSingleton("result_alice")
        ),
        ReceiveCMsgOp(IqoalaSingleton("csocket_id_bob"), IqoalaSingleton("result_bob")),
    ]
    charlie_meta = ProgramMeta(
        name="bob",
        parameters=["csocket_id_alice", "csocket_id_bob"],
        csockets={0: "alice", 1: "bob"},
        epr_sockets={},
    )
    charlie_program = create_program(instrs=charlie_instrs, meta=charlie_meta)
    charlie_process = create_process(
        pid=0,
        program=charlie_program,
        unit_module=unit_module,
        host_interface=charlie_procnode.host._interface,
        inputs={"csocket_id_alice": 0, "csocket_id_bob": 1},
    )
    charlie_procnode.add_process(charlie_process)
    charlie_host_processor.initialize(charlie_process)

    alice_procnode.connect_to(charlie_procnode)
    bob_procnode.connect_to(charlie_procnode)

    # To prevent scheduler from running
    alice_procnode.scheduler = MockScheduler(alice_procnode.scheduler)
    bob_procnode.scheduler = MockScheduler(bob_procnode.scheduler)
    charlie_procnode.scheduler = MockScheduler(charlie_procnode.scheduler)

    # First start Charlie, since Alice and Bob don't yield on anything.
    charlie_procnode.start()
    alice_procnode.start()
    bob_procnode.start()
    ns.sim_run()

    assert charlie_process.host_mem.read("result_alice") == 1337
    assert charlie_process.host_mem.read("result_bob") == 42


def test_epr():
    ns.sim_reset()

    num_qubits = 3

    topology = generic_topology(num_qubits)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()

    network_ehi = create_network_ehi(num_qubits)
    alice_id = network_ehi.get_node_id("alice")
    bob_id = network_ehi.get_node_id("bob")

    class TestProcNode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            process = self.memmgr.get_process(0)
            rrcall = RrCallTuple.no_alloc("req1")
            global_args = process.prog_instance.inputs.values
            self.netstack.processor.instantiate_routine(process, rrcall, global_args)
            yield from self.netstack.processor.handle_single_pair(
                process, "req1", 0
            )  # not s  ure about the 0

    alice_procnode = create_procnode(
        "alice",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )
    bob_procnode = create_procnode(
        "bob",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )

    alice_host_processor = alice_procnode.host.processor
    bob_host_processor = bob_procnode.host.processor

    ehi = LhiConverter.to_ehi(topology, ntf, latencies)
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_request_routine = RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=QoalaRequest(
            name="req1",
            remote_id=bob_id,
            epr_socket_id=0,
            num_pairs=1,
            virt_ids=RequestVirtIdMapping.from_str("increment 0"),
            timeout=1000,
            fidelity=1.0,
            typ=EprType.CREATE_KEEP,
            role=EprRole.CREATE,
        ),
    )

    bob_request_routine = RequestRoutine(
        name="req1",
        return_vars=[],
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
        request=QoalaRequest(
            name="req1",
            remote_id=alice_id,
            epr_socket_id=0,
            num_pairs=1,
            virt_ids=RequestVirtIdMapping.from_str("increment 0"),
            timeout=1000,
            fidelity=1.0,
            typ=EprType.CREATE_KEEP,
            role=EprRole.RECEIVE,
        ),
    )

    alice_instrs = [SendCMsgOp("csocket_id", "message")]
    alice_meta = ProgramMeta(
        name="alice",
        parameters=["csocket_id", "message"],
        csockets={0: "bob"},
        epr_sockets={},
    )
    alice_program = create_program(
        instrs=alice_instrs,
        req_routines={"req1": alice_request_routine},
        meta=alice_meta,
    )
    alice_process = create_process(
        pid=0,
        program=alice_program,
        unit_module=unit_module,
        host_interface=alice_procnode.host._interface,
        inputs={"csocket_id": 0, "message": 1337},
    )
    alice_procnode.add_process(alice_process)
    alice_host_processor.initialize(alice_process)

    bob_instrs = [ReceiveCMsgOp("csocket_id", "result")]
    bob_meta = ProgramMeta(
        name="bob", parameters=["csocket_id"], csockets={0: "alice"}, epr_sockets={}
    )
    bob_program = create_program(
        instrs=bob_instrs, req_routines={"req1": bob_request_routine}, meta=bob_meta
    )
    bob_process = create_process(
        pid=0,
        program=bob_program,
        unit_module=unit_module,
        host_interface=bob_procnode.host._interface,
        inputs={"csocket_id": 0},
    )
    bob_procnode.add_process(bob_process)
    bob_host_processor.initialize(bob_process)

    alice_procnode.connect_to(bob_procnode)

    nodes = [alice_procnode.node, bob_procnode.node]
    entdistcomp = EntDistComponent(network_ehi)
    alice_procnode.node.entdist_out_port.connect(entdistcomp.node_in_port("alice"))
    alice_procnode.node.entdist_in_port.connect(entdistcomp.node_out_port("alice"))
    bob_procnode.node.entdist_out_port.connect(entdistcomp.node_in_port("bob"))
    bob_procnode.node.entdist_in_port.connect(entdistcomp.node_out_port("bob"))
    entdist = EntDist(nodes=nodes, ehi_network=network_ehi, comp=entdistcomp)
    link_info = LhiLinkInfo.perfect(1000)
    entdist.add_sampler(alice_procnode.node.ID, bob_procnode.node.ID, link_info)

    # To prevent scheduler from running
    alice_procnode.scheduler = MockScheduler(alice_procnode.scheduler)
    bob_procnode.scheduler = MockScheduler(bob_procnode.scheduler)

    # First start Bob, since Alice won't yield on anything (she only does a Send
    # instruction) and therefore calling 'start()' on alice completes her whole
    # protocol while Bob's interface has not even been started.
    bob_procnode.start()
    alice_procnode.start()
    entdist.start()
    ns.sim_run()

    assert alice_procnode.memmgr.phys_id_for(pid=0, virt_id=0) == 0
    assert bob_procnode.memmgr.phys_id_for(pid=0, virt_id=0) == 0

    alice_qubit = alice_procnode.qdevice.get_local_qubit(0)
    bob_qubit = bob_procnode.qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)


def test_whole_program():
    ns.sim_reset()

    server_text = """
META_START
    name: server
    parameters: client_id
    csockets: 0 -> client
    epr_sockets: 0 -> client
META_END

^b0 {type = CL}:
    remote_id = assign_cval() : {client_id}
^b1 {type = QC}:
    run_request() : req1

REQUEST req1
  callback_type: wait_all
  callback:
  return_vars: 
  remote_id: {client_id}
  epr_socket_id: 0
  num_pairs: 1
  virt_ids: all 0
  timeout:1000
  fidelity: 1.0
  typ: create_keep
  role: receive
    """
    server_program = QoalaParser(server_text).parse()

    num_qubits = 3
    topology = generic_topology(num_qubits)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()
    ehi = LhiConverter.to_ehi(topology, ntf, latencies)
    unit_module = UnitModule.from_full_ehi(ehi)
    network_ehi = create_network_ehi(num_qubits, names=["client", "server"])
    server_id = network_ehi.get_node_id("server")
    client_id = network_ehi.get_node_id("client")

    class TestProcNode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            process = self.memmgr.get_process(0)
            self.scheduler.initialize_process(process)
            rrcall = RrCallTuple.no_alloc("req1")
            global_args = process.prog_instance.inputs.values
            self.netstack.processor.instantiate_routine(process, rrcall, global_args)
            yield from self.netstack.processor.handle_single_pair(
                process, "req1", 0
            )  # not s  ure about the 0

    server_procnode = create_procnode(
        "server",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )
    server_process = create_process(
        pid=0,
        program=server_program,
        unit_module=unit_module,
        host_interface=server_procnode.host._interface,
        inputs={"client_id": client_id, "csocket_id": 0, "message": 1337},
    )
    server_procnode.add_process(server_process)

    client_text = """
META_START
    name: client
    parameters: server_id
    csockets: 0 -> server
    epr_sockets: 0 -> server
META_END

^b0 {type = CL}:
    remote_id = assign_cval() : {server_id}
^b1 {type = QL}:
    run_request() : req1

REQUEST req1
  callback_type: wait_all
  callback:
  return_vars: 
  remote_id: {server_id}
  epr_socket_id: 0
  num_pairs: 1
  virt_ids: all 0
  timeout: 1000
  fidelity: 1.0
  typ: create_keep
  role: create
    """
    client_program = QoalaParser(client_text).parse()
    client_procnode = create_procnode(
        "client",
        network_ehi,
        num_qubits,
        topology,
        latencies,
        ntf,
        procnode_cls=TestProcNode,
    )
    client_process = create_process(
        pid=0,
        program=client_program,
        unit_module=unit_module,
        host_interface=client_procnode.host._interface,
        inputs={"server_id": server_id, "csocket_id": 0, "message": 1337},
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
    link_info = LhiLinkInfo.perfect(1000)
    entdist.add_sampler(client_procnode.node.ID, server_procnode.node.ID, link_info)

    # To prevent scheduler from running
    server_procnode.scheduler = MockScheduler(server_procnode.scheduler)
    client_procnode.scheduler = MockScheduler(client_procnode.scheduler)

    server_procnode.start()
    client_procnode.start()
    entdist.start()
    ns.sim_run()

    assert client_procnode.memmgr.phys_id_for(pid=0, virt_id=0) == 0
    assert server_procnode.memmgr.phys_id_for(pid=0, virt_id=0) == 0
    client_qubit = client_procnode.qdevice.get_local_qubit(0)
    server_qubit = server_procnode.qdevice.get_local_qubit(0)
    assert has_multi_state([client_qubit, server_qubit], ketstates.b00)

    # assert client_procnode.memmgr.phys_id_for(pid=0, virt_id=1) == 1
    # assert server_procnode.memmgr.phys_id_for(pid=0, virt_id=1) == 1


if __name__ == "__main__":
    test_initialize()
    test_classical_comm()
    test_classical_comm_three_nodes()
    test_epr()
    test_whole_program()
