from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import netsquid as ns
from netqasm.lang.parsing import parse_text_subroutine
from netqasm.lang.subroutine import Subroutine

from pydynaa import EventExpression
from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.hostlang import (
    AddCValueOp,
    AssignCValueOp,
    BasicBlock,
    BasicBlockType,
    BitConditionalMultiplyConstantCValueOp,
    ClassicalIqoalaOp,
    IqoalaVector,
    MultiplyConstantCValueOp,
    ReceiveCMsgOp,
    ReturnResultOp,
    RunRequestOp,
    RunSubroutineOp,
    SendCMsgOp,
)
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    IqoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
)
from qoala.lang.routine import LocalRoutine, RoutineMetadata
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import LrCallTuple, Message, RrCallTuple
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.sharedmem import SharedMemoryManager
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostinterface import HostInterface, HostLatencies
from qoala.sim.host.hostprocessor import HostProcessor
from qoala.sim.process import IqoalaProcess
from qoala.util.tests import netsquid_run, yield_from

MOCK_MESSAGE = Message(content=42)
MOCK_QNOS_RET_REG = "R0"
MOCK_QNOS_RET_VALUE = 7
MOCK_NETSTACK_RET_VALUE = 22


@dataclass(eq=True, frozen=True)
class InterfaceEvent:
    peer: str
    msg: Message


class MockHostInterface(HostInterface):
    def __init__(self, shared_mem: Optional[SharedMemoryManager] = None) -> None:
        self.send_events: List[InterfaceEvent] = []
        self.recv_events: List[InterfaceEvent] = []

        self.shared_mem = shared_mem

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
        if self.shared_mem is not None:
            # Hack to find out which addr was allocated to write results to.
            result_addr = self.shared_mem._lr_out_addrs[0]
            self.shared_mem.write_lr_out(result_addr, [MOCK_QNOS_RET_VALUE])
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    def send_netstack_msg(self, msg: Message) -> None:
        self.send_events.append(InterfaceEvent("netstack", msg))

    def receive_netstack_msg(self) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent("netstack", MOCK_MESSAGE))
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    @property
    def name(self) -> str:
        return "mock"


def create_program(
    instrs: Optional[List[ClassicalIqoalaOp]] = None,
    subroutines: Optional[Dict[str, LocalRoutine]] = None,
    requests: Optional[Dict[str, RequestRoutine]] = None,
    meta: Optional[ProgramMeta] = None,
) -> IqoalaProgram:
    if instrs is None:
        instrs = []
    if subroutines is None:
        subroutines = {}
    if requests is None:
        requests = {}
    if meta is None:
        meta = ProgramMeta.empty("prog")
    # TODO: split into proper blocks
    block = BasicBlock("b0", BasicBlockType.CL, instrs)
    return IqoalaProgram(
        blocks=[block], local_routines=subroutines, request_routines=requests, meta=meta
    )


def create_process(
    program: IqoalaProgram,
    interface: HostInterface,
    inputs: Optional[Dict[str, Any]] = None,
) -> IqoalaProcess:
    if inputs is None:
        inputs = {}
    prog_input = ProgramInput(values=inputs)

    mock_ehi = EhiBuilder.perfect_uniform(
        num_qubits=2,
        flavour=None,
        single_instructions=[],
        single_duration=0,
        two_instructions=[],
        two_duration=0,
    )

    instance = ProgramInstance(
        pid=0,
        program=program,
        inputs=prog_input,
        unit_module=UnitModule.from_full_ehi(mock_ehi),
        block_tasks=[],
    )

    mem = ProgramMemory(pid=0)
    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={
            id: ClassicalSocket(interface, name)
            for (id, name) in program.meta.csockets.items()
        },
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def test_initialize():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program()
    process = create_process(
        program, interface, inputs={"x": 1, "theta": 3.14, "name": "alice"}
    )

    processor.initialize(process)
    host_mem = process.prog_memory.host_mem
    assert host_mem.read("x") == 1
    assert host_mem.read("theta") == 3.14
    assert host_mem.read("name") == "alice"


def test_assign_cvalue():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program(instrs=[AssignCValueOp("x", 3)])
    process = create_process(program, interface)
    processor.initialize(process)

    yield_from(processor.assign_instr_index(process, 0))
    assert process.prog_memory.host_mem.read("x") == 3


def test_assign_cvalue_with_latencies():
    ns.sim_reset()

    interface = MockHostInterface()
    latencies = HostLatencies(host_instr_time=500)
    processor = HostProcessor(interface, latencies)
    program = create_program(instrs=[AssignCValueOp("x", 3)])
    process = create_process(program, interface)
    processor.initialize(process)

    assert ns.sim_time() == 0
    netsquid_run(processor.assign_instr_index(process, 0))
    assert process.prog_memory.host_mem.read("x") == 3
    assert ns.sim_time() == 500


def test_send_msg():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    meta = ProgramMeta.empty("alice")
    meta.csockets = {0: "bob"}
    program = create_program(instrs=[SendCMsgOp("bob", "msg")], meta=meta)
    process = create_process(program, interface, inputs={"bob": 0, "msg": 12})
    processor.initialize(process)

    yield_from(processor.assign_instr_index(process, 0))
    assert interface.send_events[0] == InterfaceEvent("bob", Message(content=12))


def test_recv_msg():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    meta = ProgramMeta.empty("alice")
    meta.csockets = {0: "bob"}
    program = create_program(instrs=[ReceiveCMsgOp("bob", "msg")], meta=meta)
    process = create_process(program, interface, inputs={"bob": 0})
    processor.initialize(process)

    yield_from(processor.assign_instr_index(process, 0))
    assert interface.recv_events[0] == InterfaceEvent("bob", MOCK_MESSAGE)
    assert process.prog_memory.host_mem.read("msg") == MOCK_MESSAGE.content


def test_recv_msg_with_latencies():
    ns.sim_reset()

    interface = MockHostInterface()
    latencies = HostLatencies(host_instr_time=500, host_peer_latency=1e6)
    processor = HostProcessor(interface, latencies)
    meta = ProgramMeta.empty("alice")
    meta.csockets = {0: "bob"}
    program = create_program(instrs=[ReceiveCMsgOp("bob", "msg")], meta=meta)
    process = create_process(program, interface, inputs={"bob": 0})
    processor.initialize(process)

    assert ns.sim_time() == 0
    netsquid_run(processor.assign_instr_index(process, 0))
    assert interface.recv_events[0] == InterfaceEvent("bob", MOCK_MESSAGE)
    assert process.prog_memory.host_mem.read("msg") == MOCK_MESSAGE.content
    assert ns.sim_time() == 1e6  # no host_instr_time used !


def test_add_cvalue():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program(
        instrs=[
            AssignCValueOp("a", 2),
            AssignCValueOp("b", 3),
            AddCValueOp("sum", "a", "b"),
        ]
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    assert process.prog_memory.host_mem.read("sum") == 5


def test_add_cvalue_with_latencies():
    ns.sim_reset()

    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies(host_instr_time=1200))
    program = create_program(
        instrs=[
            AssignCValueOp("a", 2),
            AssignCValueOp("b", 3),
            AddCValueOp("sum", "a", "b"),
        ]
    )
    process = create_process(program, interface)
    processor.initialize(process)

    assert ns.sim_time() == 0
    for i in range(len(program.instructions)):
        netsquid_run(processor.assign_instr_index(process, i))

    assert process.prog_memory.host_mem.read("sum") == 5
    assert ns.sim_time() == len(program.instructions) * 1200


def test_multiply_const():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program(
        instrs=[AssignCValueOp("a", 4), MultiplyConstantCValueOp("result", "a", -1)]
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    assert process.prog_memory.host_mem.read("result") == -4


def test_bit_cond_mult():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program(
        instrs=[
            AssignCValueOp("var1", 4),
            AssignCValueOp("var2", 7),
            AssignCValueOp("cond1", 0),
            AssignCValueOp("cond2", 1),
            BitConditionalMultiplyConstantCValueOp("result1", "var1", "cond1", -1),
            BitConditionalMultiplyConstantCValueOp("result2", "var2", "cond2", -1),
        ]
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    assert process.prog_memory.host_mem.read("result1") == 4
    assert process.prog_memory.host_mem.read("result2") == -7


def test_run_subroutine():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero(), asynchronous=True)

    subrt = Subroutine()
    metadata = RoutineMetadata.use_none()
    routine = LocalRoutine("subrt1", subrt, return_vars=[], metadata=metadata)

    program = create_program(
        instrs=[RunSubroutineOp(None, IqoalaVector([]), "subrt1")],
        subroutines={"subrt1": routine},
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    # Host processor should have communicated with the qnos processor.
    send_event = interface.send_events[0]
    assert send_event.peer == "qnos"
    assert isinstance(send_event.msg, Message)
    assert isinstance(send_event.msg.content, LrCallTuple)
    assert send_event.msg.content.routine_name == "subrt1"
    assert interface.recv_events[0] == InterfaceEvent("qnos", MOCK_MESSAGE)


def test_run_subroutine_2():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero(), asynchronous=True)

    subrt_text = """
    set R0 {my_value}
    ret_reg R0
    """
    subrt = parse_text_subroutine(subrt_text)
    routine = LocalRoutine(
        "subrt1",
        subrt,
        return_vars=["m"],
        metadata=RoutineMetadata.use_none(),
    )

    program = create_program(
        instrs=[
            AssignCValueOp("my_value", 16),
            RunSubroutineOp(IqoalaVector(["m"]), IqoalaVector(["my_value"]), "subrt1"),
        ],
        subroutines={"subrt1": routine},
    )
    process = create_process(program, interface)

    # Make sure interface can mimick writing subroutine results to shared memory
    interface.shared_mem = process.prog_memory.shared_memmgr

    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    send_event = interface.send_events[0]
    assert send_event.peer == "qnos"
    assert isinstance(send_event.msg, Message)
    assert isinstance(send_event.msg.content, LrCallTuple)
    assert send_event.msg.content.routine_name == "subrt1"
    assert interface.recv_events[0] == InterfaceEvent("qnos", MOCK_MESSAGE)

    # Hack to find out which addr was allocated to write results to.
    result_addr = process.prog_memory.shared_memmgr._lr_out_addrs[0]
    assert process.prog_memory.shared_memmgr.read_lr_out(result_addr, 1) == [
        MOCK_QNOS_RET_VALUE
    ]

    # Check if result was correctly written to host variable "m".
    assert process.prog_memory.host_mem.read("m") == MOCK_QNOS_RET_VALUE


def create_simple_request(
    remote_id: int,
    num_pairs: int,
    virt_ids: RequestVirtIdMapping,
    typ: EprType,
    role: EprRole,
) -> IqoalaRequest:
    return IqoalaRequest(
        name="req",
        remote_id=remote_id,
        epr_socket_id=0,
        num_pairs=num_pairs,
        virt_ids=virt_ids,
        timeout=1000,
        fidelity=0.65,
        typ=typ,
        role=role,
        result_array_addr=3,
    )


def test_run_request():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero(), asynchronous=True)

    request = create_simple_request(
        remote_id=2,
        num_pairs=5,
        virt_ids=RequestVirtIdMapping.from_str("increment 0"),
        typ=EprType.CREATE_KEEP,
        role=EprRole.CREATE,
    )
    routine = RequestRoutine("req", request, [], CallbackType.WAIT_ALL, None)

    program = create_program(
        instrs=[RunRequestOp(None, IqoalaVector([]), "req")],
        requests={"req": routine},
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    # Host processor should have communicated with the netstack processor.
    send_event = interface.send_events[0]
    assert send_event.peer == "netstack"
    assert isinstance(send_event.msg, Message)
    assert isinstance(send_event.msg.content, RrCallTuple)
    assert send_event.msg.content.routine_name == "req"
    assert interface.recv_events[0] == InterfaceEvent("netstack", MOCK_MESSAGE)


def test_return_result():
    interface = MockHostInterface()
    processor = HostProcessor(interface, HostLatencies.all_zero())
    program = create_program(
        instrs=[AssignCValueOp("result", 2), ReturnResultOp("result")]
    )
    process = create_process(program, interface)
    processor.initialize(process)

    for i in range(len(program.instructions)):
        yield_from(processor.assign_instr_index(process, i))

    assert process.prog_memory.host_mem.read("result") == 2
    assert process.result.values == {"result": 2}


if __name__ == "__main__":
    test_initialize()
    test_assign_cvalue()
    test_assign_cvalue_with_latencies()
    test_send_msg()
    test_recv_msg()
    test_recv_msg_with_latencies()
    test_add_cvalue()
    test_add_cvalue_with_latencies()
    test_multiply_const()
    test_bit_cond_mult()
    test_run_subroutine()
    test_run_subroutine_2()
    test_run_request()
    test_return_result()
