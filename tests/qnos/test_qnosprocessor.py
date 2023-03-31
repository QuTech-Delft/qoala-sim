from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
from netqasm.lang.parsing import parse_text_subroutine

from pydynaa import EventExpression
from qoala.lang.ehi import UnitModule
from qoala.lang.parse import LocalRoutineParser
from qoala.lang.program import IqoalaProgram, LocalRoutine, ProgramMeta
from qoala.lang.routine import RoutineMetadata
from qoala.runtime.lhi import LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NvToNvInterface
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.memmgr import MemoryManager
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos import GenericProcessor, QnosInterface, QnosLatencies, QnosProcessor
from qoala.util.tests import netsquid_run, yield_from

MOCK_MESSAGE = Message(content=42)
MOCK_QNOS_RET_REG = "R0"
MOCK_QNOS_RET_VALUE = 7


@dataclass(eq=True, frozen=True)
class InterfaceEvent:
    peer: str
    msg: Message


@dataclass(eq=True, frozen=True)
class FlushEvent:
    pass


@dataclass(eq=True, frozen=True)
class SignalEvent:
    pass


class MockQDevice(QDevice):
    def __init__(self, topology: LhiTopology) -> None:
        self._topology = topology

    def set_mem_pos_in_use(self, id: int, in_use: bool) -> None:
        pass


@dataclass
class MockNetstackResultInfo:
    pid: int
    array_id: int
    start_idx: int
    end_idx: int


class MockQnosInterface(QnosInterface):
    def __init__(
        self,
        qdevice: QDevice,
        netstack_result_info: Optional[MockNetstackResultInfo] = None,
    ) -> None:
        self.send_events: List[InterfaceEvent] = []
        self.recv_events: List[InterfaceEvent] = []
        self.flush_events: List[FlushEvent] = []
        self.signal_events: List[SignalEvent] = []

        self._qdevice = qdevice
        self._memmgr = MemoryManager("alice", self._qdevice)

        self.netstack_result_info: Optional[
            MockNetstackResultInfo
        ] = netstack_result_info

    def send_peer_msg(self, peer: str, msg: Message) -> None:
        self.send_events.append(InterfaceEvent(peer, msg))

    def receive_peer_msg(self, peer: str) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent(peer, MOCK_MESSAGE))
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    def send_host_msg(self, msg: Message) -> None:
        self.send_events.append(InterfaceEvent("host", msg))

    def receive_host_msg(self) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent("host", MOCK_MESSAGE))
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    def send_netstack_msg(self, msg: Message) -> None:
        self.send_events.append(InterfaceEvent("netstack", msg))

    def receive_netstack_msg(self) -> Generator[EventExpression, None, Message]:
        self.recv_events.append(InterfaceEvent("netstack", MOCK_MESSAGE))
        if self.netstack_result_info is not None:
            mem = self.memmgr._processes[
                self.netstack_result_info.pid
            ].prog_memory.shared_mem
            array_id = self.netstack_result_info.array_id
            start_idx = self.netstack_result_info.start_idx
            end_idx = self.netstack_result_info.end_idx
            for i in range(start_idx, end_idx):
                mem.set_array_value(array_id, i, 42)
        return MOCK_MESSAGE
        yield  # to make it behave as a generator

    def flush_netstack_msgs(self) -> None:
        self.flush_events.append(FlushEvent())

    def signal_memory_freed(self) -> None:
        self.signal_events.append(SignalEvent())

    @property
    def name(self) -> str:
        return "mock"


def create_program(
    subroutines: Optional[Dict[str, LocalRoutine]] = None,
    meta: Optional[ProgramMeta] = None,
) -> IqoalaProgram:
    if subroutines is None:
        subroutines = {}
    if meta is None:
        meta = ProgramMeta.empty("prog")
    return IqoalaProgram(blocks=[], local_routines=subroutines, meta=meta)


def create_process(
    pid: int,
    program: IqoalaProgram,
    unit_module: UnitModule,
    inputs: Optional[ProgramInput] = None,
) -> IqoalaProcess:
    if inputs is None:
        inputs = ProgramInput({})

    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=inputs,
        tasks=ProgramTaskList.empty(program),
        unit_module=unit_module,
        block_tasks=[],
    )
    mem = ProgramMemory(pid=pid)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def create_process_with_subrt(
    pid: int,
    subrt_text: str,
    unit_module: UnitModule,
    inputs: Optional[ProgramInput] = None,
) -> IqoalaProcess:
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata.use_none()
    iqoala_subrt = LocalRoutine("subrt", subrt, return_vars=[], metadata=metadata)
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines={"subrt": iqoala_subrt}, meta=meta)
    return create_process(pid, program, unit_module, inputs)


def create_process_with_local_routine(
    pid: int,
    routine_text: str,
    unit_module: UnitModule,
    inputs: Optional[ProgramInput] = None,
) -> IqoalaProcess:
    routines = LocalRoutineParser(routine_text).parse()
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines=routines, meta=meta)
    return create_process(pid, program, unit_module, inputs)


def execute_process(
    processor: GenericProcessor,
    process: IqoalaProcess,
    input_addr: Optional[MemAddr] = None,
    result_addr: Optional[MemAddr] = None,
) -> int:
    if input_addr is None:
        input_addr = MemAddr(0)
    if result_addr is None:
        result_addr = MemAddr(0)

    all_routines = process.program.local_routines
    routine = all_routines["subrt"]
    inputs = process.inputs.values
    processor.instantiate_routine(process, routine, inputs, input_addr, result_addr)

    netqasm_instructions = routine.subroutine.instructions

    instr_count = 0

    instr_idx = 0
    while instr_idx < len(netqasm_instructions):
        instr_count += 1
        instr_idx = yield_from(
            processor.assign_routine_instr(process, "subrt", instr_idx)
        )
    return instr_count


def execute_process_with_latencies(
    processor: GenericProcessor, process: IqoalaProcess
) -> int:
    all_routines = process.program.local_routines
    routine = all_routines["subrt"]
    # input/result arrays not used
    # TODO: add tests that do use these
    inputs = process.inputs.values
    processor.instantiate_routine(process, routine, inputs, MemAddr(0), MemAddr(0))

    netqasm_instructions = routine.subroutine.instructions

    instr_count = 0

    instr_idx = 0
    while instr_idx < len(netqasm_instructions):
        instr_count += 1
        instr_idx = netsquid_run(
            processor.assign_routine_instr(process, "subrt", instr_idx)
        )
    return instr_count


def execute_multiple_processes(
    processor: GenericProcessor, processes: List[IqoalaProcess]
) -> None:
    for proc in processes:
        all_routines = proc.program.local_routines
        routine = all_routines["subrt"]
        # input/result arrays not used
        # TODO: add tests that do use these
        inputs = proc.inputs.values
        processor.instantiate_routine(proc, routine, inputs, MemAddr(0), MemAddr(0))
        netqasm_instructions = routine.subroutine.instructions
        for i in range(len(netqasm_instructions)):
            yield_from(processor.assign_routine_instr(proc, "subrt", i))


def setup_components(
    topology: LhiTopology,
    latencies: QnosLatencies = QnosLatencies.all_zero(),
    netstack_result: Optional[MockNetstackResultInfo] = None,
    asynchronous: bool = False,
) -> Tuple[QnosProcessor, UnitModule]:
    qdevice = MockQDevice(topology)
    ehi = LhiConverter.to_ehi(topology, ntf=NvToNvInterface())
    unit_module = UnitModule.from_full_ehi(ehi)
    interface = MockQnosInterface(qdevice, netstack_result)
    processor = QnosProcessor(interface, latencies, asynchronous)
    return (processor, unit_module)


def uniform_topology(num_qubits: int) -> LhiTopology:
    return LhiTopologyBuilder.perfect_uniform(num_qubits, [], 0, [], 0)


def star_topology(num_qubits: int) -> LhiTopology:
    return LhiTopologyBuilder.perfect_star(num_qubits, [], 0, [], 0, [], 0)


def native_instr_count(subrt_text: str) -> int:
    # count the number of instructions in the subroutine when the subrt text
    # is parsed and compiled (which may lead to additional instructions)
    parsed_subrt = parse_text_subroutine(subrt_text)
    return len(parsed_subrt.instructions)


def test_set_reg():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 17
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)
    assert process.prog_memory.qnos_mem.get_reg_value("R0") == 17


def test_set_reg_with_latencies():
    ns.sim_reset()

    processor, unit_module = setup_components(
        star_topology(2), latencies=QnosLatencies(qnos_instr_time=5e3)
    )

    subrt = """
    set R0 17
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)

    assert ns.sim_time() == 0
    execute_process_with_latencies(processor, process)
    assert ns.sim_time() == 5e3

    assert process.prog_memory.qnos_mem.get_reg_value("R0") == 17


def test_add():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 2
    set R1 5
    add R2 R0 R1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)
    assert process.prog_memory.qnos_mem.get_reg_value("R2") == 7


def test_add_with_latencies():
    ns.sim_reset()

    processor, unit_module = setup_components(
        star_topology(2), latencies=QnosLatencies(qnos_instr_time=5e3)
    )

    subrt = """
    set R0 2
    set R1 5
    add R2 R0 R1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    assert native_instr_count(subrt) == 3

    assert ns.sim_time() == 0
    execute_process_with_latencies(processor, process)
    assert ns.sim_time() == 5e3 * 3

    assert process.prog_memory.qnos_mem.get_reg_value("R2") == 7


def test_mul():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 2
    set R1 5
    mul R2 R0 R1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)
    assert process.prog_memory.qnos_mem.get_reg_value("R2") == 10


def test_div():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 15
    set R1 3
    div R2 R0 R1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)
    assert process.prog_memory.qnos_mem.get_reg_value("R2") == 5


def test_div_rounded():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 16
    set R1 3
    div R2 R0 R1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)
    assert process.prog_memory.qnos_mem.get_reg_value("R2") == 5


def test_no_branch():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R3 3
    set R0 0
    beq R3 R0 LABEL1
    set R1 1
    add C0 R3 R1
LABEL1:
    """
    assert native_instr_count(subrt) == 5

    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    instr_count = execute_process(processor, process)

    assert instr_count == 5
    assert process.prog_memory.qnos_mem.get_reg_value("C0") == 4


def test_branch():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R3 3
    set C3 3
    beq R3 C3 LABEL1
    set R1 1
    add C0 R3 R1
LABEL1:
    """
    assert native_instr_count(subrt) == 5

    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    instr_count = execute_process(processor, process)

    assert instr_count == 3
    assert process.prog_memory.qnos_mem.get_reg_value("C0") == 0


def test_branch_with_latencies():
    ns.sim_reset()

    processor, unit_module = setup_components(
        star_topology(2), latencies=QnosLatencies(qnos_instr_time=5e3)
    )

    subrt = """
    set R3 3
    set C3 3
    beq R3 C3 LABEL1
    set R1 1
    add C0 R3 R1
LABEL1:
    """
    assert native_instr_count(subrt) == 5

    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)

    assert ns.sim_time() == 0
    instr_count = execute_process_with_latencies(processor, process)
    assert instr_count == 3
    assert ns.sim_time() == 5e3 * 3

    assert process.prog_memory.qnos_mem.get_reg_value("C0") == 0


def test_program_inputs():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set R0 {global_arg}
    """
    inputs = ProgramInput({"global_arg": 3})

    process = create_process_with_subrt(0, subrt, unit_module, inputs)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    assert process.qnos_mem.get_reg_value("R0") == 3


def test_program_routine_params():
    processor, unit_module = setup_components(star_topology(2))

    routine = """
SUBROUTINE subrt
    params: arg0
    returns: 
    uses: 
    keeps:
    request: 
  NETQASM_START
    load R0 @input[0]
  NETQASM_END
    """

    process = create_process_with_local_routine(0, routine, unit_module)
    processor._interface.memmgr.add_process(process)

    shared_mem = process.prog_memory.shared_memmgr
    input_addr = shared_mem.allocate_lr_in(1)
    shared_mem.write_lr_in(input_addr, [3])

    execute_process(processor, process, input_addr=input_addr, result_addr=0)

    assert process.qnos_mem.get_reg_value("R0") == 3


def test_program_routine_params_and_results():
    processor, unit_module = setup_components(star_topology(2))

    # TODO: fill in params and returns when Host/Array conversion is implemented
    routine = """
SUBROUTINE subrt
    params:
    returns: 
    uses: 
    keeps:
    request: 
  NETQASM_START
    load R0 @input[0]
    load R1 @input[1]
    add C0 R0 R0
    add C1 R1 R1
    store C0 @output[0]
    store C1 @output[1]
  NETQASM_END
    """

    process = create_process_with_local_routine(0, routine, unit_module)
    processor._interface.memmgr.add_process(process)

    shared_mem = process.prog_memory.shared_memmgr

    input_addr = shared_mem.allocate_lr_in(2)
    shared_mem.write_lr_in(input_addr, [3, 7])

    result_addr = shared_mem.allocate_lr_out(2)

    execute_process(processor, process, input_addr=input_addr, result_addr=result_addr)

    assert process.qnos_mem.get_reg_value("C0") == 6
    assert process.qnos_mem.get_reg_value("C1") == 14


if __name__ == "__main__":
    test_set_reg()
    test_set_reg_with_latencies()
    test_add()
    test_add_with_latencies()
    test_mul()
    test_div()
    test_div_rounded()
    test_no_branch()
    test_branch()
    test_branch_with_latencies()
    test_program_inputs()
    test_program_routine_params()
    test_program_routine_params_and_results()
