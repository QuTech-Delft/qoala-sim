from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
import pytest
from netqasm.lang.parsing import parse_text_subroutine

from pydynaa import EventExpression
from qoala.lang.ehi import UnitModule
from qoala.lang.program import IqoalaProgram, LocalRoutine, ProgramMeta
from qoala.lang.routine import RoutineMetadata
from qoala.runtime.lhi import LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NvToNvInterface
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.memmgr import AllocError, MemoryManager
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
    pid: int, program: IqoalaProgram, unit_module: UnitModule
) -> IqoalaProcess:
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=ProgramInput({}),
        tasks=ProgramTaskList.empty(program),
    )
    mem = ProgramMemory(pid=pid, unit_module=unit_module)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
        active_routines={},
    )
    return process


def create_process_with_subrt(
    pid: int, subrt_text: str, unit_module: UnitModule
) -> IqoalaProcess:
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata.use_none()
    iqoala_subrt = LocalRoutine("subrt", subrt, return_map={}, metadata=metadata)
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines={"subrt": iqoala_subrt}, meta=meta)
    return create_process(pid, program, unit_module)


def execute_process(processor: GenericProcessor, process: IqoalaProcess) -> int:
    subroutines = process.prog_instance.program.local_routines
    process.instantiate_routine("subrt", {})
    netqasm_instructions = subroutines["subrt"].subroutine.instructions

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
    subroutines = process.prog_instance.program.local_routines
    process.instantiate_routine("subrt", {})
    netqasm_instructions = subroutines["subrt"].subroutine.instructions

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
        subroutines = proc.prog_instance.program.local_routines
        proc.instantiate_routine("subrt", {})
        netqasm_instructions = subroutines["subrt"].subroutine.instructions
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
    assert process.prog_memory.shared_mem.get_reg_value("R0") == 17


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

    assert process.prog_memory.shared_mem.get_reg_value("R0") == 17


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
    assert process.prog_memory.shared_mem.get_reg_value("R2") == 7


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

    assert process.prog_memory.shared_mem.get_reg_value("R2") == 7


def test_alloc_qubit():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set Q0 0
    qalloc Q0
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    assert processor._interface.memmgr.phys_id_for(process.pid, 0) == 0
    assert processor._interface.memmgr.phys_id_for(process.pid, 1) is None


def test_free_qubit():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set Q0 0
    qalloc Q0
    qfree Q0
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    assert processor._interface.memmgr.phys_id_for(process.pid, 0) is None
    assert processor._interface.memmgr.phys_id_for(process.pid, 1) is None


def test_free_non_allocated():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set Q0 0
    qfree Q0
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)

    with pytest.raises(AllocError):
        execute_process(processor, process)


def test_alloc_multiple():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set Q0 0
    set Q1 1
    qalloc Q0
    qalloc Q1
    """
    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    assert processor._interface.memmgr.phys_id_for(process.pid, 0) == 0
    assert processor._interface.memmgr.phys_id_for(process.pid, 1) == 1


def test_alloc_multiprocess():
    processor, unit_module = setup_components(star_topology(2))

    subrt0 = """
    set Q0 0
    qalloc Q0
    """
    subrt1 = """
    set Q1 1
    qalloc Q1
    """
    process0 = create_process_with_subrt(0, subrt0, unit_module)
    process1 = create_process_with_subrt(1, subrt1, unit_module)
    processor._interface.memmgr.add_process(process0)
    processor._interface.memmgr.add_process(process1)
    execute_multiple_processes(processor, [process0, process1])

    assert processor._interface.memmgr.phys_id_for(process0.pid, 0) == 0
    assert processor._interface.memmgr.phys_id_for(process0.pid, 1) is None
    assert processor._interface.memmgr.phys_id_for(process1.pid, 0) is None
    assert processor._interface.memmgr.phys_id_for(process1.pid, 1) == 1

    assert processor._interface.memmgr._physical_mapping[0].pid == process0.pid
    assert processor._interface.memmgr._physical_mapping[1].pid == process1.pid


def test_alloc_multiprocess_same_virt_id():
    processor, unit_module = setup_components(uniform_topology(2))

    subrt0 = """
    set Q0 0
    qalloc Q0
    """
    subrt1 = """
    set Q0 0
    qalloc Q0
    """

    process0 = create_process_with_subrt(0, subrt0, unit_module)
    process1 = create_process_with_subrt(1, subrt1, unit_module)
    processor._interface.memmgr.add_process(process0)
    processor._interface.memmgr.add_process(process1)
    execute_multiple_processes(processor, [process0, process1])

    assert processor._interface.memmgr.phys_id_for(process0.pid, 0) == 0
    assert processor._interface.memmgr.phys_id_for(process0.pid, 1) is None
    assert processor._interface.memmgr.phys_id_for(process1.pid, 0) == 1
    assert processor._interface.memmgr.phys_id_for(process1.pid, 1) is None

    assert processor._interface.memmgr._physical_mapping[0].pid == process0.pid
    assert processor._interface.memmgr._physical_mapping[1].pid == process1.pid


def test_alloc_multiprocess_same_virt_id_trait_not_available():
    processor, unit_module = setup_components(star_topology(2))

    subrt0 = """
    set Q0 0
    qalloc Q0
    """
    subrt1 = """
    set Q0 0
    qalloc Q0
    """

    process0 = create_process_with_subrt(0, subrt0, unit_module)
    process1 = create_process_with_subrt(1, subrt1, unit_module)
    processor._interface.memmgr.add_process(process0)
    processor._interface.memmgr.add_process(process1)

    with pytest.raises(AllocError):
        execute_multiple_processes(processor, [process0, process1])


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
    assert process.prog_memory.shared_mem.get_reg_value("C0") == 4


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
    assert process.prog_memory.shared_mem.get_reg_value("C0") == 0


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

    assert process.prog_memory.shared_mem.get_reg_value("C0") == 0


def test_array():
    processor, unit_module = setup_components(star_topology(2))

    subrt = """
    set C10 10
    array C10 @0
    set R4 4
    set C8 8
    store C8 @0[R4]
    """
    assert native_instr_count(subrt) == 5

    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    instr_count = execute_process(processor, process)

    assert instr_count == 5
    array = process.prog_memory.shared_mem.get_array(0)
    assert len(array) == 10
    assert all(
        process.prog_memory.shared_mem.get_array_value(0, i) is None
        for i in range(10)
        if i != 4
    )
    assert process.prog_memory.shared_mem.get_array_value(0, 4) == 8


def test_wait_all():
    pid = 0
    array_id = 3
    start_idx = 5
    end_idx = 9

    # Let the mock interface write some result to the array such that
    # our "wait_all" instruction will unblock
    netstack_result = MockNetstackResultInfo(
        pid=pid, array_id=array_id, start_idx=start_idx, end_idx=end_idx
    )

    processor, unit_module = setup_components(
        uniform_topology(1), netstack_result=netstack_result
    )

    subrt = f"""
    array 10 @{array_id}
    wait_all @{array_id}[{start_idx}:{end_idx}]
    """
    process = create_process_with_subrt(pid, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    mem = process.prog_memory.shared_mem
    assert all(
        mem.get_array_value(array_id, i) is not None for i in range(start_idx, end_idx)
    )


if __name__ == "__main__":
    test_set_reg()
    test_set_reg_with_latencies()
    test_add()
    test_add_with_latencies()
    test_alloc_qubit()
    test_free_qubit()
    test_free_non_allocated()
    test_alloc_multiple()
    test_alloc_multiprocess()
    test_alloc_multiprocess_same_virt_id()
    test_alloc_multiprocess_same_virt_id_trait_not_available()
    test_no_branch()
    test_branch()
    test_branch_with_latencies()
    test_array()
    test_wait_all()
