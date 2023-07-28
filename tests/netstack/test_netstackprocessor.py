from typing import Generator, List, Optional

import pytest

from pydynaa import EventExpression
from qoala.lang.ehi import UnitModule
from qoala.lang.program import ProgramMeta, QoalaProgram
from qoala.lang.request import EprRole, EprType, QoalaRequest, RequestVirtIdMapping
from qoala.runtime.lhi import LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.ntf import GenericNtf
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.sim.entdist.entdist import EntDistRequest
from qoala.sim.eprsocket import EprSocket
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.netstack import NetstackInterface, NetstackLatencies, NetstackProcessor
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice, QDeviceCommand


class MockNetstackInterface(NetstackInterface):
    def __init__(
        self,
        qdevice: QDevice,
        memmgr: MemoryManager,
    ) -> None:
        self._qdevice = qdevice
        self._memmgr = memmgr

    def send_qnos_msg(self, msg: Message) -> None:
        pass

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


def create_process(pid: int, unit_module: UnitModule) -> QoalaProcess:
    program = QoalaProgram(blocks=[], local_routines={}, meta=ProgramMeta.empty("prog"))
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=ProgramInput({}),
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=pid)

    process = QoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def create_simple_request(
    remote_id: int,
    num_pairs: int,
    virt_ids: RequestVirtIdMapping,
    epr_socket_id: int = 0,
) -> QoalaRequest:
    return QoalaRequest(
        name="req",
        remote_id=remote_id,
        epr_socket_id=epr_socket_id,
        num_pairs=num_pairs,
        virt_ids=virt_ids,
        timeout=1000,
        fidelity=0.65,
        typ=EprType.CREATE_KEEP,
        role=EprRole.CREATE,
    )


def test__allocate_for_pair():
    topology = generic_topology(5)
    qdevice = MockQDevice(topology)
    ehi = LhiConverter.to_ehi(topology, ntf=GenericNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    memmgr = MemoryManager("alice", qdevice)

    process = create_process(0, unit_module)
    memmgr.add_process(process)

    interface = MockNetstackInterface(qdevice, memmgr)
    latencies = NetstackLatencies.all_zero()
    processor = NetstackProcessor(interface, latencies)

    request = create_simple_request(
        remote_id=1, num_pairs=5, virt_ids=RequestVirtIdMapping.from_str("all 0")
    )

    assert memmgr.phys_id_for(process.pid, 0) is None
    assert processor._allocate_for_pair(process, request, 0) == 0
    assert memmgr.phys_id_for(process.pid, 0) == 0

    assert memmgr.phys_id_for(process.pid, 1) is None
    with pytest.raises(AllocError):
        # Index 1 also requires virt ID but it's already alloacted
        processor._allocate_for_pair(process, request, 1)

    request2 = create_simple_request(
        remote_id=1, num_pairs=2, virt_ids=RequestVirtIdMapping.from_str("increment 1")
    )
    assert processor._allocate_for_pair(process, request2, index=0) == 1
    assert memmgr.phys_id_for(process.pid, virt_id=1) == 1
    assert processor._allocate_for_pair(process, request2, index=1) == 2
    assert memmgr.phys_id_for(process.pid, virt_id=2) == 2


def test__create_entdist_request():
    topology = generic_topology(5)
    qdevice = MockQDevice(topology)
    ehi = LhiConverter.to_ehi(topology, ntf=GenericNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    memmgr = MemoryManager("alice", qdevice)

    process = create_process(0, unit_module)
    memmgr.add_process(process)

    interface = MockNetstackInterface(qdevice, memmgr)
    latencies = NetstackLatencies.all_zero()
    processor = NetstackProcessor(interface, latencies)

    request = create_simple_request(
        remote_id=1,
        num_pairs=5,
        virt_ids=RequestVirtIdMapping.from_str("all 0"),
        epr_socket_id=7,
    )

    phys_id = memmgr.allocate(process.pid, 3)
    process.epr_sockets[7] = EprSocket(7, 1, 0, 42, 1.0)
    assert processor._create_entdist_request(process, request, 3) == EntDistRequest(
        local_node_id=interface.node_id,
        remote_node_id=1,
        local_qubit_id=phys_id,
        local_pid=0,
        remote_pid=42,
    )


if __name__ == "__main__":
    test__allocate_for_pair()
    test__create_entdist_request()
