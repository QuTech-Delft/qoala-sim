from __future__ import annotations

from typing import Dict, Generator, Tuple

import netsquid as ns
from netqasm.sdk.build_epr import SER_RESPONSE_KEEP_IDX_GOODNESS, SER_RESPONSE_KEEP_LEN
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.nodes import Node
from netsquid.protocols import Protocol
from netsquid.qubits import ketstates
from netsquid_magic.link_layer import (
    MagicLinkLayerProtocolWithSignaling,
    SingleClickTranslationUnit,
)
from netsquid_magic.magic_distributor import PerfectStateMagicDistributor

from pydynaa import EventExpression
from qoala.lang.ehi import UnitModule
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.request import EprType
from qoala.runtime.environment import (
    GlobalEnvironment,
    GlobalNodeInfo,
    LocalEnvironment,
)
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import (
    GenericToVanillaInterface,
    LhiConverter,
    NvToNvInterface,
)
from qoala.runtime.memory import ProgramMemory, SharedMemory
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.egp import EgpProtocol
from qoala.sim.egpmgr import EgpManager
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import (
    NetstackComponent,
    NetstackInterface,
    NetstackLatencies,
    NetstackProcessor,
)
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.sim.requests import NetstackCreateRequest, NetstackReceiveRequest
from qoala.util.tests import has_multi_state


def perfect_uniform_qdevice(node_name: str, num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[
            INSTR_INIT,
            INSTR_X,
            INSTR_Y,
            INSTR_Z,
            INSTR_H,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
        single_duration=5e3,
        two_instructions=[INSTR_CNOT],
        two_duration=100e3,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name=node_name, qmemory=processor)
    return QDevice(node=node, topology=topology)


def perfect_nv_star_qdevice(node_name: str, num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.perfect_star(
        num_qubits=num_qubits,
        comm_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_MEASURE,
        ],
        comm_duration=5e3,
        mem_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
        ],
        mem_duration=1e4,
        two_instructions=[INSTR_CXDIR, INSTR_CYDIR],
        two_duration=1e5,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name=node_name, qmemory=processor)
    return QDevice(node=node, topology=topology)


def create_process(pid: int, unit_module: UnitModule) -> IqoalaProcess:
    program = IqoalaProgram(
        blocks=[], local_routines={}, meta=ProgramMeta.empty("prog")
    )
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


def setup_components(
    alice_qdevice: QDevice,
    bob_qdevice: QDevice,
    latencies: NetstackLatencies = NetstackLatencies.all_zero(),
) -> Tuple[NetstackProcessor, NetstackProcessor]:
    alice_node = alice_qdevice._node
    bob_node = bob_qdevice._node

    env = GlobalEnvironment()
    env.add_node(alice_node.ID, GlobalNodeInfo(alice_node.name, alice_node.ID))
    env.add_node(bob_node.ID, GlobalNodeInfo(bob_node.name, bob_node.ID))

    alice_comp = NetstackComponent(node=alice_node, global_env=env)
    bob_comp = NetstackComponent(node=bob_node, global_env=env)

    alice_comp.peer_out_port("bob").connect(bob_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("bob").connect(bob_comp.peer_out_port("alice"))

    alice_memmgr = MemoryManager(alice_node.name, alice_qdevice)
    bob_memmgr = MemoryManager(bob_node.name, bob_qdevice)

    alice_egpmgr = EgpManager()
    bob_egpmgr = EgpManager()

    alice_intf = NetstackInterface(
        alice_comp,
        LocalEnvironment(env, alice_node.ID),
        alice_qdevice,
        alice_memmgr,
        alice_egpmgr,
    )
    bob_intf = NetstackInterface(
        bob_comp,
        LocalEnvironment(env, bob_node.ID),
        bob_qdevice,
        bob_memmgr,
        bob_egpmgr,
    )

    alice_processor = NetstackProcessor(alice_intf, latencies)
    bob_processor = NetstackProcessor(bob_intf, latencies)

    return (alice_processor, bob_processor)


def setup_components_generic(
    num_qubits: int, latencies: NetstackLatencies = NetstackLatencies.all_zero()
) -> Tuple[NetstackProcessor, NetstackProcessor]:
    alice_qdevice = perfect_uniform_qdevice("alice", num_qubits)
    bob_qdevice = perfect_uniform_qdevice("bob", num_qubits)

    return setup_components(alice_qdevice, bob_qdevice, latencies)


def setup_components_nv(
    num_qubits: int, latencies: NetstackLatencies = NetstackLatencies.all_zero()
) -> Tuple[NetstackProcessor, NetstackProcessor]:
    alice_qdevice = perfect_nv_star_qdevice("alice", num_qubits)
    bob_qdevice = perfect_nv_star_qdevice("bob", num_qubits)

    return setup_components(alice_qdevice, bob_qdevice, latencies)


def create_egp_protocols(
    node1: Node, node2: Node, duration: float = 1000
) -> Tuple[EgpProtocol, EgpProtocol]:
    link_dist = PerfectStateMagicDistributor(nodes=[node1, node2], state_delay=duration)
    link_prot = MagicLinkLayerProtocolWithSignaling(
        nodes=[node1, node2],
        magic_distributor=link_dist,
        translation_unit=SingleClickTranslationUnit(),
    )
    return EgpProtocol(node1, link_prot), EgpProtocol(node2, link_prot)


def create_netstack_create_request(remote_id: int) -> NetstackCreateRequest:
    return NetstackCreateRequest(
        remote_id=remote_id,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=1,
        fidelity=0.75,
        virt_qubit_ids=[0],
        result_array_addr=0,
    )


def create_netstack_receive_request(remote_id: int) -> NetstackReceiveRequest:
    return NetstackReceiveRequest(
        remote_id=remote_id,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=1,
        fidelity=0.75,
        virt_qubit_ids=[0],
        result_array_addr=0,
    )


def test_single_pair():
    ns.sim_reset()

    alice_processor, bob_processor = setup_components_nv(num_qubits=2)

    alice_topology = alice_processor.qdevice.topology
    bob_topology = alice_processor.qdevice.topology
    assert alice_topology == bob_topology
    ehi = LhiConverter.to_ehi(alice_topology, ntf=NvToNvInterface())
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_node = alice_processor._interface._comp.node
    bob_node = bob_processor._interface._comp.node

    num_pairs = 3
    fidelity = 0.75
    result_array_addr = 0

    alice_request = NetstackCreateRequest(
        remote_id=bob_node.ID,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=num_pairs,
        fidelity=fidelity,
        virt_qubit_ids=[1],
        result_array_addr=result_array_addr,
    )
    bob_request = NetstackReceiveRequest(
        remote_id=alice_node.ID,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=num_pairs,
        fidelity=fidelity,
        virt_qubit_ids=[1],
        result_array_addr=result_array_addr,
    )

    alice_egpmgr = alice_processor._interface.egpmgr
    bob_egpmgr = bob_processor._interface.egpmgr

    alice_egp, bob_egp = create_egp_protocols(alice_node, bob_node)
    alice_egpmgr.add_egp(bob_node.ID, alice_egp)
    bob_egpmgr.add_egp(alice_node.ID, bob_egp)

    class NetstackProcessorProtocol(Protocol):
        def __init__(self, name: str, processor: NetstackProcessor) -> None:
            super().__init__(name)
            self._processor = processor
            self._memmgr = processor._interface.memmgr
            self._pid = 0
            self._process = create_process(self._pid, unit_module)
            self._memmgr.add_process(self._process)

        @property
        def pid(self) -> int:
            return self._pid

        @property
        def memmgr(self) -> MemoryManager:
            return self._memmgr

    class AliceProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self._processor._interface.receive_peer_msg("bob")
            yield from self._processor.create_single_pair(
                self._process, alice_request, virt_id=0
            )

    class BobProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            self._processor._interface.send_peer_msg("alice", Message("ready"))
            yield from self._processor.receive_single_pair(
                self._process, bob_request, virt_id=0
            )

    alice = AliceProtocol("alice", alice_processor)
    alice_processor._interface.start()  # also starts peer listeners
    alice.start()
    alice_egp.start()

    bob = BobProtocol("bob", bob_processor)
    bob_processor._interface.start()  # also starts peer listeners
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    ns.sim_run()

    assert alice.memmgr.phys_id_for(alice.pid, 0) == 0
    assert bob.memmgr.phys_id_for(bob.pid, 0) == 0


def test_handle_ck_request():
    ns.sim_reset()

    alice_processor, bob_processor = setup_components_nv(num_qubits=2)

    alice_topology = alice_processor.qdevice.topology
    bob_topology = alice_processor.qdevice.topology
    assert alice_topology == bob_topology
    ehi = LhiConverter.to_ehi(alice_topology, ntf=NvToNvInterface())
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_node = alice_processor._interface._comp.node
    bob_node = bob_processor._interface._comp.node

    num_pairs = 1
    fidelity = 0.75
    result_array_addr = 0

    alice_request = NetstackCreateRequest(
        remote_id=bob_node.ID,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=num_pairs,
        fidelity=fidelity,
        virt_qubit_ids=[0],
        result_array_addr=result_array_addr,
    )
    bob_request = NetstackReceiveRequest(
        remote_id=alice_node.ID,
        epr_socket_id=0,
        typ=EprType.CREATE_KEEP,
        num_pairs=num_pairs,
        fidelity=fidelity,
        virt_qubit_ids=[0],
        result_array_addr=result_array_addr,
    )

    alice_egpmgr = alice_processor._interface.egpmgr
    bob_egpmgr = bob_processor._interface.egpmgr

    alice_egp, bob_egp = create_egp_protocols(alice_node, bob_node)
    alice_egpmgr.add_egp(bob_node.ID, alice_egp)
    bob_egpmgr.add_egp(alice_node.ID, bob_egp)

    class NetstackProcessorProtocol(Protocol):
        def __init__(self, name: str, processor: NetstackProcessor) -> None:
            super().__init__(name)
            self._processor = processor
            self._memmgr = processor._interface.memmgr
            self._pid = 0
            self._process = create_process(self._pid, unit_module)
            self._memmgr.add_process(self._process)
            self._process.prog_memory.shared_mem.init_new_array(result_array_addr, 10)

        @property
        def pid(self) -> int:
            return self._pid

        @property
        def process(self) -> IqoalaProcess:
            return self._process

        @property
        def memmgr(self) -> MemoryManager:
            return self._memmgr

        @property
        def shared_mem(self) -> SharedMemory:
            return self.process.prog_memory.shared_mem

    class AliceProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self._processor._interface.receive_peer_msg("bob")
            yield from self._processor.handle_create_ck_request(
                self._process, alice_request
            )

    class BobProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            self._processor._interface.send_peer_msg("alice", Message("ready"))
            yield from self._processor.handle_receive_ck_request(
                self._process, bob_request
            )

    alice = AliceProtocol("alice", alice_processor)
    alice_processor._interface.start()  # also starts peer listeners
    alice.start()
    alice_egp.start()

    bob = BobProtocol("bob", bob_processor)
    bob_processor._interface.start()  # also starts peer listeners
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    assert (
        alice.shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is None
    )

    ns.sim_run()

    assert alice.memmgr.phys_id_for(alice.pid, 0) == 0
    assert bob.memmgr.phys_id_for(bob.pid, 0) == 0

    assert (
        alice.shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is not None
    )

    alice_qubit = alice_processor.qdevice.get_local_qubit(0)
    bob_qubit = bob_processor.qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)


def test_two_requests():
    ns.sim_reset()

    alice_processor, bob_processor = setup_components_generic(num_qubits=3)

    alice_topology = alice_processor.qdevice.topology
    bob_topology = alice_processor.qdevice.topology
    assert alice_topology == bob_topology
    ehi = LhiConverter.to_ehi(alice_topology, ntf=GenericToVanillaInterface())
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_node = alice_processor._interface._comp.node
    bob_node = bob_processor._interface._comp.node

    result_array_addr = 0

    alice_requests = {
        0: create_netstack_create_request(remote_id=bob_node.ID),
        1: create_netstack_create_request(remote_id=bob_node.ID),
    }
    bob_requests = {
        0: create_netstack_receive_request(remote_id=alice_node.ID),
        1: create_netstack_receive_request(remote_id=alice_node.ID),
    }

    alice_memmgr = alice_processor._interface.memmgr
    bob_memmgr = bob_processor._interface.memmgr

    alice_egpmgr = alice_processor._interface.egpmgr
    bob_egpmgr = bob_processor._interface.egpmgr

    alice_egp, bob_egp = create_egp_protocols(alice_node, bob_node)
    alice_egpmgr.add_egp(bob_node.ID, alice_egp)
    bob_egpmgr.add_egp(alice_node.ID, bob_egp)

    class NetstackProcessorProtocol(Protocol):
        def __init__(
            self,
            name: str,
            processor: NetstackProcessor,
            processes: Dict[int, IqoalaProcess],
        ) -> None:
            super().__init__(name)
            self._processor = processor
            self._memmgr = processor._interface.memmgr
            self._processes = processes

        @property
        def processes(self) -> Dict[int, IqoalaProcess]:
            return self._processes

        @property
        def memmgr(self) -> MemoryManager:
            return self._memmgr

    class AliceProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                yield from self._processor._interface.receive_peer_msg("bob")
                yield from self._processor.handle_create_ck_request(
                    process, alice_requests[pid]
                )

    class BobProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                self._processor._interface.send_peer_msg("alice", Message("ready"))
                yield from self._processor.handle_receive_ck_request(
                    process, bob_requests[pid]
                )

    alice_process0 = create_process(pid=0, unit_module=unit_module)
    alice_process1 = create_process(pid=1, unit_module=unit_module)
    alice_process0.shared_mem.init_new_array(0, 10)
    alice_process1.shared_mem.init_new_array(0, 10)
    alice_memmgr.add_process(alice_process0)
    alice_memmgr.add_process(alice_process1)
    alice = AliceProtocol(
        "alice", alice_processor, {0: alice_process0, 1: alice_process1}
    )
    alice_processor._interface.start()  # also starts peer listeners
    alice.start()
    alice_egp.start()

    bob_process0 = create_process(pid=0, unit_module=unit_module)
    bob_process1 = create_process(pid=1, unit_module=unit_module)
    bob_process0.shared_mem.init_new_array(0, 10)
    bob_process1.shared_mem.init_new_array(0, 10)
    bob_memmgr.add_process(bob_process0)
    bob_memmgr.add_process(bob_process1)
    bob = BobProtocol("bob", bob_processor, {0: bob_process0, 1: bob_process1})
    bob_processor._interface.start()  # also starts peer listeners
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    assert (
        alice.processes[0].shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is None
    )

    ns.sim_run()

    assert alice.memmgr.phys_id_for(alice.processes[0].pid, 0) == 0
    assert alice.memmgr.phys_id_for(alice.processes[1].pid, 0) == 1
    assert bob.memmgr.phys_id_for(bob.processes[0].pid, 0) == 0
    assert bob.memmgr.phys_id_for(bob.processes[1].pid, 0) == 1

    assert (
        alice.processes[0].shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is not None
    )

    alice_qubit = alice_processor.qdevice.get_local_qubit(0)
    bob_qubit = bob_processor.qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)

    alice_qubit = alice_processor.qdevice.get_local_qubit(1)
    bob_qubit = bob_processor.qdevice.get_local_qubit(1)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)

    alice_qubit = alice_processor.qdevice.get_local_qubit(2)
    assert alice_qubit is None
    bob_qubit = bob_processor.qdevice.get_local_qubit(2)
    assert bob_qubit is None


def test_handle_request():
    ns.sim_reset()

    alice_processor, bob_processor = setup_components_generic(num_qubits=3)

    alice_topology = alice_processor.qdevice.topology
    bob_topology = alice_processor.qdevice.topology
    assert alice_topology == bob_topology
    ehi = LhiConverter.to_ehi(alice_topology, ntf=GenericToVanillaInterface())
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_node = alice_processor._interface._comp.node
    bob_node = bob_processor._interface._comp.node

    result_array_addr = 0

    alice_requests = {
        0: create_netstack_create_request(remote_id=bob_node.ID),
        1: create_netstack_create_request(remote_id=bob_node.ID),
    }
    bob_requests = {
        0: create_netstack_receive_request(remote_id=alice_node.ID),
        1: create_netstack_receive_request(remote_id=alice_node.ID),
    }

    alice_memmgr = alice_processor._interface.memmgr
    bob_memmgr = bob_processor._interface.memmgr

    alice_egpmgr = alice_processor._interface.egpmgr
    bob_egpmgr = bob_processor._interface.egpmgr

    alice_egp, bob_egp = create_egp_protocols(alice_node, bob_node)
    alice_egpmgr.add_egp(bob_node.ID, alice_egp)
    bob_egpmgr.add_egp(alice_node.ID, bob_egp)

    class NetstackProcessorProtocol(Protocol):
        def __init__(
            self,
            name: str,
            processor: NetstackProcessor,
            processes: Dict[int, IqoalaProcess],
        ) -> None:
            super().__init__(name)
            self._processor = processor
            self._memmgr = processor._interface.memmgr
            self._processes = processes

        @property
        def processes(self) -> Dict[int, IqoalaProcess]:
            return self._processes

        @property
        def memmgr(self) -> MemoryManager:
            return self._memmgr

    class AliceProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                yield from self._processor.handle_create_request(
                    process, alice_requests[pid]
                )

    class BobProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                yield from self._processor.handle_receive_request(
                    process, bob_requests[pid]
                )

    alice_process0 = create_process(pid=0, unit_module=unit_module)
    alice_process1 = create_process(pid=1, unit_module=unit_module)
    alice_process0.shared_mem.init_new_array(0, 10)
    alice_process1.shared_mem.init_new_array(0, 10)
    alice_memmgr.add_process(alice_process0)
    alice_memmgr.add_process(alice_process1)
    alice = AliceProtocol(
        "alice", alice_processor, {0: alice_process0, 1: alice_process1}
    )
    alice_processor._interface.start()  # also starts peer listeners
    alice.start()
    alice_egp.start()

    bob_process0 = create_process(pid=0, unit_module=unit_module)
    bob_process1 = create_process(pid=1, unit_module=unit_module)
    bob_process0.shared_mem.init_new_array(0, 10)
    bob_process1.shared_mem.init_new_array(0, 10)
    bob_memmgr.add_process(bob_process0)
    bob_memmgr.add_process(bob_process1)
    bob = BobProtocol("bob", bob_processor, {0: bob_process0, 1: bob_process1})
    bob_processor._interface.start()  # also starts peer listeners
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    assert (
        alice.processes[0].shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is None
    )

    ns.sim_run()

    assert alice.memmgr.phys_id_for(alice.processes[0].pid, 0) == 0
    assert alice.memmgr.phys_id_for(alice.processes[1].pid, 0) == 1
    assert bob.memmgr.phys_id_for(bob.processes[0].pid, 0) == 0
    assert bob.memmgr.phys_id_for(bob.processes[1].pid, 0) == 1

    assert (
        alice.processes[0].shared_mem.get_array_part(
            result_array_addr,
            SER_RESPONSE_KEEP_LEN * 0 + SER_RESPONSE_KEEP_IDX_GOODNESS,
        )
        is not None
    )

    alice_qubit = alice_processor.qdevice.get_local_qubit(0)
    bob_qubit = bob_processor.qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)

    alice_qubit = alice_processor.qdevice.get_local_qubit(1)
    bob_qubit = bob_processor.qdevice.get_local_qubit(1)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)

    alice_qubit = alice_processor.qdevice.get_local_qubit(2)
    assert alice_qubit is None
    bob_qubit = bob_processor.qdevice.get_local_qubit(2)
    assert bob_qubit is None


def test_4_with_latencies():
    ns.sim_reset()

    ns.sim_reset()
    netstack_peer_latency = 200e3
    epr_creation_duration = 100e3

    alice_processor, bob_processor = setup_components_generic(
        num_qubits=3,
        latencies=NetstackLatencies(netstack_peer_latency=netstack_peer_latency),
    )

    alice_topology = alice_processor.qdevice.topology
    bob_topology = alice_processor.qdevice.topology
    assert alice_topology == bob_topology
    ehi = LhiConverter.to_ehi(alice_topology, ntf=GenericToVanillaInterface())
    unit_module = UnitModule.from_full_ehi(ehi)

    alice_node = alice_processor._interface._comp.node
    bob_node = bob_processor._interface._comp.node

    alice_requests = {
        0: create_netstack_create_request(remote_id=bob_node.ID),
    }
    bob_requests = {
        0: create_netstack_receive_request(remote_id=alice_node.ID),
    }

    alice_memmgr = alice_processor._interface.memmgr
    bob_memmgr = bob_processor._interface.memmgr

    alice_egpmgr = alice_processor._interface.egpmgr
    bob_egpmgr = bob_processor._interface.egpmgr

    alice_egp, bob_egp = create_egp_protocols(
        alice_node, bob_node, epr_creation_duration
    )
    alice_egpmgr.add_egp(bob_node.ID, alice_egp)
    bob_egpmgr.add_egp(alice_node.ID, bob_egp)

    class NetstackProcessorProtocol(Protocol):
        def __init__(
            self,
            name: str,
            processor: NetstackProcessor,
            processes: Dict[int, IqoalaProcess],
        ) -> None:
            super().__init__(name)
            self._processor = processor
            self._memmgr = processor._interface.memmgr
            self._processes = processes

        @property
        def processes(self) -> Dict[int, IqoalaProcess]:
            return self._processes

        @property
        def memmgr(self) -> MemoryManager:
            return self._memmgr

    class AliceProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                yield from self._processor.handle_create_request(
                    process, alice_requests[pid]
                )

    class BobProtocol(NetstackProcessorProtocol):
        def run(self) -> Generator[EventExpression, None, None]:
            for pid, process in self.processes.items():
                yield from self._processor.handle_receive_request(
                    process, bob_requests[pid]
                )

    alice_process0 = create_process(pid=0, unit_module=unit_module)
    alice_process0.shared_mem.init_new_array(0, 10)
    alice_memmgr.add_process(alice_process0)
    alice = AliceProtocol("alice", alice_processor, {0: alice_process0})
    alice_processor._interface.start()  # also starts peer listeners
    alice.start()
    alice_egp.start()

    bob_process0 = create_process(pid=0, unit_module=unit_module)
    bob_process0.shared_mem.init_new_array(0, 10)
    bob_memmgr.add_process(bob_process0)
    bob = BobProtocol("bob", bob_processor, {0: bob_process0})
    bob_processor._interface.start()  # also starts peer listeners
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    assert ns.sim_time() == 0
    ns.sim_run()
    assert ns.sim_time() == 2 * netstack_peer_latency + 1 * epr_creation_duration


if __name__ == "__main__":
    test_single_pair()
    test_handle_ck_request()
    test_two_requests()
    test_handle_request()
    test_4_with_latencies()
