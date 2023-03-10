from __future__ import annotations

from typing import Dict, Generator, Tuple

import netsquid as ns
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
from qlink_interface import (
    ReqCreateAndKeep,
    ReqCreateBase,
    ReqMeasureDirectly,
    ReqReceive,
    ReqRemoteStatePrep,
    ResCreateAndKeep,
    ResMeasureDirectly,
)
from qlink_interface.interface import (
    ReqCreateBase,
    ResCreateAndKeep,
    ResMeasureDirectly,
)

from pydynaa import EventExpression
from qoala.lang.request import EprType
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.message import Message
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.egp import EgpProtocol
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import NetstackProcessor
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


def setup_components_generic(num_qubits: int) -> Tuple[QDevice, QDevice]:
    alice_qdevice = perfect_uniform_qdevice("alice", num_qubits)
    bob_qdevice = perfect_uniform_qdevice("bob", num_qubits)

    return alice_qdevice, bob_qdevice


def setup_components_nv(num_qubits: int) -> Tuple[QDevice, QDevice]:
    alice_qdevice = perfect_nv_star_qdevice("alice", num_qubits)
    bob_qdevice = perfect_nv_star_qdevice("bob", num_qubits)

    return alice_qdevice, bob_qdevice


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

    alice_qdevice, bob_qdevice = setup_components_nv(num_qubits=2)
    alice_node = alice_qdevice._node
    bob_node = bob_qdevice._node

    num_pairs = 3
    fidelity = 0.75

    alice_request = ReqCreateAndKeep(
        remote_node_id=bob_node.ID, number=num_pairs, minimum_fidelity=fidelity
    )
    bob_request = ReqReceive(remote_node_id=alice_node.ID)

    alice_egp, bob_egp = create_egp_protocols(alice_node, bob_node)

    class EgpUserProtocol(Protocol):
        def __init__(self, name: str, egp: EgpProtocol, request: ReqCreateBase) -> None:
            super().__init__(name)
            self._egp = egp
            self._request = request

        def run(self) -> Generator[EventExpression, None, None]:
            self._egp.put(self._request)
            yield self.await_signal(
                sender=self._egp,
                signal_label=ResCreateAndKeep.__name__,
            )
            result: ResCreateAndKeep = self._egp.get_signal_result(
                ResCreateAndKeep.__name__, receiver=self
            )

    alice = EgpUserProtocol("alice", alice_egp, alice_request)
    alice.start()
    alice_egp.start()

    bob = EgpUserProtocol("bob", bob_egp, bob_request)
    bob.start()
    bob_egp.start()

    link_prot = alice_egp._ll_prot  # same as bob_egp._ll_prot
    link_prot.start()

    ns.sim_run()

    alice_qubit = alice_qdevice.get_local_qubit(0)
    bob_qubit = bob_qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], ketstates.b00)


if __name__ == "__main__":
    test_single_pair()
