import itertools
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
import pytest
from netqasm.sdk.build_epr import (
    SER_RESPONSE_KEEP_IDX_BELL_STATE,
    SER_RESPONSE_KEEP_IDX_GOODNESS,
    SER_RESPONSE_KEEP_LEN,
)
from netsquid.components.instructions import INSTR_ROT_X
from netsquid.nodes import Node
from netsquid.qubits.ketstates import BellIndex
from netsquid_magic.state_delivery_sampler import PerfectStateSamplerFactory
from qlink_interface import (
    ReqCreateAndKeep,
    ReqCreateBase,
    ReqMeasureDirectly,
    ReqReceive,
    ReqRemoteStatePrep,
    ResCreateAndKeep,
)
from qlink_interface.interface import ResCreate

from pydynaa import EventExpression
from qoala.lang.ehi import UnitModule
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.request import EprRole, EprType, IqoalaRequest, RequestVirtIdMapping
from qoala.runtime.environment import (
    GlobalEnvironment,
    GlobalNodeInfo,
    LocalEnvironment,
)
from qoala.runtime.lhi import LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import (
    GenericToVanillaInterface,
    LhiConverter,
    NvToNvInterface,
)
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist.entdist import EntDist, GEDRequest
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.netstack import NetstackInterface, NetstackLatencies, NetstackProcessor
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice, QDeviceCommand
from qoala.sim.requests import NetstackCreateRequest, NetstackReceiveRequest
from qoala.util.constants import PI
from qoala.util.tests import (
    B00_DENS,
    B01_DENS,
    B10_DENS,
    S00_DENS,
    S10_DENS,
    TWO_MAX_MIXED,
    density_matrices_equal,
    has_multi_state,
    netsquid_run,
)


class MockNetstackInterface(NetstackInterface):
    def __init__(
        self,
        comp: NetstackComponent,
        local_env: LocalEnvironment,
        qdevice: QDevice,
        requests: List[GEDRequest],
    ) -> None:
        super().__init__(comp, local_env, qdevice, None, None)
        self._requests = requests


def create_n_qdevices(n: int, num_qubits: int = 1) -> List[QDevice]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
    qdevices: List[QDevice] = []
    for i in range(n):
        qproc = build_qprocessor_from_topology(name=f"qproc_{i}", topology=topology)
        node = Node(name=f"node_{i}", qmemory=qproc)
        qdevices.append(QDevice(node=node, topology=topology))

    return qdevices


def create_alice_bob_qdevices(num_qubits: int = 1) -> Tuple[QDevice, QDevice]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)

    alice_qproc = build_qprocessor_from_topology(name=f"qproc_alice", topology=topology)
    bob_qproc = build_qprocessor_from_topology(name=f"qproc_bob", topology=topology)

    alice_node = Node(name="alice", qmemory=alice_qproc)
    bob_node = Node(name="bob", qmemory=bob_qproc)

    alice_qdevice = QDevice(node=alice_node, topology=topology)
    bob_qdevice = QDevice(node=bob_node, topology=topology)

    return alice_qdevice, bob_qdevice


def create_request(node1_id: int, node2_id: int, local_qubit_id: int = 0) -> GEDRequest:
    return GEDRequest(
        local_node_id=node1_id, remote_node_id=node2_id, local_qubit_id=local_qubit_id
    )


def setup_components() -> Tuple[
    NetstackComponent, QDevice, NetstackComponent, QDevice, EntDist
]:
    alice, bob = create_alice_bob_qdevices(num_qubits=3)

    env = GlobalEnvironment()
    alice_info = GlobalNodeInfo(alice.node.name, alice.node.ID)
    env.add_node(alice.node.ID, alice_info)
    bob_info = GlobalNodeInfo(bob.node.name, bob.node.ID)
    env.add_node(bob.node.ID, bob_info)
    alice_comp = NetstackComponent(alice.node, env)
    bob_comp = NetstackComponent(bob.node, env)
    entdist_comp = EntDistComponent(env)

    ged = EntDist(nodes=[alice.node, bob.node], global_env=env, comp=entdist_comp)

    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(alice.node.ID, bob.node.ID, factory, kwargs=kwargs)

    return alice_comp, alice, bob_comp, bob, ged


def test_single_pair():
    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.wait(500)
            self.send_entdist_msg(Message(self._requests[0]))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.wait(800)
            self.send_entdist_msg(Message(self._requests[0]))

    alice_comp, alice_qdevice, bob_comp, bob_qdevice, ged = setup_components()
    env: GlobalEnvironment = ged._global_env
    alice_id = alice_comp.node.ID
    bob_id = bob_comp.node.ID

    request_alice = create_request(alice_id, bob_id)
    request_bob = create_request(bob_id, alice_id)

    alice_intf = AliceNetstackInterface(
        alice_comp,
        LocalEnvironment(env, alice_id),
        alice_qdevice,
        requests=[request_alice],
    )
    bob_intf = BobNetstackInterface(
        bob_comp, LocalEnvironment(env, bob_id), bob_qdevice, requests=[request_bob]
    )

    alice_intf.start()
    bob_intf.start()
    ged.start()
    ns.sim_run()

    alice_qubit = alice_qdevice.get_local_qubit(0)
    bob_qubit = bob_qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test_multiple_pairs():
    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            for request in self._requests:
                yield from self.wait(500)
                self.send_entdist_msg(Message(request))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            for request in self._requests:
                yield from self.wait(500)
                self.send_entdist_msg(Message(request))

    alice_comp, alice_qdevice, bob_comp, bob_qdevice, ged = setup_components()
    env: GlobalEnvironment = ged._global_env
    alice_id = alice_comp.node.ID
    bob_id = bob_comp.node.ID

    requests_alice = [
        create_request(alice_id, bob_id, 0),
        create_request(alice_id, bob_id, 1),
        create_request(alice_id, bob_id, 2),
    ]
    requests_bob = [
        create_request(bob_id, alice_id, 1),
        create_request(bob_id, alice_id, 2),
        create_request(bob_id, alice_id, 0),
    ]

    alice_intf = AliceNetstackInterface(
        alice_comp,
        LocalEnvironment(env, alice_id),
        alice_qdevice,
        requests=requests_alice,
    )
    bob_intf = BobNetstackInterface(
        bob_comp, LocalEnvironment(env, bob_id), bob_qdevice, requests=requests_bob
    )

    alice_intf.start()
    bob_intf.start()
    ged.start()
    ns.sim_run()

    alice_q0, alice_q1, alice_q2 = [alice_qdevice.get_local_qubit(i) for i in range(3)]
    bob_q0, bob_q1, bob_q2 = [bob_qdevice.get_local_qubit(i) for i in range(3)]
    assert has_multi_state([alice_q0, bob_q1], B00_DENS)
    assert has_multi_state([alice_q1, bob_q2], B00_DENS)
    assert has_multi_state([alice_q2, bob_q0], B00_DENS)


if __name__ == "__main__":
    test_single_pair()
    test_multiple_pairs()
