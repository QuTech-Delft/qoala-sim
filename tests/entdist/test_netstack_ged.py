from typing import Dict, Generator, List, Optional, Tuple, Type

import netsquid as ns
from netsquid.nodes import Node
from netsquid_magic.state_delivery_sampler import PerfectStateSamplerFactory

from pydynaa import EventExpression
from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.program import IqoalaProgram, ProgramMeta
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    IqoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
)
from qoala.lang.routine import LocalRoutine
from qoala.runtime.environment import (
    GlobalEnvironment,
    GlobalNodeInfo,
    LocalEnvironment,
)
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.egpmgr import EgpManager
from qoala.sim.entdist.entdist import EntDist, GEDRequest
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import NetstackInterface, NetstackLatencies
from qoala.sim.netstack.netstack import Netstack
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.util.tests import B00_DENS, has_multi_state


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


def create_alice_bob_qdevices(
    num_qubits: int = 1, alice_id: int = 0, bob_id: int = 1
) -> Tuple[QDevice, QDevice]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)

    alice_qproc = build_qprocessor_from_topology(name="qproc_alice", topology=topology)
    bob_qproc = build_qprocessor_from_topology(name="qproc_bob", topology=topology)

    alice_node = Node(name="alice", qmemory=alice_qproc, ID=alice_id)
    bob_node = Node(name="bob", qmemory=bob_qproc, ID=bob_id)

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


def test_single_pair_only_netstack_interface():
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


def test_multiple_pairs_only_netstack_interface():
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


def setup_components_full_netstack(
    num_qubits: int,
    alice_id: int,
    bob_id: int,
    alice_netstack_cls: Type[Netstack],
    bob_netstack_cls: Type[Netstack],
) -> Tuple[Netstack, Netstack, EntDist]:
    alice_qdevice, bob_qdevice = create_alice_bob_qdevices(
        num_qubits=num_qubits, alice_id=alice_id, bob_id=bob_id
    )

    env = GlobalEnvironment()
    alice_info = GlobalNodeInfo(alice_qdevice.node.name, alice_qdevice.node.ID)
    env.add_node(alice_qdevice.node.ID, alice_info)
    bob_info = GlobalNodeInfo(bob_qdevice.node.name, bob_qdevice.node.ID)
    env.add_node(bob_qdevice.node.ID, bob_info)
    alice_comp = NetstackComponent(alice_qdevice.node, env)
    bob_comp = NetstackComponent(bob_qdevice.node, env)
    entdist_comp = EntDistComponent(env)

    ged = EntDist(
        nodes=[alice_qdevice.node, bob_qdevice.node], global_env=env, comp=entdist_comp
    )

    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(alice_qdevice.node.ID, bob_qdevice.node.ID, factory, kwargs=kwargs)

    alice_netstack = alice_netstack_cls(
        comp=alice_comp,
        local_env=LocalEnvironment(env, alice_qdevice.node.ID),
        memmgr=MemoryManager("alice", alice_qdevice),
        egpmgr=EgpManager(),
        qdevice=alice_qdevice,
        latencies=NetstackLatencies.all_zero(),
    )
    bob_netstack = bob_netstack_cls(
        comp=bob_comp,
        local_env=LocalEnvironment(env, bob_qdevice.node.ID),
        memmgr=MemoryManager("bob", bob_qdevice),
        egpmgr=EgpManager(),
        qdevice=bob_qdevice,
        latencies=NetstackLatencies.all_zero(),
    )

    return alice_netstack, bob_netstack, ged


def test_single_pair_full_netstack():
    alice_id = 0
    bob_id = 1

    request_alice = create_request(alice_id, bob_id)
    request_bob = create_request(bob_id, alice_id)

    class AliceNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.processor.execute_ged_request(request_alice)

    class BobNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.processor.execute_ged_request(request_bob)

    alice_netstack, bob_netstack, ged = setup_components_full_netstack(
        1, alice_id, bob_id, AliceNetstack, BobNetstack
    )

    alice_netstack.start()
    bob_netstack.start()
    ged.start()
    ns.sim_run()

    alice_qubit = alice_netstack.qdevice.get_local_qubit(0)
    bob_qubit = bob_netstack.qdevice.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test_multiple_pairs_full_netstack():
    ns.sim_reset()

    alice_id = 0
    bob_id = 1

    requests_alice = [
        create_request(alice_id, bob_id, 0),
        create_request(alice_id, bob_id, 1),
    ]
    requests_bob = [
        create_request(bob_id, alice_id, 0),
        create_request(bob_id, alice_id, 2),
    ]

    class AliceNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            for request in requests_alice:
                yield from self.processor.execute_ged_request(request)

    class BobNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            for request in requests_bob:
                yield from self.processor.execute_ged_request(request)

    alice_netstack, bob_netstack, ged = setup_components_full_netstack(
        3, alice_id, bob_id, AliceNetstack, BobNetstack
    )

    alice_netstack.start()
    bob_netstack.start()
    ged.start()
    ns.sim_run()

    aq0 = alice_netstack.qdevice.get_local_qubit(0)
    bq0 = bob_netstack.qdevice.get_local_qubit(0)
    aq1 = alice_netstack.qdevice.get_local_qubit(1)
    bq2 = bob_netstack.qdevice.get_local_qubit(2)
    assert has_multi_state([aq0, bq0], B00_DENS)
    assert has_multi_state([aq1, bq2], B00_DENS)


def create_simple_request(
    remote_id: int, num_pairs: int, virt_ids: RequestVirtIdMapping
) -> IqoalaRequest:
    return IqoalaRequest(
        name="req",
        remote_id=remote_id,
        epr_socket_id=0,
        num_pairs=num_pairs,
        virt_ids=virt_ids,
        timeout=1000,
        fidelity=0.65,
        typ=EprType.CREATE_KEEP,
        role=EprRole.CREATE,
        result_array_addr=3,
    )


def create_process(
    num_qubits: int, routines: Optional[Dict[str, LocalRoutine]] = None
) -> IqoalaProcess:
    if routines is None:
        routines = {}
    program = IqoalaProgram(
        blocks=[], local_routines=routines, meta=ProgramMeta.empty("")
    )

    instance = ProgramInstance(
        pid=0,
        program=program,
        inputs=ProgramInput({}),
        tasks=ProgramTaskList.empty(program),
    )
    ehi = EhiBuilder.perfect_uniform(num_qubits, None, [], 0, [], 0)
    unit_module = UnitModule.from_full_ehi(ehi)
    mem = ProgramMemory(pid=0, unit_module=unit_module)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
        active_routines={},
    )
    return process


def test_single_pair_qoala_request():
    num_qubits = 3
    alice_id = 0
    bob_id = 1

    request_alice = create_simple_request(
        remote_id=bob_id,
        num_pairs=2,
        virt_ids=RequestVirtIdMapping.from_str("increment 0"),
    )
    routine_alice = RequestRoutine(
        name="req1",
        request=request_alice,
        callback_type=CallbackType.WAIT_ALL,
        callback=None,
    )

    requests_bob = [
        create_request(bob_id, alice_id, 0),
        create_request(bob_id, alice_id, 1),
    ]
    process_alice = create_process(num_qubits)

    class AliceNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.processor.assign_request_routine(
                process_alice, routine_alice
            )

    class BobNetstack(Netstack):
        def run(self) -> Generator[EventExpression, None, None]:
            for request in requests_bob:
                yield from self.processor.execute_ged_request(request)

    alice_netstack, bob_netstack, ged = setup_components_full_netstack(
        num_qubits, alice_id, bob_id, AliceNetstack, BobNetstack
    )
    alice_netstack.interface.memmgr.add_process(process_alice)

    alice_netstack.start()
    bob_netstack.start()
    ged.start()
    ns.sim_run()

    aq0 = alice_netstack.qdevice.get_local_qubit(0)
    bq0 = bob_netstack.qdevice.get_local_qubit(0)
    aq1 = alice_netstack.qdevice.get_local_qubit(1)
    bq1 = bob_netstack.qdevice.get_local_qubit(1)
    assert has_multi_state([aq0, bq0], B00_DENS)
    assert has_multi_state([aq1, bq1], B00_DENS)


if __name__ == "__main__":
    test_single_pair_only_netstack_interface()
    test_multiple_pairs_only_netstack_interface()
    test_single_pair_full_netstack()
    test_multiple_pairs_full_netstack()
    test_single_pair_qoala_request()
