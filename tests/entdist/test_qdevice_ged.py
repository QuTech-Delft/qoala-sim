import itertools
from typing import List, Tuple

import netsquid as ns
from netsquid.nodes import Node
from netsquid_magic.state_delivery_sampler import PerfectStateSamplerFactory

from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist.entdist import EntDist, GEDRequest
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.qdevice import QDevice
from qoala.util.tests import B00_DENS, has_multi_state, netsquid_run


def create_n_qdevices(n: int, num_qubits: int = 1) -> List[QDevice]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
    qdevices: List[QDevice] = []
    for i in range(n):
        qproc = build_qprocessor_from_topology(name=f"qproc_{i}", topology=topology)
        node = Node(name=f"node_{i}", qmemory=qproc)
        qdevices.append(QDevice(node=node, topology=topology))

    return qdevices


def create_entdist(qdevices: List[QDevice]) -> EntDist:
    env = GlobalEnvironment()
    for qdevice in qdevices:
        node_info = GlobalNodeInfo(qdevice.node.name, qdevice.node.ID)
        env.add_node(qdevice.node.ID, node_info)
    comp = EntDistComponent(env)
    ged = EntDist(
        nodes=[qdevice.node for qdevice in qdevices], global_env=env, comp=comp
    )

    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    for qd1, qd2 in itertools.combinations(qdevices, 2):
        ged.add_sampler(qd1.node.ID, qd2.node.ID, factory, kwargs=kwargs)

    return ged


def create_request(node1_id: int, node2_id: int, local_qubit_id: int = 0) -> GEDRequest:
    return GEDRequest(
        local_node_id=node1_id, remote_node_id=node2_id, local_qubit_id=local_qubit_id
    )


def create_request_pair(
    node1_id: int, node2_id: int, node1_qubit_id: int = 0, node2_qubit_id: int = 0
) -> Tuple[GEDRequest]:
    req1 = GEDRequest(
        local_node_id=node1_id, remote_node_id=node2_id, local_qubit_id=node1_qubit_id
    )
    req2 = GEDRequest(
        local_node_id=node2_id, remote_node_id=node1_id, local_qubit_id=node2_qubit_id
    )
    return req1, req2


def test1():
    alice, bob = create_n_qdevices(2)
    ged = create_entdist([alice, bob])

    request_alice = create_request(alice.node.ID, bob.node.ID)
    request_bob = create_request(bob.node.ID, alice.node.ID)

    ged.put_request(request_alice)
    ged.put_request(request_bob)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(ged.serve_all_requests())
    assert ns.sim_time() == 1000

    alice_qubit = alice.get_local_qubit(0)
    bob_qubit = bob.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test2():
    qdevices = create_n_qdevices(4, num_qubits=2)
    ged = create_entdist(qdevices)

    ids = [qdevices[i].node.ID for i in range(4)]

    req01, req10 = create_request_pair(ids[0], ids[1], 0, 0)
    ged.put_request(req01)
    ged.put_request(req10)

    req02, req20 = create_request_pair(ids[0], ids[2], 1, 0)
    ged.put_request(req02)
    ged.put_request(req20)

    req13, req31 = create_request_pair(ids[1], ids[3], 1, 0)
    ged.put_request(req13)
    ged.put_request(req31)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(ged.serve_all_requests())
    assert ns.sim_time() == 3 * 1000  # 3 request pairs

    n0_q0 = qdevices[0].get_local_qubit(0)
    n0_q1 = qdevices[0].get_local_qubit(1)
    n1_q0 = qdevices[1].get_local_qubit(0)
    n1_q1 = qdevices[1].get_local_qubit(1)
    n2_q0 = qdevices[2].get_local_qubit(0)
    n2_q1 = qdevices[2].get_local_qubit(1)
    n3_q0 = qdevices[3].get_local_qubit(0)
    n3_q1 = qdevices[3].get_local_qubit(1)

    assert has_multi_state([n0_q0, n1_q0], B00_DENS)
    assert has_multi_state([n0_q1, n2_q0], B00_DENS)
    assert has_multi_state([n1_q1, n3_q0], B00_DENS)

    assert n2_q1 is None
    assert n3_q1 is None


if __name__ == "__main__":
    test1()
    test2()
