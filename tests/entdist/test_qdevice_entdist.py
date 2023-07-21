import itertools
from typing import List, Optional, Tuple

import netsquid as ns
from netsquid.nodes import Node

from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.lhi import LhiLinkInfo, LhiTopologyBuilder
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist.entdist import EntDist, EntDistRequest
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.qdevice import QDevice
from qoala.util.math import B00_DENS, has_multi_state
from qoala.util.tests import netsquid_run


def create_n_qdevices(n: int, num_qubits: int = 1) -> List[QDevice]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
    qdevices: List[QDevice] = []
    for i in range(n):
        qproc = build_qprocessor_from_topology(name=f"qproc_{i}", topology=topology)
        node = Node(name=f"node_{i}", qmemory=qproc)
        qdevices.append(QDevice(node=node, topology=topology))

    return qdevices


def create_entdist(qdevices: List[QDevice]) -> EntDist:
    ehi_network = EhiNetworkInfo.only_nodes(
        {qdevice.node.ID: qdevice.node.name for qdevice in qdevices}
    )
    comp = EntDistComponent(ehi_network)
    entdist = EntDist(
        nodes=[qdevice.node for qdevice in qdevices], ehi_network=ehi_network, comp=comp
    )

    link_info = LhiLinkInfo.perfect(1000)
    for qd1, qd2 in itertools.combinations(qdevices, 2):
        entdist.add_sampler(qd1.node.ID, qd2.node.ID, link_info)

    return entdist


def create_request(
    node1_id: int,
    node2_id: int,
    local_qubit_id: int = 0,
    lpid: Optional[int] = 0,
    rpid: Optional[int] = 0,
) -> EntDistRequest:
    return EntDistRequest(
        local_node_id=node1_id,
        remote_node_id=node2_id,
        local_qubit_id=local_qubit_id,
        local_pid=lpid,
        remote_pid=rpid,
    )


def create_request_pair(
    node1_id: int,
    node2_id: int,
    node1_qubit_id: int = 0,
    node2_qubit_id: int = 0,
    lpid: Optional[int] = 0,
    rpid: Optional[int] = 0,
) -> Tuple[EntDistRequest]:
    req1 = EntDistRequest(
        local_node_id=node1_id,
        remote_node_id=node2_id,
        local_qubit_id=node1_qubit_id,
        local_pid=lpid,
        remote_pid=rpid,
    )
    req2 = EntDistRequest(
        local_node_id=node2_id,
        remote_node_id=node1_id,
        local_qubit_id=node2_qubit_id,
        local_pid=rpid,
        remote_pid=lpid,
    )
    return req1, req2


def test1():
    alice, bob = create_n_qdevices(2)
    entdist = create_entdist([alice, bob])

    request_alice = create_request(alice.node.ID, bob.node.ID, 0, [0], [0])
    request_bob = create_request(bob.node.ID, alice.node.ID, 0, [0], [0])

    entdist.put_request(request_alice)
    entdist.put_request(request_bob)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(entdist.serve_all_requests())
    assert ns.sim_time() == 1000

    alice_qubit = alice.get_local_qubit(0)
    bob_qubit = bob.get_local_qubit(0)
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test2():
    qdevices = create_n_qdevices(4, num_qubits=2)
    assert all(qdevice.get_qubit_count() == 2 for qdevice in qdevices)

    entdist = create_entdist(qdevices)

    ids = [qdevices[i].node.ID for i in range(4)]

    req01, req10 = create_request_pair(ids[0], ids[1], 0, 0, [0], [0])
    entdist.put_request(req01)
    entdist.put_request(req10)

    req02, req20 = create_request_pair(ids[0], ids[2], 1, 0, [0], [0])
    entdist.put_request(req02)
    entdist.put_request(req20)

    req13, req31 = create_request_pair(ids[1], ids[3], 1, 0, [0], [0])
    entdist.put_request(req13)
    entdist.put_request(req31)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(entdist.serve_all_requests())
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
