import itertools
from typing import List, Tuple

import netsquid as ns
import numpy as np
import pytest
from netsquid import QFormalism
from netsquid.nodes import Node
from netsquid.qubits import ketstates
from netsquid_magic.state_delivery_sampler import (
    DeliverySample,
    DepolariseWithFailureStateSamplerFactory,
    PerfectStateSamplerFactory,
    StateDeliverySampler,
)

from qoala.runtime.environment import GlobalEnvironment, GlobalNodeInfo
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist.entdist import (
    EntDist,
    EprDeliverySample,
    GEDRequest,
    JointRequest,
)
from qoala.sim.entdist.entdistcomp import EntDistComponent
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


def create_n_nodes(n: int, num_qubits: int = 1) -> List[Node]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
    nodes: List[Node] = []
    for i in range(n):
        qproc = build_qprocessor_from_topology(name=f"qproc_{i}", topology=topology)
        nodes.append(Node(name=f"node_{i}", qmemory=qproc))

    return nodes


def create_request(node1_id: int, node2_id: int) -> GEDRequest:
    return GEDRequest(local_node_id=node1_id, remote_node_id=node2_id, local_qubit_id=0)


def create_joint_request(
    node1_id: int, node2_id: int, node1_qubit_id: int = 0, node2_qubit_id: int = 0
) -> JointRequest:
    return JointRequest(
        node1_id=node1_id,
        node2_id=node2_id,
        node1_qubit_id=node1_qubit_id,
        node2_qubit_id=node2_qubit_id,
    )


def create_entdist(nodes: List[Node]) -> EntDist:
    env = GlobalEnvironment()
    for node in nodes:
        node_info = GlobalNodeInfo(node.name, node.ID)
        env.add_node(node.ID, node_info)
    comp = EntDistComponent(env)
    return EntDist(nodes=nodes, global_env=env, comp=comp)


def test_add_sampler():
    alice, bob = create_n_nodes(2)

    ged = create_entdist(nodes=[alice, bob])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert len(ged._samplers) == 1
    assert type(ged._samplers[(alice.ID, bob.ID)]) == StateDeliverySampler


def test_add_sampler_many_nodes():
    n = 10
    nodes = create_n_nodes(n)

    ged = create_entdist(nodes=nodes)
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(nodes[0].ID, nodes[1].ID, factory, kwargs=kwargs)

    with pytest.raises(ValueError):
        ged.add_sampler(nodes[0].ID, nodes[1].ID, factory, kwargs=kwargs)
    with pytest.raises(ValueError):
        ged.add_sampler(nodes[1].ID, nodes[0].ID, factory, kwargs=kwargs)

    ged.add_sampler(nodes[0].ID, nodes[2].ID, factory, kwargs=kwargs)
    ged.add_sampler(nodes[2].ID, nodes[9].ID, factory, kwargs=kwargs)

    assert len(ged._samplers) == 3
    for (i, j) in [(0, 1), (0, 2), (2, 9)]:
        assert type(ged._samplers[(nodes[i].ID, nodes[j].ID)]) == StateDeliverySampler


def test_sample_perfect():
    sampler_factory = PerfectStateSamplerFactory()

    kwargs = {"cycle_time": 1000}
    sampler: StateDeliverySampler = sampler_factory.create_state_delivery_sampler(
        **kwargs
    )

    assert sampler._cycle_time == 1000
    assert sampler._success_probability == 1
    number_of_attempts = np.random.geometric(p=sampler._success_probability) - 1
    assert number_of_attempts == 0

    raw_sample: DeliverySample = sampler.sample()
    sample = EprDeliverySample.from_ns_magic_delivery_sample(raw_sample)

    expected_duration = 1000 * number_of_attempts  # = 0
    assert sample.duration == expected_duration
    assert density_matrices_equal(sample.state, B00_DENS)


def test_sample_depolar():
    sampler_factory = DepolariseWithFailureStateSamplerFactory()
    kwargs = {"cycle_time": 1000, "prob_max_mixed": 0.2, "prob_success": 1}
    sampler: StateDeliverySampler = sampler_factory.create_state_delivery_sampler(
        **kwargs
    )
    raw_sample: DeliverySample = sampler.sample()
    sample = EprDeliverySample.from_ns_magic_delivery_sample(raw_sample)

    expected = 0.2 * TWO_MAX_MIXED + 0.8 * B00_DENS

    assert density_matrices_equal(sample.state, expected)


def test_create_epr_pair_with_state():
    alice, bob = create_n_nodes(2)
    ged = create_entdist(nodes=[alice, bob])

    q0, q1 = ged.create_epr_pair_with_state(B00_DENS)
    assert has_multi_state([q0, q1], B00_DENS)

    q0, q1 = ged.create_epr_pair_with_state(S10_DENS)
    assert has_multi_state([q0, q1], S10_DENS)


def test_deliver_perfect():
    alice, bob = create_n_nodes(2)

    ged = create_entdist(nodes=[alice, bob])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert not alice.qmemory.mem_positions[0].in_use
    assert not bob.qmemory.mem_positions[0].in_use

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(ged.deliver(alice.ID, 0, bob.ID, 0, state_delay=1000))
    assert ns.sim_time() == 1000

    assert alice.qmemory.mem_positions[0].in_use
    assert bob.qmemory.mem_positions[0].in_use

    alice_qubit = alice.qmemory.peek([0])[0]
    bob_qubit = bob.qmemory.peek([0])[0]
    assert alice_qubit is not None
    assert bob_qubit is not None
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test_deliver_depolar():
    alice, bob = create_n_nodes(2)

    ged = create_entdist(nodes=[alice, bob])
    factory = DepolariseWithFailureStateSamplerFactory()
    kwargs = {"cycle_time": 10, "prob_max_mixed": 0.2, "prob_success": 1}
    ged.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert not alice.qmemory.mem_positions[0].in_use
    assert not bob.qmemory.mem_positions[0].in_use

    ns.sim_reset()
    ns.set_qstate_formalism(QFormalism.DM)
    assert ns.sim_time() == 0
    netsquid_run(ged.deliver(alice.ID, 0, bob.ID, 0, state_delay=1000))
    assert ns.sim_time() == 1000

    assert alice.qmemory.mem_positions[0].in_use
    assert bob.qmemory.mem_positions[0].in_use

    alice_qubit = alice.qmemory.peek([0])[0]
    bob_qubit = bob.qmemory.peek([0])[0]
    assert alice_qubit is not None
    assert bob_qubit is not None

    expected = 0.2 * TWO_MAX_MIXED + 0.8 * B00_DENS
    assert has_multi_state([alice_qubit, bob_qubit], expected)


def test_put_request():
    alice, bob = create_n_nodes(2)

    ged = create_entdist(nodes=[alice, bob])

    assert len(ged._requests) == 2
    assert len(ged.get_requests(alice.ID)) == 0
    assert len(ged.get_requests(bob.ID)) == 0

    request = create_request(alice.ID, bob.ID)
    ged.put_request(request)

    assert len(ged.get_requests(alice.ID)) == 1
    assert ged.get_requests(alice.ID)[0] == request
    assert len(ged.get_requests(bob.ID)) == 0


def test_put_request_many_nodes():
    n = 10
    nodes = create_n_nodes(n)

    ged = create_entdist(nodes=nodes)

    assert len(ged._requests) == n

    request01 = create_request(nodes[0].ID, nodes[1].ID)
    ged.put_request(request01)

    request10 = create_request(nodes[1].ID, nodes[0].ID)
    ged.put_request(request10)

    request05 = create_request(nodes[0].ID, nodes[5].ID)
    ged.put_request(request05)

    assert ged.get_requests(nodes[0].ID)[0] == request01
    assert ged.get_requests(nodes[0].ID)[1] == request05
    assert ged.get_requests(nodes[1].ID)[0] == request10

    request_invalid = create_request(nodes[0].ID, 100)
    with pytest.raises(ValueError):
        ged.put_request(request_invalid)


def test_get_remote_request_for():
    alice, bob, charlie = create_n_nodes(3)
    ged = create_entdist(nodes=[alice, bob, charlie])

    request_alice = create_request(alice.ID, bob.ID)
    ged.put_request(request_alice)

    # Only Alice's request registered; no corresponding request from Bob yet.
    assert ged.get_remote_request_for(request_alice) is None

    request_bob = create_request(bob.ID, alice.ID)
    ged.put_request(request_bob)

    # Bob's first request corresponds to Alice's request.
    assert ged.get_remote_request_for(request_alice) == 0

    invalid_request = create_request(100, 100)
    with pytest.raises(ValueError):
        ged.get_remote_request_for(invalid_request)

    # Put 2 new identical requests for Alice.
    request_alice_1 = create_request(alice.ID, bob.ID)
    ged.put_request(request_alice_1)
    request_alice_2 = create_request(alice.ID, bob.ID)
    ged.put_request(request_alice_2)

    # Put 2 new requests for Bob. Only the 2nd corresponds to Alice.
    request_bob_1 = create_request(bob.ID, charlie.ID)
    ged.put_request(request_bob_1)
    request_bob_2 = create_request(bob.ID, alice.ID)
    ged.put_request(request_bob_2)

    # Bob's first request still corresponds to Alice.
    assert ged.get_remote_request_for(request_alice_1) == 0
    # Remove this request.
    ged.pop_request(bob.ID, 0)
    # The corresponding request is now 2nd in Bob's queue.
    assert ged.get_remote_request_for(request_alice_1) == 1


def test_get_next_joint_request():
    alice, bob = create_n_nodes(2)
    ged = create_entdist(nodes=[alice, bob])

    request_alice = create_request(alice.ID, bob.ID)
    ged.put_request(request_alice)
    assert len(ged.get_requests(alice.ID)) == 1
    assert len(ged.get_requests(bob.ID)) == 0

    # Only Alice's request registered; no corresponding request from Bob yet.
    assert ged.get_next_joint_request() is None

    # No requests should have been popped.
    assert len(ged.get_requests(alice.ID)) == 1
    assert len(ged.get_requests(bob.ID)) == 0

    request_bob = create_request(bob.ID, alice.ID)
    ged.put_request(request_bob)
    assert len(ged.get_requests(bob.ID)) == 1

    # Alice and Bob have corresponding requests.
    assert ged.get_next_joint_request() is not None
    assert len(ged.get_requests(alice.ID)) == 0
    assert len(ged.get_requests(bob.ID)) == 0

    # Requests have been popped.
    assert ged.get_next_joint_request() is None


def test_get_next_joint_request_2():
    alice, bob, charlie = create_n_nodes(3)
    ged = create_entdist(nodes=[alice, bob, charlie])

    request_ab = create_request(alice.ID, bob.ID)
    request_ac = create_request(alice.ID, charlie.ID)
    ged.put_request(request_ab)
    ged.put_request(request_ac)

    assert ged.get_next_joint_request() is None

    request_bc = create_request(bob.ID, charlie.ID)
    request_ba = create_request(bob.ID, alice.ID)
    ged.put_request(request_bc)
    ged.put_request(request_ba)

    # Alice and Bob have corresponding requests.
    assert ged.get_next_joint_request() == create_joint_request(alice.ID, bob.ID)

    # Both Alice and Bob still have a request pending with Charlie.
    assert len(ged.get_requests(alice.ID)) == 1
    assert len(ged.get_requests(bob.ID)) == 1

    # No next joint request at this moment.
    assert ged.get_next_joint_request() is None

    request_cb = create_request(charlie.ID, bob.ID)
    request_ca = create_request(charlie.ID, alice.ID)
    ged.put_request(request_cb)
    ged.put_request(request_ca)

    # Alice <-> Charlie
    # TODO: even though Bob put his request first, Alice's request is handled first
    # since Alice is first in the node list. Improve this!
    assert ged.get_next_joint_request() == create_joint_request(alice.ID, charlie.ID)
    # Bob <-> Charlie
    assert ged.get_next_joint_request() == create_joint_request(bob.ID, charlie.ID)


def test_deliver_request():
    alice, bob = create_n_nodes(2, num_qubits=2)

    ged = create_entdist(nodes=[alice, bob])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    ged.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert not alice.qmemory.mem_positions[0].in_use
    assert not bob.qmemory.mem_positions[0].in_use

    alice_mem = 0
    bob_mem = 0
    joint_request = create_joint_request(alice.ID, bob.ID, alice_mem, bob_mem)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(ged.deliver_request(joint_request))
    assert ns.sim_time() == 1000

    alice_qubit = alice.qmemory.peek([0])[0]
    bob_qubit = bob.qmemory.peek([0])[0]
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)

    alice_mem = 1
    bob_mem = 1
    joint_request = create_joint_request(alice.ID, bob.ID, alice_mem, bob_mem)
    netsquid_run(ged.deliver_request(joint_request))
    assert ns.sim_time() == 2000

    alice_qubit = alice.qmemory.peek([1])[0]
    bob_qubit = bob.qmemory.peek([1])[0]
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test_deliver_request_multiple_nodes():
    alice, bob, charlie, david = create_n_nodes(4, num_qubits=2)

    ged = create_entdist(nodes=[alice, bob, charlie, david])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    for (node1, node2) in itertools.combinations([alice, bob, charlie, david], 2):
        ged.add_sampler(node1.ID, node2.ID, factory, kwargs=kwargs)

    req_ab = create_joint_request(alice.ID, bob.ID, 0, 0)
    req_ac = create_joint_request(alice.ID, charlie.ID, 1, 0)
    req_dc = create_joint_request(david.ID, charlie.ID, 0, 1)
    req_bd = create_joint_request(bob.ID, david.ID, 1, 1)

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(ged.deliver_request(req_ab))
    netsquid_run(ged.deliver_request(req_ac))
    netsquid_run(ged.deliver_request(req_dc))
    netsquid_run(ged.deliver_request(req_bd))
    assert ns.sim_time() == 4000

    alice_qubits = alice.qmemory.peek([0, 1])
    bob_qubits = bob.qmemory.peek([0, 1])
    charlie_qubits = charlie.qmemory.peek([0, 1])
    david_qubits = david.qmemory.peek([0, 1])
    assert has_multi_state([alice_qubits[0], bob_qubits[0]], B00_DENS)
    assert has_multi_state([alice_qubits[1], charlie_qubits[0]], B00_DENS)
    assert has_multi_state([david_qubits[0], charlie_qubits[1]], B00_DENS)
    assert has_multi_state([bob_qubits[1], david_qubits[1]], B00_DENS)


if __name__ == "__main__":
    test_add_sampler()
    test_add_sampler_many_nodes()
    test_sample_perfect()
    test_sample_depolar()
    test_create_epr_pair_with_state()
    test_deliver_perfect()
    test_deliver_depolar()
    test_put_request()
    test_put_request_many_nodes()
    test_get_remote_request_for()
    test_get_next_joint_request()
    test_get_next_joint_request_2()
    test_deliver_request()
    test_deliver_request_multiple_nodes()
