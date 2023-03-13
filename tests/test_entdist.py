from typing import Tuple

import netsquid as ns
import numpy as np
from netsquid import QFormalism
from netsquid.nodes import Node
from netsquid.qubits import ketstates
from netsquid_magic.state_delivery_sampler import (
    DeliverySample,
    DepolariseWithFailureStateSamplerFactory,
    DoubleClickDeliverySamplerFactory,
    PerfectStateSamplerFactory,
    SingleClickDeliverySamplerFactory,
    StateDeliverySampler,
)
from netsquid_nv.state_delivery_sampler_factory import NVStateDeliverySamplerFactory

from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.entdist import EprDeliverySample, GlobalEntanglementDistributor
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


def create_two_nodes() -> Tuple[Node, Node]:
    topology = LhiTopologyBuilder.perfect_uniform_default_gates(1)
    alice_qproc = build_qprocessor_from_topology(name="alice_qproc", topology=topology)
    alice = Node(name="alice", qmemory=alice_qproc)
    bob_qproc = build_qprocessor_from_topology(name="bob_qproc", topology=topology)
    bob = Node(name="bob", qmemory=bob_qproc)

    return alice, bob


def test_add_sampler():
    alice, bob = create_two_nodes()

    global_dist = GlobalEntanglementDistributor(nodes=[alice, bob])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    global_dist.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert len(global_dist._samplers) == 1
    assert type(global_dist._samplers[(alice.ID, bob.ID)]) == StateDeliverySampler


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
    alice, bob = create_two_nodes()
    global_dist = GlobalEntanglementDistributor(nodes=[alice, bob])

    q0, q1 = global_dist.create_epr_pair_with_state(B00_DENS)
    assert has_multi_state([q0, q1], B00_DENS)

    q0, q1 = global_dist.create_epr_pair_with_state(S10_DENS)
    assert has_multi_state([q0, q1], S10_DENS)


def test_deliver_perfect():
    alice, bob = create_two_nodes()

    global_dist = GlobalEntanglementDistributor(nodes=[alice, bob])
    factory = PerfectStateSamplerFactory()
    kwargs = {"cycle_time": 1000}
    global_dist.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert not alice.qmemory.mem_positions[0].in_use
    assert not bob.qmemory.mem_positions[0].in_use

    ns.sim_reset()
    assert ns.sim_time() == 0
    netsquid_run(global_dist.deliver(alice.ID, 0, bob.ID, 0, state_delay=1000))
    assert ns.sim_time() == 1000

    assert alice.qmemory.mem_positions[0].in_use
    assert bob.qmemory.mem_positions[0].in_use

    alice_qubit = alice.qmemory.peek([0])[0]
    bob_qubit = bob.qmemory.peek([0])[0]
    assert alice_qubit is not None
    assert bob_qubit is not None
    assert has_multi_state([alice_qubit, bob_qubit], B00_DENS)


def test_deliver_depolar():
    alice, bob = create_two_nodes()

    global_dist = GlobalEntanglementDistributor(nodes=[alice, bob])
    factory = DepolariseWithFailureStateSamplerFactory()
    kwargs = {"cycle_time": 10, "prob_max_mixed": 0.2, "prob_success": 1}
    global_dist.add_sampler(alice.ID, bob.ID, factory, kwargs=kwargs)

    assert not alice.qmemory.mem_positions[0].in_use
    assert not bob.qmemory.mem_positions[0].in_use

    ns.sim_reset()
    ns.set_qstate_formalism(QFormalism.DM)
    assert ns.sim_time() == 0
    netsquid_run(global_dist.deliver(alice.ID, 0, bob.ID, 0, state_delay=1000))
    assert ns.sim_time() == 1000

    assert alice.qmemory.mem_positions[0].in_use
    assert bob.qmemory.mem_positions[0].in_use

    alice_qubit = alice.qmemory.peek([0])[0]
    bob_qubit = bob.qmemory.peek([0])[0]
    assert alice_qubit is not None
    assert bob_qubit is not None

    expected = 0.2 * TWO_MAX_MIXED + 0.8 * B00_DENS
    assert has_multi_state([alice_qubit, bob_qubit], expected)


if __name__ == "__main__":
    test_add_sampler()
    test_sample_perfect()
    test_sample_depolar()
    test_create_epr_pair_with_state()
    test_deliver_perfect()
    test_deliver_depolar()
