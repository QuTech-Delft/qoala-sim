from typing import Generator

import netsquid as ns
from netsquid.components.cchannel import ClassicalChannel

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.lhi import LhiLatencies, LhiTopologyBuilder
from qoala.runtime.message import Message
from qoala.runtime.ntf import GenericNtf
from qoala.sim.procnode import ProcNode
from qoala.sim.procnodecomp import ProcNodeComponent


def create_procnodecomp(num_other_nodes: int) -> ProcNodeComponent:
    nodes = {id: f"node_{id}" for id in range(1, num_other_nodes + 1)}
    nodes[0] = "alice"
    ehi_network = EhiNetworkInfo.only_nodes(nodes)

    return ProcNodeComponent(name="alice", qprocessor=None, ehi_network=ehi_network)


def test_no_other_nodes():
    comp = create_procnodecomp(num_other_nodes=0)

    # should not have 2 entdist ports
    assert len(comp.ports) == 2
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports

    assert len(comp.subcomponents) == 3


def test_one_other_node():
    comp = create_procnodecomp(num_other_nodes=1)

    # should have 2 host peer ports + 2 netstack peer ports + 2 entdist ports
    assert len(comp.ports) == 6
    assert "host_peer_node_1_in" in comp.ports
    assert "host_peer_node_1_out" in comp.ports
    assert "netstack_peer_node_1_in" in comp.ports
    assert "netstack_peer_node_1_out" in comp.ports
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports

    # Test properties
    assert comp.host_peer_in_port("node_1") == comp.ports["host_peer_node_1_in"]
    assert comp.host_peer_out_port("node_1") == comp.ports["host_peer_node_1_out"]
    assert comp.netstack_peer_in_port("node_1") == comp.ports["netstack_peer_node_1_in"]
    assert (
        comp.netstack_peer_out_port("node_1") == comp.ports["netstack_peer_node_1_out"]
    )

    assert len(comp.subcomponents) == 3


def test_many_other_nodes():
    comp = create_procnodecomp(num_other_nodes=5)

    # should 5 * 4 peer ports + 2 entdist ports
    assert len(comp.ports) == 22

    for i in range(1, 6):
        assert f"host_peer_node_{i}_in" in comp.ports
        assert f"host_peer_node_{i}_out" in comp.ports
        assert f"netstack_peer_node_{i}_in" in comp.ports
        assert f"netstack_peer_node_{i}_out" in comp.ports
        # Test properties
        assert (
            comp.host_peer_in_port(f"node_{i}") == comp.ports[f"host_peer_node_{i}_in"]
        )
        assert (
            comp.host_peer_out_port(f"node_{i}")
            == comp.ports[f"host_peer_node_{i}_out"]
        )
        assert (
            comp.netstack_peer_in_port(f"node_{i}")
            == comp.ports[f"netstack_peer_node_{i}_in"]
        )
        assert (
            comp.netstack_peer_out_port(f"node_{i}")
            == comp.ports[f"netstack_peer_node_{i}_out"]
        )

    assert len(comp.subcomponents) == 3


def test_connection_with_channel():
    ns.sim_reset()

    nodes = {0: "alice", 1: "bob"}

    ehi_network = EhiNetworkInfo.only_nodes(nodes)
    alice = ProcNodeComponent("alice", None, ehi_network=ehi_network, node_id=0)
    bob = ProcNodeComponent("bob", None, ehi_network=ehi_network, node_id=1)

    channel_ab = ClassicalChannel("chan_ab", delay=1000)

    alice.host_peer_out_port("bob").connect(channel_ab.ports["send"])
    channel_ab.ports["recv"].connect(bob.host_peer_in_port("alice"))
    alice.host_peer_in_port("bob").connect(bob.host_peer_out_port("alice"))

    class AliceProcnode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            self.host.interface.send_peer_msg("bob", Message(0, 0, "hello"))

    class BobProcnode(ProcNode):
        def run(self) -> Generator[EventExpression, None, None]:
            assert ns.sim_time() == 0
            msg = yield from self.host.interface.receive_peer_msg("alice")
            assert ns.sim_time() == 1000
            assert msg == Message(0, 0, "hello")

    topology = LhiTopologyBuilder.perfect_uniform_default_gates(1)
    latencies = LhiLatencies.all_zero()
    ntf = GenericNtf()
    alice_proc = AliceProcnode(
        "alice", None, topology, latencies, ntf, ehi_network, node=alice
    )
    bob_proc = BobProcnode("bob", None, topology, latencies, ntf, ehi_network, node=bob)

    alice_proc.start()
    bob_proc.start()

    ns.sim_run()


if __name__ == "__main__":
    test_no_other_nodes()
    test_one_other_node()
    test_many_other_nodes()

    test_connection_with_channel()
