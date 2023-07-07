from __future__ import annotations

from typing import Generator

import netsquid as ns
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.netstack import NetstackComponent, NetstackInterface


class MockNetstackInterface(NetstackInterface):
    def __init__(self, comp: NetstackComponent, ehi_network: EhiNetworkInfo) -> None:
        super().__init__(comp, ehi_network, None, None)


def create_netstackcomp(num_other_nodes: int) -> NetstackComponent:
    node = Node(name="alice", ID=0)

    nodes = {id: f"node_{id}" for id in range(1, num_other_nodes + 1)}
    nodes[0] = "alice"
    ehi_network = EhiNetworkInfo.only_nodes(nodes)

    return NetstackComponent(node, ehi_network)


def test_no_other_nodes():
    comp = create_netstackcomp(num_other_nodes=0)

    # should have host_{in|out}, qnos_{in|out}, qnos_mem_{in|out}, entdist_{in|out} ports
    assert len(comp.ports) == 8
    assert "host_in" in comp.ports
    assert "host_out" in comp.ports
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "qnos_mem_in" in comp.ports
    assert "qnos_mem_out" in comp.ports
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports


def test_one_other_node():
    comp = create_netstackcomp(num_other_nodes=1)

    # should have 2 host ports + 4 qnos ports + 2 entdist ports + 2 peer ports
    assert len(comp.ports) == 10
    assert "host_in" in comp.ports
    assert "host_out" in comp.ports
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "qnos_mem_in" in comp.ports
    assert "qnos_mem_out" in comp.ports
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports

    assert "peer_node_1_in" in comp.ports
    assert "peer_node_1_out" in comp.ports

    # Test properties
    assert comp.peer_in_port("node_1") == comp.ports["peer_node_1_in"]
    assert comp.peer_out_port("node_1") == comp.ports["peer_node_1_out"]


def test_many_other_nodes():
    comp = create_netstackcomp(num_other_nodes=5)

    # should have 2 host ports + 4 qnos ports + 2 entdist ports + 5 * 2 peer ports
    assert len(comp.ports) == 18
    assert "host_in" in comp.ports
    assert "host_out" in comp.ports
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "qnos_mem_in" in comp.ports
    assert "qnos_mem_out" in comp.ports
    assert "entdist_in" in comp.ports
    assert "entdist_out" in comp.ports

    for i in range(1, 6):
        assert f"peer_node_{i}_in" in comp.ports
        assert f"peer_node_{i}_out" in comp.ports
        # Test properties
        assert comp.peer_in_port(f"node_{i}") == comp.ports[f"peer_node_{i}_in"]
        assert comp.peer_out_port(f"node_{i}") == comp.ports[f"peer_node_{i}_out"]


def test_connection():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    ehi_network = EhiNetworkInfo.only_nodes({alice.ID: alice.name, bob.ID: bob.name})

    alice_comp = NetstackComponent(alice, ehi_network)
    bob_comp = NetstackComponent(bob, ehi_network)

    alice_comp.peer_out_port("bob").connect(bob_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("bob").connect(bob_comp.peer_out_port("alice"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_peer_msg("bob", Message(0, 0, "hello"))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello"

    alice_intf = AliceNetstackInterface(alice_comp, ehi_network)
    bob_intf = BobNetstackInterface(bob_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()

    ns.sim_run()


def test_three_way_connection():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    charlie = Node(name="charlie", ID=2)
    ehi_network = EhiNetworkInfo.only_nodes(
        {alice.ID: alice.name, bob.ID: bob.name, charlie.ID: charlie.name}
    )

    alice_comp = NetstackComponent(alice, ehi_network)
    bob_comp = NetstackComponent(bob, ehi_network)
    charlie_comp = NetstackComponent(charlie, ehi_network)

    alice_comp.peer_out_port("bob").connect(bob_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("bob").connect(bob_comp.peer_out_port("alice"))
    alice_comp.peer_out_port("charlie").connect(charlie_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("charlie").connect(charlie_comp.peer_out_port("alice"))
    bob_comp.peer_out_port("charlie").connect(charlie_comp.peer_in_port("bob"))
    bob_comp.peer_in_port("charlie").connect(charlie_comp.peer_out_port("bob"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_peer_msg("bob", Message(0, 0, "hello bob"))
            self.send_peer_msg("charlie", Message(0, 0, "hello charlie"))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello bob"

    class CharlieNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello charlie"

    alice_intf = AliceNetstackInterface(alice_comp, ehi_network)
    bob_intf = BobNetstackInterface(bob_comp, ehi_network)
    charlie_intf = CharlieNetstackInterface(charlie_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()
    charlie_intf.start()

    ns.sim_run()


if __name__ == "__main__":
    test_no_other_nodes()
    test_one_other_node()
    test_many_other_nodes()

    test_connection()
    test_three_way_connection()
