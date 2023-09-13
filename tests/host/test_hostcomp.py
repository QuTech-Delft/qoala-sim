from __future__ import annotations

from typing import Generator

import netsquid as ns
from netsquid.components.cchannel import ClassicalChannel
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostInterface


def create_hostcomp(num_other_nodes: int) -> HostComponent:
    node = Node(name="alice", ID=0)

    nodes = {id: f"node_{id}" for id in range(1, num_other_nodes + 1)}
    nodes[0] = "alice"
    ehi_network = EhiNetworkInfo.only_nodes(nodes)

    return HostComponent(node, ehi_network)


def test_no_other_nodes():
    comp = create_hostcomp(num_other_nodes=0)

    # should have qnos_{in|out} and ntsk_{in|out} ports
    assert len(comp.ports) == 4
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "nstk_in" in comp.ports
    assert "nstk_out" in comp.ports


def test_one_other_node():
    comp = create_hostcomp(num_other_nodes=1)

    # should have 2 qnos ports + 2 nstk ports + 2 peer ports
    assert len(comp.ports) == 6
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "nstk_in" in comp.ports
    assert "nstk_out" in comp.ports

    assert "peer_node_1_in" in comp.ports
    assert "peer_node_1_out" in comp.ports

    # Test properties
    assert comp.peer_in_port("node_1") == comp.ports["peer_node_1_in"]
    assert comp.peer_out_port("node_1") == comp.ports["peer_node_1_out"]


def test_many_other_nodes():
    comp = create_hostcomp(num_other_nodes=5)

    # should have 2 qnos ports + 2 nstk ports + 5 * 2 peer ports
    assert len(comp.ports) == 14
    assert "qnos_in" in comp.ports
    assert "qnos_out" in comp.ports
    assert "nstk_in" in comp.ports
    assert "nstk_out" in comp.ports

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

    alice_comp = HostComponent(alice, ehi_network)
    bob_comp = HostComponent(bob, ehi_network)

    alice_comp.peer_out_port("bob").connect(bob_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("bob").connect(bob_comp.peer_out_port("alice"))

    class AliceHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_peer_msg("bob", Message(0, 0, "hello"))

    class BobHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello"

    alice_intf = AliceHostInterface(alice_comp, ehi_network)
    bob_intf = BobHostInterface(bob_comp, ehi_network)

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

    alice_comp = HostComponent(alice, ehi_network)
    bob_comp = HostComponent(bob, ehi_network)
    charlie_comp = HostComponent(charlie, ehi_network)

    alice_comp.peer_out_port("bob").connect(bob_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("bob").connect(bob_comp.peer_out_port("alice"))
    alice_comp.peer_out_port("charlie").connect(charlie_comp.peer_in_port("alice"))
    alice_comp.peer_in_port("charlie").connect(charlie_comp.peer_out_port("alice"))
    bob_comp.peer_out_port("charlie").connect(charlie_comp.peer_in_port("bob"))
    bob_comp.peer_in_port("charlie").connect(charlie_comp.peer_out_port("bob"))

    class AliceHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_peer_msg("bob", Message(0, 0, "hello bob"))
            self.send_peer_msg("charlie", Message(0, 0, "hello charlie"))

    class BobHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello bob"

    class CharlieHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            msg = yield from self.receive_peer_msg("alice")
            assert msg.content == "hello charlie"

    alice_intf = AliceHostInterface(alice_comp, ehi_network)
    bob_intf = BobHostInterface(bob_comp, ehi_network)
    charlie_intf = CharlieHostInterface(charlie_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()
    charlie_intf.start()

    ns.sim_run()


def test_connection_with_channel():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    ehi_network = EhiNetworkInfo.only_nodes({alice.ID: alice.name, bob.ID: bob.name})

    alice_comp = HostComponent(alice, ehi_network)
    bob_comp = HostComponent(bob, ehi_network)

    channel_ab = ClassicalChannel("chan_ab", delay=1000)
    channel_ba = ClassicalChannel("chan_ba", delay=1000)

    alice_comp.peer_out_port("bob").connect(channel_ab.ports["send"])
    channel_ab.ports["recv"].connect(bob_comp.peer_in_port("alice"))

    bob_comp.peer_out_port("alice").connect(channel_ba.ports["send"])
    channel_ba.ports["recv"].connect(alice_comp.peer_in_port("bob"))

    class AliceHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_peer_msg("bob", Message(0, 0, "hello"))

    class BobHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            assert ns.sim_time() == 0
            msg = yield from self.receive_peer_msg("alice")
            assert msg == Message(0, 0, "hello")
            assert ns.sim_time() == 1000

    alice_intf = AliceHostInterface(alice_comp, ehi_network)
    bob_intf = BobHostInterface(bob_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()

    ns.sim_run()


def test_connection_with_pids():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    ehi_network = EhiNetworkInfo.only_nodes({alice.ID: alice.name, bob.ID: bob.name})

    alice_comp = HostComponent(alice, ehi_network)
    bob_comp = HostComponent(bob, ehi_network)

    channel_ab = ClassicalChannel("chan_ab", delay=1000)
    channel_ba = ClassicalChannel("chan_ba", delay=1000)

    alice_comp.peer_out_port("bob").connect(channel_ab.ports["send"])
    channel_ab.ports["recv"].connect(bob_comp.peer_in_port("alice"))

    bob_comp.peer_out_port("alice").connect(channel_ba.ports["send"])
    channel_ba.ports["recv"].connect(alice_comp.peer_in_port("bob"))

    class AliceHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            # Send message from PID 0 (alice) to PID 0 (bob)
            self.send_peer_msg("bob", Message(0, 0, "hello (0, 0)"))

            yield from self.wait(500)

            # Send message from PID 2 (alice) to PID 1 (bob)
            self.send_peer_msg("bob", Message(2, 1, "hello (2, 1)"))

    class BobHostInterface(HostInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            assert len(self.get_available_messages("alice")) == 0
            assert ns.sim_time() == 0
            yield from self.wait_for_msg("alice")
            assert ns.sim_time() == 1000
            assert self.get_available_messages("alice") == [(0, 0)]
            msg00 = self.pop_msg("alice", 0, 0)
            assert msg00 == Message(0, 0, "hello (0, 0)")

            yield from self.wait_for_msg("alice")
            assert ns.sim_time() == 1500  # 1000 + 500
            assert self.get_available_messages("alice") == [(2, 1)]
            msg21 = self.pop_msg("alice", 2, 1)
            print(f"{self.name}: received msg (0, 0) with content: {msg00.content}")
            print(f"{self.name}: received msg (2, 1) with content: {msg21.content}")
            assert msg21 == Message(2, 1, "hello (2, 1)")

    alice_intf = AliceHostInterface(alice_comp, ehi_network)
    bob_intf = BobHostInterface(bob_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()

    ns.sim_run()


if __name__ == "__main__":
    test_no_other_nodes()
    test_one_other_node()
    test_many_other_nodes()

    test_connection()
    test_three_way_connection()
    test_connection_with_channel()
    test_connection_with_pids()
