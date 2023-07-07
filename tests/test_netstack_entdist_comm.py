from __future__ import annotations

from typing import Generator, List

import netsquid as ns
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.netstack.netstackinterface import NetstackInterface


class MockNetstackInterface(NetstackInterface):
    def __init__(self, comp: NetstackComponent, ehi_network: EhiNetworkInfo) -> None:
        super().__init__(comp, ehi_network, None, None)

    def send_entdist_str(self, msg: str) -> None:
        self.send_entdist_msg(Message(0, 0, msg))


def test_connection():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    ehi_network = EhiNetworkInfo.only_nodes({alice.ID: alice.name, bob.ID: bob.name})

    alice_comp = NetstackComponent(alice, ehi_network)
    bob_comp = NetstackComponent(bob, ehi_network)
    entdist_comp = EntDistComponent(ehi_network)

    # Connect both nodes to the Entdist.
    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg(Message(0, 0, "hello this is Alice"))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg(Message(0, 0, "hello this is Bob"))

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, ehi_network: EhiNetworkInfo) -> None:
            super().__init__(comp, ehi_network)
            self.msg_alice = None
            self.msg_bob = None

        def run(self) -> Generator[EventExpression, None, None]:
            self.msg_alice = yield from self.receive_node_msg("alice")
            self.msg_bob = yield from self.receive_node_msg("bob")

    alice_intf = AliceNetstackInterface(alice_comp, ehi_network)
    bob_intf = BobNetstackInterface(bob_comp, ehi_network)
    entdist_intf = TestEntDistInterface(entdist_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.msg_alice.content == "hello this is Alice"
    assert entdist_intf.msg_bob.content == "hello this is Bob"


def test_wait_for_any_node():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    ehi_network = EhiNetworkInfo.only_nodes({alice.ID: alice.name, bob.ID: bob.name})

    alice_comp = NetstackComponent(alice, ehi_network)
    bob_comp = NetstackComponent(bob, ehi_network)
    entdist_comp = EntDistComponent(ehi_network)

    # Connect both nodes to the Entdist.
    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg(Message(0, 0, "hello this is Alice"))
            yield from self.wait(2000)
            self.send_entdist_msg(Message(0, 0, "hello again from Alice"))

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg(Message(0, 0, "hello this is Bob"))
            yield from self.wait(1000)
            self.send_entdist_msg(Message(0, 0, "hello again from Bob"))

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, ehi_network: EhiNetworkInfo) -> None:
            super().__init__(comp, ehi_network)
            self.messages: List[str] = []

        def run(self) -> Generator[EventExpression, None, None]:
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))

    alice_intf = AliceNetstackInterface(alice_comp, ehi_network)
    bob_intf = BobNetstackInterface(bob_comp, ehi_network)
    entdist_intf = TestEntDistInterface(entdist_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.messages == [
        Message(0, 0, "hello this is Alice"),
        Message(0, 0, "hello this is Bob"),
        Message(0, 0, "hello again from Bob"),
        Message(0, 0, "hello again from Alice"),
    ]


def test_wait_for_any_node_2():
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
    entdist_comp = EntDistComponent(ehi_network)

    # Connect all nodes to the Entdist.
    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))
    charlie_comp.entdist_out_port.connect(entdist_comp.node_in_port("charlie"))
    charlie_comp.entdist_in_port.connect(entdist_comp.node_out_port("charlie"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.wait(500)
            self.send_entdist_str("hello this is Alice at time 500")
            yield from self.wait(2000)
            self.send_entdist_str("hello again from Alice at time 2500")

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_str("hello this is Bob at time 0")
            yield from self.wait(1000)
            self.send_entdist_str("hello again from Bob at time 1000")

    class CharlieNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.wait(300)
            self.send_entdist_str("hello this is Charlie at time 300")
            yield from self.wait(1500)
            self.send_entdist_str("hello again from Charlie at time 1800")

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, ehi_network: EhiNetworkInfo) -> None:
            super().__init__(comp, ehi_network)
            self.messages: List[str] = []

        def run(self) -> Generator[EventExpression, None, None]:
            for _ in range(6):
                self.messages.append((yield from self.receive_msg()))

    alice_intf = AliceNetstackInterface(alice_comp, ehi_network)
    bob_intf = BobNetstackInterface(bob_comp, ehi_network)
    charlie_intf = CharlieNetstackInterface(charlie_comp, ehi_network)
    entdist_intf = TestEntDistInterface(entdist_comp, ehi_network)

    alice_intf.start()
    bob_intf.start()
    charlie_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.messages == [
        Message(0, 0, "hello this is Bob at time 0"),
        Message(0, 0, "hello this is Charlie at time 300"),
        Message(0, 0, "hello this is Alice at time 500"),
        Message(0, 0, "hello again from Bob at time 1000"),
        Message(0, 0, "hello again from Charlie at time 1800"),
        Message(0, 0, "hello again from Alice at time 2500"),
    ]


if __name__ == "__main__":
    test_connection()
    test_wait_for_any_node()
    test_wait_for_any_node_2()
