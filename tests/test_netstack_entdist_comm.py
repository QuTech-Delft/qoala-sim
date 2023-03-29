from __future__ import annotations

from typing import Generator, List

import netsquid as ns
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.runtime.environment import LocalEnvironment, NetworkEhi
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.netstack.netstackinterface import NetstackInterface


class MockNetstackInterface(NetstackInterface):
    def __init__(self, comp: NetstackComponent, local_env: LocalEnvironment) -> None:
        super().__init__(comp, local_env, None, None, None)


def test_connection():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    env = NetworkEhi()

    env.add_node(alice.ID, alice.name)
    env.add_node(bob.ID, bob.name)

    alice_comp = NetstackComponent(alice, env)
    bob_comp = NetstackComponent(bob, env)
    entdist_comp = EntDistComponent(env)

    # Connect both nodes to the Entdist.
    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg("hello this is Alice")

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg("hello this is Bob")

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, env: NetworkEhi) -> None:
            super().__init__(comp, env)
            self.msg_alice = None
            self.msg_bob = None

        def run(self) -> Generator[EventExpression, None, None]:
            self.msg_alice = yield from self.receive_node_msg("alice")
            self.msg_bob = yield from self.receive_node_msg("bob")

    alice_intf = AliceNetstackInterface(alice_comp, LocalEnvironment(env, alice.ID))
    bob_intf = BobNetstackInterface(bob_comp, LocalEnvironment(env, bob.ID))
    entdist_intf = TestEntDistInterface(entdist_comp, env)

    alice_intf.start()
    bob_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.msg_alice == "hello this is Alice"
    assert entdist_intf.msg_bob == "hello this is Bob"


def test_wait_for_any_node():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    env = NetworkEhi()

    env.add_node(alice.ID, alice.name)
    env.add_node(bob.ID, bob.name)

    alice_comp = NetstackComponent(alice, env)
    bob_comp = NetstackComponent(bob, env)
    entdist_comp = EntDistComponent(env)

    # Connect both nodes to the Entdist.
    alice_comp.entdist_out_port.connect(entdist_comp.node_in_port("alice"))
    alice_comp.entdist_in_port.connect(entdist_comp.node_out_port("alice"))
    bob_comp.entdist_out_port.connect(entdist_comp.node_in_port("bob"))
    bob_comp.entdist_in_port.connect(entdist_comp.node_out_port("bob"))

    class AliceNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg("hello this is Alice")
            yield from self.wait(2000)
            self.send_entdist_msg("hello again from Alice")

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg("hello this is Bob")
            yield from self.wait(1000)
            self.send_entdist_msg("hello again from Bob")

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, env: NetworkEhi) -> None:
            super().__init__(comp, env)
            self.messages: List[str] = []

        def run(self) -> Generator[EventExpression, None, None]:
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))
            self.messages.append((yield from self.receive_msg()))

    alice_intf = AliceNetstackInterface(alice_comp, LocalEnvironment(env, alice.ID))
    bob_intf = BobNetstackInterface(bob_comp, LocalEnvironment(env, bob.ID))
    entdist_intf = TestEntDistInterface(entdist_comp, env)

    alice_intf.start()
    bob_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.messages == [
        "hello this is Alice",
        "hello this is Bob",
        "hello again from Bob",
        "hello again from Alice",
    ]


def test_wait_for_any_node_2():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    charlie = Node(name="charlie", ID=2)
    env = NetworkEhi()

    env.add_node(alice.ID, alice.name)
    env.add_node(bob.ID, bob.name)
    env.add_node(charlie.ID, charlie.name)

    alice_comp = NetstackComponent(alice, env)
    bob_comp = NetstackComponent(bob, env)
    charlie_comp = NetstackComponent(charlie, env)
    entdist_comp = EntDistComponent(env)

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
            self.send_entdist_msg("hello this is Alice at time 500")
            yield from self.wait(2000)
            self.send_entdist_msg("hello again from Alice at time 2500")

    class BobNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            self.send_entdist_msg("hello this is Bob at time 0")
            yield from self.wait(1000)
            self.send_entdist_msg("hello again from Bob at time 1000")

    class CharlieNetstackInterface(MockNetstackInterface):
        def run(self) -> Generator[EventExpression, None, None]:
            yield from self.wait(300)
            self.send_entdist_msg("hello this is Charlie at time 300")
            yield from self.wait(1500)
            self.send_entdist_msg("hello again from Charlie at time 1800")

    class TestEntDistInterface(EntDistInterface):
        def __init__(self, comp: EntDistComponent, env: NetworkEhi) -> None:
            super().__init__(comp, env)
            self.messages: List[str] = []

        def run(self) -> Generator[EventExpression, None, None]:
            for _ in range(6):
                self.messages.append((yield from self.receive_msg()))

    alice_intf = AliceNetstackInterface(alice_comp, LocalEnvironment(env, alice.ID))
    bob_intf = BobNetstackInterface(bob_comp, LocalEnvironment(env, bob.ID))
    charlie_intf = CharlieNetstackInterface(
        charlie_comp, LocalEnvironment(env, charlie.ID)
    )
    entdist_intf = TestEntDistInterface(entdist_comp, env)

    alice_intf.start()
    bob_intf.start()
    charlie_intf.start()
    entdist_intf.start()

    ns.sim_run()

    assert entdist_intf.messages == [
        "hello this is Bob at time 0",
        "hello this is Charlie at time 300",
        "hello this is Alice at time 500",
        "hello again from Bob at time 1000",
        "hello again from Charlie at time 1800",
        "hello again from Alice at time 2500",
    ]


if __name__ == "__main__":
    test_connection()
    test_wait_for_any_node()
    test_wait_for_any_node_2()
