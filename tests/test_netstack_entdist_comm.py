from __future__ import annotations

from typing import Generator

import netsquid as ns
from netsquid.nodes import Node

from pydynaa import EventExpression
from qoala.runtime.environment import (
    GlobalEnvironment,
    GlobalNodeInfo,
    LocalEnvironment,
)
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.netstack.netstackinterface import NetstackInterface


class MockNetstackInterface(NetstackInterface):
    def __init__(self, comp: NetstackComponent, local_env: LocalEnvironment) -> None:
        super().__init__(comp, local_env, None, None, None)


def test_connection():
    ns.sim_reset()

    alice = Node(name="alice", ID=0)
    bob = Node(name="bob", ID=1)
    env = GlobalEnvironment()

    alice_info = GlobalNodeInfo(alice.name, alice.ID)
    env.add_node(alice.ID, alice_info)
    bob_info = GlobalNodeInfo(bob.name, bob.ID)
    env.add_node(bob.ID, bob_info)

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
        def __init__(self, comp: EntDistComponent, env: GlobalEnvironment) -> None:
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


if __name__ == "__main__":
    test_connection()
