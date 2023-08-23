from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import netsquid as ns
from netsquid.components.cchannel import ClassicalChannel
from netsquid.nodes import Node
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.message import Message
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostInterface
from qoala.util.tests import yield_from


class MockHostInterface(HostInterface):
    def __init__(self) -> None:
        self.remote: Optional[MockHostInterface] = None
        self.messages: List[Message] = []

    def send_peer_msg(self, peer: str, msg: Message) -> None:
        assert self.remote is not None
        self.remote.messages.append(msg)

    def receive_peer_msg(self, peer: str) -> Generator[EventExpression, None, Message]:
        return self.messages.pop()
        yield  # to make it behave as a generator


def setup_alice_bob() -> Tuple[ClassicalSocket, ClassicalSocket]:
    alice = MockHostInterface()
    bob = MockHostInterface()
    alice.remote = bob
    bob.remote = alice

    return ClassicalSocket(alice, "bob", 0, 0), ClassicalSocket(bob, "alice", 0, 0)


def test_send_str():
    alice, bob = setup_alice_bob()

    alice.send_str("hello")
    msg = yield_from(bob.recv())
    assert msg == Message(0, 0, "hello")


def test_send_int():
    alice, bob = setup_alice_bob()

    alice.send_int(3)
    msg = yield_from(bob.recv_int())
    assert msg == 3


def test_send_float():
    alice, bob = setup_alice_bob()

    alice.send_float(3.14)
    msg = yield_from(bob.recv_float())
    assert msg == 3.14


def test_multiple_pids():
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

    alice_intf = HostInterface(alice_comp, ehi_network)
    bob_intf = HostInterface(bob_comp, ehi_network)

    alice_csck00 = ClassicalSocket(alice_intf, "bob", pid=0, remote_pid=0)
    alice_csck11 = ClassicalSocket(alice_intf, "bob", pid=1, remote_pid=1)
    bob_csck00 = ClassicalSocket(bob_intf, "alice", pid=0, remote_pid=0)
    bob_csck11 = ClassicalSocket(bob_intf, "alice", pid=1, remote_pid=1)

    class Alice(Protocol):
        def run(self) -> Generator[EventExpression, None, None]:
            alice_csck11.send_int(42)
            alice_csck00.send_int(3)

    class Bob(Protocol):
        def run(self) -> Generator[EventExpression, None, None]:
            assert bob_intf.get_available_messages("alice") == []
            yield from bob_intf.wait_for_msg("alice")
            assert (0, 0) in bob_intf.get_available_messages("alice")
            assert (1, 1) in bob_intf.get_available_messages("alice")
            value00 = bob_csck00.read_int()
            print(value00)
            assert value00 == 3
            value11 = bob_csck11.read_int()
            print(value11)
            assert value11 == 42

    alice_intf.start()
    bob_intf.start()
    Alice().start()
    Bob().start()

    ns.sim_run()


if __name__ == "__main__":
    test_send_str()
    test_send_int()
    test_send_float()
    test_multiple_pids()
