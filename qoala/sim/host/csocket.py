from __future__ import annotations

from typing import Generator

from pydynaa import EventExpression
from qoala.runtime.message import Message
from qoala.sim.host.hostinterface import HostInterface


class ClassicalSocket:
    """Wrapper around classical ports"""

    def __init__(
        self, host: HostInterface, remote_name: str, pid: int, remote_pid: int
    ):
        self._host = host
        self._remote_name = remote_name
        self._pid = pid
        self._remote_pid = remote_pid

    @property
    def remote_name(self) -> str:
        return self._remote_name

    @property
    def remote_pid(self) -> int:
        return self._remote_pid

    def send(self, msg: Message) -> None:
        """Send a message to the remote node."""
        self._host.send_peer_msg(self._remote_name, msg)

    def recv(self) -> Generator[EventExpression, None, Message]:
        msg = yield from self._host.receive_peer_msg(self._remote_name)
        return msg

    def read(self) -> Message:
        return self._host.pop_msg(self._remote_name, self._pid, self._remote_pid)

    def send_str(self, msg: str) -> None:
        self.send(Message(src_pid=self._pid, dst_pid=self._remote_pid, content=msg))

    def recv_str(self) -> Generator[EventExpression, None, str]:
        msg = yield from self.recv()
        assert isinstance(msg.content, str)
        return msg.content

    def read_str(self) -> str:
        msg = self.read()
        assert isinstance(msg.content, str)
        return msg.content

    def send_int(self, value: int) -> None:
        self.send(Message(src_pid=self._pid, dst_pid=self._remote_pid, content=value))

    def recv_int(self) -> Generator[EventExpression, None, int]:
        msg = yield from self.recv()
        assert isinstance(msg.content, int)
        return msg.content

    def read_int(self) -> int:
        msg = self.read()
        assert isinstance(msg.content, int)
        return msg.content

    def send_float(self, value: float) -> None:
        self.send(Message(src_pid=self._pid, dst_pid=self._remote_pid, content=value))

    def recv_float(self) -> Generator[EventExpression, None, float]:
        msg = yield from self.recv()
        assert isinstance(msg.content, float)
        return msg.content

    def read_float(self) -> float:
        msg = self.read()
        assert isinstance(msg.content, float)
        return msg.content
