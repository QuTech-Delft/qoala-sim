import logging
from typing import Dict, Generator, List

from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.runtime.message import Message
from qoala.util.logging import LogManager


class PortListener(Protocol):
    def __init__(self, port: Port, signal_label: str) -> None:
        self._buffer: List[Message] = []
        self._port: Port = port
        self._signal_label = signal_label
        self.add_signal(signal_label)

    @property
    def buffer(self) -> List[Message]:
        return self._buffer

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            # Wait for an event saying that there is new input.
            yield self.await_port_input(self._port)

            counter = 0
            # Read all inputs and count them.
            while True:
                input = self._port.rx_input()
                if input is None:
                    break
                self._buffer += input.items
                counter += 1
            # If there are n inputs, there have been n events, but we yielded only
            # on one of them so far. "Flush" these n-1 additional events:
            while counter > 1:
                yield self.await_port_input(self._port)
                counter -= 1

            # Only after having yielded on all current events, we can schedule a
            # notification event, so that its reactor can handle all inputs at once.
            self.send_signal(self._signal_label)


class ComponentProtocol(Protocol):
    def __init__(self, name: str, comp: Component) -> None:
        super().__init__(name)
        self._listeners: Dict[str, PortListener] = {}
        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({comp.name})"
        )

    def add_listener(self, name, listener: PortListener) -> None:
        self._listeners[name] = listener

    def _receive_msg(
        self, listener_name: str, wake_up_signal: str
    ) -> Generator[EventExpression, None, Message]:
        listener = self._listeners[listener_name]
        if len(listener.buffer) == 0:
            yield self.await_signal(sender=listener, signal_label=wake_up_signal)
        return listener.buffer.pop(0)

    def _receive_msg_any_source(
        self, listener_names: List[str], wake_up_signals: List[str]
    ) -> Generator[EventExpression, None, Message]:
        # TODO rewrite two separate lists as function arguments

        # First check if there is any listener with messages in their buffer.
        for listener_name, wake_up_signal in zip(listener_names, wake_up_signals):
            listener = self._listeners[listener_name]
            if len(listener.buffer) != 0:
                return listener.buffer.pop(0)

        # Else, get an EventExpression for each listener.
        expressions: List[EventExpression] = []

        for listener_name, wake_up_signal in zip(listener_names, wake_up_signals):
            listener = self._listeners[listener_name]
            assert len(listener.buffer) == 0  # already checked this above
            ev_expr = self.await_signal(sender=listener, signal_label=wake_up_signal)
            expressions.append(ev_expr)

        # Create a union of all expressoins.
        assert len(expressions) > 0
        union = expressions[0]
        for i in range(1, len(expressions)):
            union = union | expressions[i]  # type: ignore

        # Yield until at least one of the listeners got a message.
        yield union

        # Count the messages in listener's buffers.
        msg_count = 0
        msg_to_return = None
        for listener_name, wake_up_signal in zip(listener_names, wake_up_signals):
            listener = self._listeners[listener_name]

            # Count the number of new messages.
            msg_count += len(listener.buffer)

            # Only the first buffered message we encounter is actually popped
            # and returned. The others messages will get returned in a new call
            # to `_receive_msg_any_source`.
            if len(listener.buffer) != 0 and msg_to_return is None:
                msg_to_return = listener.buffer.pop(0)

        # There *must* be at least one since we yielded.
        assert msg_count > 0
        # "Flush away" the events that were also in the union.
        # We already yielded on one (above), but need to yield on the rest.
        for _ in range(msg_count - 1):
            yield union

        assert msg_to_return is not None
        return msg_to_return

    def start(self) -> None:
        super().start()
        for listener in self._listeners.values():
            listener.start()

    def stop(self) -> None:
        for listener in self._listeners.values():
            listener.stop()
        super().stop()
