import logging
from typing import Dict, Generator, List, Optional, Tuple

from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.runtime.message import Message
from qoala.util.logging import LogManager


class MessageBuffer:
    def __init__(self) -> None:
        # Added logger for debugging
        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}(EntDist)"
        )
        self._messages: Dict[
            Tuple[int, int], List[Message]
        ] = {}  # (src PID, dst PID) -> message list

    def add_msg(self, msg: Message) -> None:
        if (msg.src_pid, msg.dst_pid) not in self._messages:
            self._messages[(msg.src_pid, msg.dst_pid)] = [msg]
        else:
            self._messages[(msg.src_pid, msg.dst_pid)].append(msg)

    def has_msg(self, src_pid: int, dst_pid: int) -> bool:
        if (src_pid, dst_pid) not in self._messages:
            return False
        return len(self._messages[(src_pid, dst_pid)]) > 0

    def has_any(self) -> bool:
        if len(self._messages) == 0:
            return False
        return any(len(buf) > 0 for buf in self._messages.values())

    def get_all(self) -> List[Tuple[int, int]]:
        # Does *NOT* pop messages.
        return [
            (src, dst) for ((src, dst), buf) in self._messages.items() if len(buf) > 0
        ]

    def count_all(self) -> int:
        return sum(len(buf) for buf in self._messages.values())

    def pop_msg(self, src_pid: int, dst_pid: int) -> Message:
        return self._messages[(src_pid, dst_pid)].pop(0)

    def pop_any(self) -> Message:
        for buf in self._messages.values():
            if len(buf) > 0:
                return buf.pop(0)
        raise RuntimeError

    def pop_all(self) -> List[Message]:
        messages = []
        self._logger.info(f"{self._messages}")
        for _, buf in self._messages.items():
            self._logger.info(f"BEFORE POP: {_} {len(buf)}, {buf}")
            messages.extend(buf)
            buf.clear()
            self._logger.info(f"AFTER POP: {_} {len(buf)} {buf}")

        return messages


class PortListener(Protocol):
    def __init__(self, port: Port, signal_label: str) -> None:
        self._buffer = MessageBuffer()
        self._port = port
        self._signal_label = signal_label
        self.add_signal(signal_label)

    @property
    def buffer(self) -> MessageBuffer:
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
                for item in input.items:
                    self._buffer.add_msg(item)
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

    def _wait_for_msg(
        self, listener_name: str, wake_up_signal: str
    ) -> Generator[EventExpression, None, None]:
        listener = self._listeners[listener_name]
        if not listener.buffer.has_any():
            yield self.await_signal(sender=listener, signal_label=wake_up_signal)

    def _pop_any_msg(self, listener_name: str) -> Message:
        listener = self._listeners[listener_name]
        self._logger.info(f"MESSAGES IN THE QUEUE {listener.buffer._messages}")
        return listener.buffer.pop_any()

    def _pop_msg(self, listener_name: str, src_pid: int, dst_pid: int) -> Message:
        listener = self._listeners[listener_name]
        return listener.buffer.pop_msg(src_pid, dst_pid)

    def _has_msg(self, listener_name: str, src_pid: int, dst_pid: int) -> bool:
        listener = self._listeners[listener_name]
        return listener.buffer.has_msg(src_pid, dst_pid)

    def _pop_any_msg_any_source(self, listener_names: List[str]) -> Message:
        for listener_name in listener_names:
            listener = self._listeners[listener_name]
            if listener.buffer.has_any():
                return listener.buffer.pop_any()
        raise RuntimeError

    def _pop_all_messages(self, listener_names: List[str]) -> List[Message]:
        messages = []
        for listener_name in listener_names:
            listener = self._listeners[listener_name]
            if listener.buffer.has_any():
                messages.extend(listener.buffer.pop_all())
        return messages

    def _get_evexpr_for_any_msg(
        self, listener_names: List[str], wake_up_signals: List[str]
    ) -> Optional[EventExpression]:
        # Returns None if there are already messages and no event expression is needed.

        # TODO rewrite two separate lists as function arguments

        # First check if there is any listener with messages in their buffer.
        for listener_name, wake_up_signal in zip(listener_names, wake_up_signals):
            listener = self._listeners[listener_name]
            if listener.buffer.has_any():
                return None

        # Else, get an EventExpression for each listener.
        expressions: List[EventExpression] = []

        for listener_name, wake_up_signal in zip(listener_names, wake_up_signals):
            listener = self._listeners[listener_name]
            assert not listener.buffer.has_any()  # already checked this above
            ev_expr = self.await_signal(sender=listener, signal_label=wake_up_signal)
            expressions.append(ev_expr)

        # Create a union of all expressions.
        assert len(expressions) > 0
        union = expressions[0]
        for i in range(1, len(expressions)):
            union = union | expressions[i]  # type: ignore
        self._logger.info(f"SOMETHING IS HAPPENEING HERE? {union}")
        return union

    def _handle_msg_evexpr(
        self, evexpr: EventExpression, listener_names: List[str]
    ) -> Generator[EventExpression, None, None]:
        # Count the number of listeners that has messages in their buffers.
        # This number is equal to the number of events that have fired.

        ev_count = 0
        for listener_name in listener_names:
            listener = self._listeners[listener_name]

            if listener.buffer.has_any():
                ev_count += 1
                self._logger.info(f"NODE {listener_name} HAS A MESSAGE")

        # There *must* be at least one event that has fired since we yielded.
        assert ev_count > 0
        # "Flush away" the events that were also in the union.
        # We already yielded on one (above), but need to yield on the rest.
        for _ in range(ev_count - 1):
            yield evexpr
            self._logger.info("ANOTHER NODE HAS MESSAGES?")

        self._logger.info(f"RECEIVED MESSAGES FROM {ev_count} NODES IN TOTAL")

    def _wait_for_msg_any_source(
        self, listener_names: List[str], wake_up_signals: List[str]
    ) -> Generator[EventExpression, None, None]:
        # listener names are the node names
        # wake up signals are the type of messages that cause the listener to wake up

        # Get an event expression for any one of the listeners getting a message.
        evexpr = self._get_evexpr_for_any_msg(listener_names, wake_up_signals)

        # If None, there are already messages available.
        # this causes wait_for_any_msg() to stop yielding, which causes run_with_netschedule to stop yielding
        if evexpr is None:
            return
        self._logger.info("WAITING FOR A MESSAGE")
        # Wait until at least one of the events in the evexpr happens.
        # (i.e. wait until at least one message arrives)
        yield evexpr
        self._logger.info("A SINGLE NODE HAS A MESSAGE ARRIVED")

        yield from self._handle_msg_evexpr(evexpr, listener_names)

    def start(self) -> None:
        super().start()
        for listener in self._listeners.values():
            listener.start()

    def stop(self) -> None:
        for listener in self._listeners.values():
            listener.stop()
        super().stop()
