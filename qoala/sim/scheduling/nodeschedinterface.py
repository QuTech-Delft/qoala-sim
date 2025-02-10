from __future__ import annotations

from typing import Generator, List, Optional

from pydynaa import EventExpression
from qoala.runtime.message import Message
from qoala.sim.componentprot import ComponentProtocol, PortListener
from qoala.sim.events import SIGNAL_CPU_NODE_SCH_MSG, SIGNAL_QPU_NODE_SCH_MSG
from qoala.sim.scheduling.nodeschedcomp import NodeSchedulerComponent


class NodeSchedulerInterface(ComponentProtocol):
    """Interface for handling messages to and from the Node Scheduler."""

    def __init__(self, comp: NodeSchedulerComponent) -> None:
        super().__init__(name=f"{comp.name}_protocol", comp=comp)
        self._comp = comp

        self.add_listener(
            "cpu",
            PortListener(self._comp.cpu_scheduler_in_port, SIGNAL_CPU_NODE_SCH_MSG),
        )
        self.add_listener(
            "qpu",
            PortListener(self._comp.qpu_scheduler_in_port, SIGNAL_QPU_NODE_SCH_MSG),
        )

    def wait_for_any_msg(self) -> Generator[EventExpression, None, None]:
        yield from self._wait_for_msg_any_source(
            ["cpu", "qpu"], [SIGNAL_CPU_NODE_SCH_MSG, SIGNAL_QPU_NODE_SCH_MSG]
        )

    def pop_available_messages(self, peer: str) -> List[Message]:
        listener = self._listeners[peer]
        return listener.buffer.pop_all()

    def get_evexpr_for_any_msg(self) -> Optional[EventExpression]:
        return self._get_evexpr_for_any_msg(
            ["cpu", "qpu"], [SIGNAL_CPU_NODE_SCH_MSG, SIGNAL_QPU_NODE_SCH_MSG]
        )
