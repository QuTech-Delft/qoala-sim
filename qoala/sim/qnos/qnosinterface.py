from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from pydynaa import EventExpression
from qoala.runtime.message import Message
from qoala.sim.componentprot import ComponentProtocol, PortListener
from qoala.sim.events import (
    EVENT_WAIT,
    SIGNAL_HOST_QNOS_MSG,
    SIGNAL_MEMORY_FREED,
    SIGNAL_NSTK_QNOS_MSG,
)
from qoala.sim.memmgr import MemoryManager
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos.qnoscomp import QnosComponent


@dataclass
class QnosLatencies:
    qnos_instr_time: float = 0  # duration of classical Qnos instr execution

    @classmethod
    def all_zero(cls) -> QnosLatencies:
        # NOTE: can also just use QnosLatencies() which will default all values to 0
        # However, using this classmethod makes this behavior more explicit and clear.
        return QnosLatencies(0)


class QnosInterface(ComponentProtocol):
    """NetSquid protocol representing a QNodeOS processor."""

    def __init__(
        self, comp: QnosComponent, qdevice: QDevice, memmgr: MemoryManager
    ) -> None:
        """Processor protocol constructor. Typically created indirectly through
        constructing a `Qnos` instance.

        :param comp: NetSquid component representing the processor
        :param qnos: `Qnos` protocol that owns this protocol
        """
        super().__init__(name=f"{comp.name}_protocol", comp=comp)
        self._comp = comp
        self._qdevice = qdevice
        self._memmgr = memmgr

        self.add_listener(
            "host",
            PortListener(self._comp.ports["host_in"], SIGNAL_HOST_QNOS_MSG),
        )
        self.add_listener(
            "netstack",
            PortListener(self._comp.ports["nstk_in"], SIGNAL_NSTK_QNOS_MSG),
        )

        self.add_signal(SIGNAL_MEMORY_FREED)

    def signal_memory_freed(self) -> None:
        self._comp.netstack_mem_out_port.tx_output(Message(0, 0, content=None))

    @property
    def qdevice(self) -> QDevice:
        return self._qdevice

    @property
    def memmgr(self) -> MemoryManager:
        return self._memmgr

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr
