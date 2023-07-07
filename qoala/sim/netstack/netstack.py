from __future__ import annotations

from netsquid.protocols import Protocol

from qoala.lang.ehi import EhiNetworkInfo
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack.netstackcomp import NetstackComponent
from qoala.sim.netstack.netstackinterface import NetstackInterface, NetstackLatencies
from qoala.sim.netstack.netstackprocessor import NetstackProcessor
from qoala.sim.qdevice import QDevice


class Netstack(Protocol):
    """NetSquid protocol representing the QNodeOS network stack."""

    def __init__(
        self,
        comp: NetstackComponent,
        ehi_network: EhiNetworkInfo,
        memmgr: MemoryManager,
        qdevice: QDevice,
        latencies: NetstackLatencies,
    ) -> None:
        """Network stack protocol constructor. Typically created indirectly through
        constructing a `Qnos` instance.

        :param comp: NetSquid component representing the network stack
        :param qnos: `Qnos` protocol that owns this protocol
        """
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._comp = comp
        self._ehi_network = ehi_network

        # Owned objects.
        self._interface = NetstackInterface(comp, ehi_network, qdevice, memmgr)
        self._processor = NetstackProcessor(self._interface, latencies)

    @property
    def qdevice(self) -> QDevice:
        return self._interface.qdevice

    @qdevice.setter
    def qdevice(self, qdevice: QDevice) -> None:
        self._interface._qdevice = qdevice

    @property
    def interface(self) -> NetstackInterface:
        return self._interface

    @interface.setter
    def interface(self, interface: NetstackInterface) -> None:
        self._interface = interface
        self._processor._interface = interface

    @property
    def processor(self) -> NetstackProcessor:
        return self._processor

    @processor.setter
    def processor(self, processor: NetstackProcessor) -> None:
        self._processor = processor

    def start(self) -> None:
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()
