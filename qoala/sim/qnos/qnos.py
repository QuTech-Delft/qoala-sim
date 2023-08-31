from __future__ import annotations

from netsquid.protocols import Protocol

from qoala.lang.ehi import EhiNetworkInfo
from qoala.runtime.ntf import GenericNtf, NtfInterface, NvNtf, TrappedIonNtf
from qoala.sim.memmgr import MemoryManager
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos.qnoscomp import QnosComponent
from qoala.sim.qnos.qnosinterface import QnosInterface, QnosLatencies
from qoala.sim.qnos.qnosprocessor import (
    GenericProcessor,
    IonTrapProcessor,
    NVProcessor,
    QnosProcessor,
)


class Qnos(Protocol):
    """NetSquid protocol representing a QNodeOS instance."""

    def __init__(
        self,
        comp: QnosComponent,
        ehi_network: EhiNetworkInfo,
        memmgr: MemoryManager,
        qdevice: QDevice,
        latencies: QnosLatencies,
        ntf_interface: NtfInterface,
        asynchronous: bool = False,
    ) -> None:
        """Qnos protocol constructor.

        :param comp: NetSquid component representing the QNodeOS instance
        :param qdevice_type: hardware type of the QDevice of this node
        """
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._comp = comp
        self._ehi_network = ehi_network

        # Owned objects.
        self._interface = QnosInterface(comp, qdevice, memmgr)
        self._processor: QnosProcessor
        self._asynchronous = asynchronous

        self.create_processor(ntf_interface, latencies)

    def create_processor(
        self, ntf_interface: NtfInterface, latencies: QnosLatencies
    ) -> None:
        if isinstance(ntf_interface, GenericNtf):
            self._processor = GenericProcessor(
                self._interface, latencies, self._asynchronous
            )
        elif isinstance(ntf_interface, NvNtf):
            self._processor = NVProcessor(
                self._interface, latencies, self._asynchronous
            )
        elif isinstance(ntf_interface, TrappedIonNtf):
            self._processor = IonTrapProcessor(
                self._interface, latencies, self._asynchronous
            )
        else:
            raise ValueError

    @property
    def qdevice(self) -> QDevice:
        return self._interface.qdevice

    @qdevice.setter
    def qdevice(self, qdevice: QDevice) -> None:
        self._interface._qdevice = qdevice

    @property
    def processor(self) -> QnosProcessor:
        return self._processor

    @processor.setter
    def processor(self, processor: QnosProcessor) -> None:
        self._processor = processor

    def start(self) -> None:
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()
