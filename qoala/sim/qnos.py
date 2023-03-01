from __future__ import annotations

from netsquid.protocols import Protocol

from qoala.runtime.environment import LocalEnvironment
from qoala.runtime.lhi_to_ehi import (
    GenericToVanillaInterface,
    NativeToFlavourInterface,
    NvToNvInterface,
)
from qoala.sim.memmgr import MemoryManager
from qoala.sim.qdevice import QDevice
from qoala.sim.qnoscomp import QnosComponent
from qoala.sim.qnosinterface import QnosInterface
from qoala.sim.qnosprocessor import GenericProcessor, NVProcessor, QnosProcessor


class Qnos(Protocol):
    """NetSquid protocol representing a QNodeOS instance."""

    def __init__(
        self,
        comp: QnosComponent,
        local_env: LocalEnvironment,
        memmgr: MemoryManager,
        qdevice: QDevice,
        ntf_interface: NativeToFlavourInterface,
        asynchronous: bool = False,
    ) -> None:
        """Qnos protocol constructor.

        :param comp: NetSquid component representing the QNodeOS instance
        :param qdevice_type: hardware type of the QDevice of this node
        """
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._comp = comp
        self._local_env = local_env

        # Owned objects.
        self._interface = QnosInterface(comp, qdevice, memmgr)
        self._processor: QnosProcessor
        self._asynchronous = asynchronous

        self.create_processor(ntf_interface)

    def create_processor(self, ntf_interface: NativeToFlavourInterface) -> None:
        # TODO: rethink the way NTF interfaces are used
        if isinstance(ntf_interface, GenericToVanillaInterface):
            self._processor = GenericProcessor(self._interface, self._asynchronous)
        elif isinstance(ntf_interface, NvToNvInterface):
            self._processor = NVProcessor(self._interface, self._asynchronous)
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

    @property
    def physical_memory(self) -> PhysicalQuantumMemory:
        return self._interface.qdevice.memory

    def start(self) -> None:
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()
