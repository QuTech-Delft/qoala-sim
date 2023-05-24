from __future__ import annotations

from netsquid.protocols import Protocol

from qoala.runtime.environment import StaticNetworkInfo
from qoala.sim.host.csocket import ClassicalSocket
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostInterface, HostLatencies
from qoala.sim.host.hostprocessor import HostProcessor


class Host(Protocol):
    """NetSquid protocol representing a Host."""

    def __init__(
        self,
        comp: HostComponent,
        static_network_info: StaticNetworkInfo,
        latencies: HostLatencies,
        asynchronous: bool = False,
    ) -> None:
        """Host protocol constructor.

        :param comp: NetSquid component representing the Host
        """
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._comp = comp
        self._static_network_info = static_network_info

        # Owned objects.
        self._interface = HostInterface(comp, static_network_info)
        self._processor = HostProcessor(self._interface, latencies, asynchronous)

    @property
    def interface(self) -> HostInterface:
        return self._interface

    @interface.setter
    def interface(self, interface: HostInterface) -> None:
        self._interface = interface
        self._processor._interface = interface

    @property
    def processor(self) -> HostProcessor:
        return self._processor

    @property
    def static_network_info(self) -> StaticNetworkInfo:
        return self._static_network_info

    def start(self) -> None:
        assert self._interface is not None
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()

    def create_csocket(self, remote_name: str) -> ClassicalSocket:
        return ClassicalSocket(self._interface, remote_name)
