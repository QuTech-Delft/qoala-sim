from __future__ import annotations

from typing import Dict, List, Optional

from netsquid.components import QuantumProcessor
from netsquid.components.cchannel import ClassicalChannel
from netsquid.protocols import Protocol

from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.runtime.lhi import LhiLatencies, LhiTopology
from qoala.runtime.lhi_to_ehi import LhiConverter, NtfInterface
from qoala.runtime.program import BatchInfo, ProgramBatch
from qoala.sim.host.host import Host
from qoala.sim.host.hostcomp import HostComponent
from qoala.sim.host.hostinterface import HostLatencies
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack, NetstackComponent, NetstackLatencies
from qoala.sim.process import QoalaProcess
from qoala.sim.procnodecomp import ProcNodeComponent
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos import Qnos, QnosComponent, QnosLatencies
from qoala.sim.scheduler import NodeScheduler


class ProcNode(Protocol):
    """NetSquid protocol representing a node with a software stack."""

    def __init__(
        self,
        name: str,
        qprocessor: QuantumProcessor,
        qdevice_topology: LhiTopology,
        latencies: LhiLatencies,
        ntf_interface: NtfInterface,
        network_ehi: EhiNetworkInfo,
        node: Optional[ProcNodeComponent] = None,
        node_id: Optional[int] = None,
        scheduler: Optional[NodeScheduler] = None,
        asynchronous: bool = False,
        deterministic_scheduler: bool = True,
        use_deadlines: bool = True,
        is_predictable: bool = False,
    ) -> None:
        """ProcNode constructor.

        :param name: name of this node
        :param node: an existing ProcNodeComponent object containing the static
            components or None. If None, a ProcNodeComponent is automatically
            created.
        :param qdevice_type: hardware type of the QDevice, defaults to "generic"
        :param qprocessor: NetSquid `QuantumProcessor` representing the QDevice,
            defaults to None. If None, a QuantumProcessor is created
            automatically.
        :param node_id: ID to use for the internal NetSquid node object
        :param use_default_components: whether to automatically create NetSquid
            components for the Host and QNodeOS, defaults to True. If False,
            this allows for manually creating and adding these components.
        """
        super().__init__(name=f"{name}")
        if node:
            self._node = node
        else:
            self._node = ProcNodeComponent(name, qprocessor, network_ehi, node_id)

        self._network_ehi = network_ehi
        self._ntf_interface = ntf_interface
        self._asynchronous = asynchronous

        # Create internal components.
        self._qdevice: QDevice = QDevice(self._node, qdevice_topology)
        self._local_ehi: EhiNodeInfo = LhiConverter.to_ehi(
            qdevice_topology, ntf_interface, latencies
        )

        host_latencies = HostLatencies(
            latencies.host_instr_time,
            latencies.host_peer_latency,
        )
        qnos_latencies = QnosLatencies(
            latencies.qnos_instr_time,
        )
        # TODO: decide if still needed
        netstack_latencies = NetstackLatencies(0)

        self._host = Host(
            self.host_comp,
            self.network_ehi,
            host_latencies,
            self._asynchronous,
        )
        self._memmgr = MemoryManager(self.node.name, self._qdevice, self._local_ehi)
        self._qnos = Qnos(
            self.qnos_comp,
            self.network_ehi,
            self._memmgr,
            self._qdevice,
            qnos_latencies,
            self._ntf_interface,
            self._asynchronous,
        )
        self._netstack = Netstack(
            self.netstack_comp,
            self.network_ehi,
            self._memmgr,
            self._qdevice,
            netstack_latencies,
        )

        if scheduler is None:
            self._scheduler = NodeScheduler(
                self._node.name,
                self._host,
                self._qnos,
                self._netstack,
                self._memmgr,
                self._local_ehi,
                self._network_ehi,
                deterministic_scheduler,
                use_deadlines,
                is_predictable,
            )
        else:
            self._scheduler = scheduler

    @property
    def node(self) -> ProcNodeComponent:
        return self._node

    @property
    def host_comp(self) -> HostComponent:
        return self.node.host_comp

    @property
    def qnos_comp(self) -> QnosComponent:
        return self.node.qnos_comp

    @property
    def netstack_comp(self) -> NetstackComponent:
        return self.node.netstack_comp

    @property
    def qdevice(self) -> QDevice:
        return self._qdevice

    @qdevice.setter
    def qdevice(self, qdevice) -> None:
        self._qdevice = qdevice
        self.qnos.qdevice = qdevice
        self.netstack.qdevice = qdevice

    @property
    def host(self) -> Host:
        return self._host

    @host.setter
    def host(self, host: Host) -> None:
        self._host = host

    @property
    def qnos(self) -> Qnos:
        return self._qnos

    @qnos.setter
    def qnos(self, qnos: Qnos) -> None:
        self._qnos = qnos

    @property
    def netstack(self) -> Netstack:
        return self._netstack

    @netstack.setter
    def netstack(self, netstack: Netstack) -> None:
        self._netstack = netstack

    @property
    def memmgr(self) -> MemoryManager:
        return self._memmgr

    @property
    def scheduler(self) -> NodeScheduler:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: NodeScheduler) -> None:
        self._scheduler = scheduler

    @property
    def local_ehi(self) -> EhiNodeInfo:
        return self._local_ehi

    @local_ehi.setter
    def local_ehi(self, local_ehi: EhiNodeInfo) -> None:
        self._local_ehi = local_ehi

    @property
    def network_ehi(self) -> EhiNetworkInfo:
        return self._network_ehi

    @network_ehi.setter
    def network_ehi(self, network_ehi: EhiNetworkInfo) -> None:
        self._network_ehi = network_ehi

    def connect_to(self, other: ProcNode, latency: float = 0.0) -> None:
        """Create connections between ports of this ProcNode and those of
        another ProcNode."""
        here = self.node.name
        there = other.node.name
        chan_host_ht = ClassicalChannel(f"chan_host_{here}_{there}", delay=latency)
        self.node.host_peer_out_port(there).connect(chan_host_ht.ports["send"])
        chan_host_ht.ports["recv"].connect(other.node.host_peer_in_port(here))

        chan_host_th = ClassicalChannel(f"chan_host_{there}_{here}", delay=latency)
        other.node.host_peer_out_port(here).connect(chan_host_th.ports["send"])
        chan_host_th.ports["recv"].connect(self.node.host_peer_in_port(there))

        self.node.netstack_peer_out_port(there).connect(
            other.node.netstack_peer_in_port(here)
        )
        self.node.netstack_peer_in_port(there).connect(
            other.node.netstack_peer_out_port(here)
        )

    def start(self) -> None:
        assert self._host is not None
        assert self._qnos is not None
        assert self._netstack is not None
        super().start()
        self._host.start()
        self._qnos.start()
        self._netstack.start()
        self._scheduler.start()

    def stop(self) -> None:
        assert self._host is not None
        assert self._qnos is not None
        assert self._netstack is not None
        self._scheduler.stop()
        self._netstack.stop()
        self._qnos.stop()
        self._host.stop()
        super().stop()

    def submit_batch(self, batch_info: BatchInfo) -> ProgramBatch:
        return self.scheduler.submit_batch(batch_info)

    def initialize_processes(
        self,
        remote_pids: Optional[Dict[int, List[int]]] = None,
        linear: bool = False
        # batch ID -> PID list
    ) -> None:
        self.scheduler.create_processes_for_batches(remote_pids, linear)

    def add_process(self, process: QoalaProcess) -> None:
        self.memmgr.add_process(process)

    def get_batches(self) -> Dict[int, ProgramBatch]:
        return self.scheduler.get_batches()
