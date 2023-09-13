import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from netsquid.protocols import Protocol

from qoala.lang.ehi import EhiNodeInfo, UnitModule
from qoala.sim.events import SIGNAL_MEMORY_FREED
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.util.logging import LogManager


class AllocError(Exception):
    """Allocating a qubit did not succeed."""

    pass


class NotAllocatedError(Exception):
    """A virtual qubit is not mapped to any physical qubit."""

    pass


@dataclass
class VirtualMapping:
    # mapping from virt ID in specific unit module to phys ID
    unit_module: UnitModule
    mapping: Dict[int, Optional[int]]  # virt ID -> phys ID


@dataclass
class VirtualLocation:
    # particular virt ID in particular unit module in particular process
    pid: int
    unit_module: UnitModule
    virt_id: int


class MemoryManager(Protocol):
    def __init__(
        self,
        node_name: str,
        qdevice: QDevice,
        ehi: Optional[EhiNodeInfo] = None,  # TODO refactor?
    ) -> None:
        self._node_name = node_name
        self._processes: Dict[int, QoalaProcess] = {}
        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({self._node_name})"
        )
        self._ehi = ehi

        self._qdevice = qdevice
        self._process_mappings: Dict[int, VirtualMapping] = {}  # pid -> mapping
        self._physical_mapping: Dict[int, Optional[VirtualLocation]] = {
            i: None for i in qdevice.get_all_qubit_ids()
        }  # phys ID -> virt location

        self.add_signal(SIGNAL_MEMORY_FREED)

    def _get_free_comm_phys_id(self) -> int:
        for phys_id in self._qdevice.get_comm_qubit_ids():
            if self._physical_mapping[phys_id] is None:
                return phys_id  # type: ignore
        raise AllocError

    def _get_free_mem_phys_id(self) -> int:
        for phys_id in self._qdevice.get_non_comm_qubit_ids():
            if self._physical_mapping[phys_id] is None:
                return phys_id  # type: ignore
        raise AllocError

    def get_ehi(self) -> EhiNodeInfo:
        assert self._ehi is not None  # TODO: already enforce this in constructor?
        return self._ehi

    def add_process(self, process: QoalaProcess) -> None:
        self._processes[process.pid] = process
        unit_module = process.prog_instance.unit_module
        self._process_mappings[process.pid] = VirtualMapping(
            unit_module, {x: None for x in unit_module.get_all_qubit_ids()}
        )

    def get_process(self, pid: int) -> QoalaProcess:
        return self._processes[pid]

    def get_all_program_ids(self) -> List[int]:
        return list(self._processes.keys())

    def allocate(self, pid: int, virt_id: int) -> int:
        vmap = self._process_mappings[pid]
        # Check if the virtual ID is in the unit module
        if virt_id not in vmap.unit_module.get_all_qubit_ids():
            raise AllocError

        # Check whether this virt ID is already mapped to a physical qubit.
        if vmap.mapping[virt_id] is not None:
            raise AllocError

        phys_id: int
        if vmap.unit_module.is_communication(virt_id):
            phys_id = self._get_free_comm_phys_id()
        else:
            phys_id = self._get_free_mem_phys_id()

        # update mappings
        self._physical_mapping[phys_id] = VirtualLocation(
            pid, vmap.unit_module, virt_id
        )
        self._process_mappings[pid].mapping[virt_id] = phys_id
        return phys_id

    def allocate_comm(self, pid: int, virt_id: int) -> int:
        vmap = self._process_mappings[pid]
        # Check that the virt ID is indeed a (virtual) comm qubit.
        if virt_id not in vmap.unit_module.get_all_qubit_ids():
            raise AllocError
        if not vmap.unit_module.is_communication(virt_id):
            raise AllocError

        return self.allocate(pid, virt_id)

    def free(self, pid: int, virt_id: int) -> None:
        vmap = self._process_mappings[pid]
        # Check if the virtual ID is in the unit module
        assert virt_id in vmap.unit_module.get_all_qubit_ids()
        assert virt_id in vmap.mapping

        phys_id = vmap.mapping[virt_id]
        if phys_id is None:
            raise AllocError

        # update mappings
        self._physical_mapping[phys_id] = None
        vmap.mapping[virt_id] = None

        # update netsquid memory
        self._qdevice.set_mem_pos_in_use(phys_id, False)

        # send a signal for components that may be blocked on resources
        self.send_signal(SIGNAL_MEMORY_FREED)

    def get_unmapped_non_comm_qubit(self, pid: int) -> int:
        """returns virt ID"""
        vp_map = self._process_mappings[pid].mapping
        unit_module = self._process_mappings[pid].unit_module
        free_ids = [
            v
            for v, p in vp_map.items()
            if p is None and not unit_module.is_communication(v)
        ]
        if len(free_ids) == 0:
            raise AllocError
        return min(free_ids)

    def phys_id_for(self, pid: int, virt_id: int) -> Optional[int]:
        virt_mapping = self._process_mappings[pid]
        if virt_id not in virt_mapping.mapping:
            raise RuntimeError(f"virt ID {virt_id} not in Unit Module")
        phys_id = virt_mapping.mapping[virt_id]
        return phys_id

    def virt_id_for(self, pid: int, phys_id: int) -> Optional[int]:
        if phys_id not in self._qdevice.get_all_qubit_ids():
            raise RuntimeError(f"phys ID {phys_id} not in QDevice")
        if virt_loc := self._physical_mapping[phys_id]:
            if virt_loc.pid == pid:
                return virt_loc.virt_id
        return None

    def get_all_qubit_ids(self) -> Set[int]:
        return self._qdevice.get_all_qubit_ids()
