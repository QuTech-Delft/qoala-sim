from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Type

from qoala.lang.target import ExposedHardwareInfo


class QubitTrait:
    pass


class CommQubitTrait(QubitTrait):
    pass


class MemQubitTrait(QubitTrait):
    pass


class DecorenceQubitTrait(QubitTrait):
    def __init__(self, t1: int, t2: int) -> None:
        self._t1, self._t2 = t1, t2


class GateTrait:
    pass


class SingleGateTrait(GateTrait):
    def __init__(self, depolarizing_factor: float) -> None:
        self._depolarizing_factor = depolarizing_factor


class TwoGateTrait(GateTrait):
    def __init__(self, depolarizing_factor: float) -> None:
        self._depolarizing_factor = depolarizing_factor


@dataclass(eq=True, frozen=True)
class QubitInfo:
    traits: List[QubitTrait]


@dataclass(eq=True, frozen=True)
class GateInfo:
    traits: List[GateTrait]


@dataclass(eq=True, frozen=True)
class Topology:
    qubits: Dict[int, QubitInfo]  # qubit ID -> info
    gates: Dict[List[int], GateInfo]  # ordered list of qubit IDs -> info

    def to_ehi(self) -> ExposedHardwareInfo:
        pass


class GateSet:
    pass


class TopologyBuilder:
    """Convenience methods for creating a Topology object."""

    @classmethod
    def build_star(cls, num_mem_qubits: int) -> Topology:
        comm_info = QubitInfo(traits=[CommQubitTrait, MemQubitTrait])
        mem_info = QubitInfo(traits=[MemQubitTrait])
        qubits = {i: mem_info for i in range(1, num_mem_qubits + 1)}
        qubits[0] = comm_info
        return Topology(qubits=qubits)


@dataclass
class StarTopology:
    comm_ids: Set[int]
    mem_ids: Set[int]


@dataclass(eq=True, frozen=True)
class UnitModule:
    """
    :param qubit_ids: list of qubit IDs
    :param qubit_traits: map from qubit ID to list of qubit traits
    :param gate_traits: map from list of qubit IDs to list of gate traits
    """

    qubit_ids: List[int]
    qubit_traits: Dict[int, List[Type[QubitTrait]]]
    gate_traits: Dict[List[int], List[GateTrait]]

    @property
    def num_qubits(self) -> int:
        return len(self.qubit_ids)

    @classmethod
    def from_topology(cls, topology: Topology) -> UnitModule:
        all_ids = topology.comm_ids.union(topology.mem_ids)
        traits: Dict[int, List[Type[QubitTrait]]] = {i: [] for i in all_ids}
        for i in all_ids:
            if i in topology.comm_ids:
                traits[i].append(CommQubitTrait)
            if i in topology.mem_ids:
                traits[i].append(MemQubitTrait)
        return UnitModule(qubit_ids=list(all_ids), qubit_traits=traits, gate_traits={})

    @classmethod
    def default_generic(cls, num_qubits: int) -> UnitModule:
        return UnitModule(
            qubit_ids=[i for i in range(num_qubits)],
            qubit_traits={i: [CommQubitTrait, MemQubitTrait] for i in range(num_qubits)},  # type: ignore
            gate_traits={},
        )


class QuantumMemory:
    """Quantum memory only available to Qnos. Represented as unit modules."""

    # Only describes the virtual memory space (i.e. unit module).
    # Does not contain 'values' (quantum states) of the virtual memory locations.
    # Does not contain the mapping from virtual to physical space. (Managed by memmgr)

    def __init__(self, pid: int, unit_module: UnitModule) -> None:
        self._pid = pid
        self._unit_module = unit_module

    @property
    def unit_module(self) -> UnitModule:
        return self._unit_module
