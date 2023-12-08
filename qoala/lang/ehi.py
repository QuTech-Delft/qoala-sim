from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, FrozenSet, List, Optional, Tuple, Type

import numpy as np
from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour

from qoala.lang.common import MultiQubit


@dataclass(frozen=True)
class EhiQubitInfo:
    is_communication: bool
    decoherence_rate: float  # rate per second


@dataclass(frozen=True)
class EhiGateInfo:
    instruction: Type[NetQASMInstruction]
    duration: float  # ns
    decoherence: float  # rate per second, for all qubits


@dataclass(frozen=True)
class EhiLinkInfo:
    duration: float  # ns
    fidelity: float


@dataclass(frozen=True)
class EhiLatencies:
    host_instr_time: float  # duration of classical Host instr execution (CL)
    qnos_instr_time: float  # duration of classical Qnos instr execution (QL)
    host_peer_latency: float  # processing time for Host messages from remote node (CC)
    internal_sched_latency: float  # processing time for messaging between node scheduler and processor schedulers

    @classmethod
    def all_zero(cls) -> EhiLatencies:
        return EhiLatencies(0, 0, 0, 0)


@dataclass(frozen=True)
class EhiNodeInfo:
    """Hardware made available to offline compiler."""

    qubit_infos: Dict[int, EhiQubitInfo]  # qubit ID -> info

    flavour: Type[
        Flavour
    ]  # set of NetQASM instrs, no info about which qubits can do what instr
    single_gate_infos: Dict[int, List[EhiGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[EhiGateInfo]
    ]  # ordered qubit ID list -> gates
    latencies: EhiLatencies
    all_qubit_gate_infos: Optional[
        List[EhiGateInfo]
    ] = None  # gates that are applied to all qubits

    def find_single_gate(
            self, qubit_id: int, instr: Type[NetQASMInstruction]
    ) -> Optional[EhiGateInfo]:
        if qubit_id not in self.single_gate_infos:
            return None
        for info in self.single_gate_infos[qubit_id]:
            if info.instruction == instr:
                return info
        return None

    def find_multi_gate(
            self, qubit_ids: List[int], instr: Type[NetQASMInstruction]
    ) -> Optional[EhiGateInfo]:
        multi = MultiQubit(qubit_ids)
        if multi not in self.multi_gate_infos:
            return None
        for info in self.multi_gate_infos[multi]:
            if info.instruction == instr:
                return info
        return None

    def find_all_qubit_gate(
            self, instr: Type[NetQASMInstruction]
    ) -> Optional[EhiGateInfo]:
        if self.all_qubit_gate_infos is None:
            return None

        for info in self.all_qubit_gate_infos:
            if info.instruction == instr:
                return info
        return None


class EhiBuilder:
    @classmethod
    def decoherence_qubit(
            cls, is_communication: bool, decoherence_rate: float
    ) -> EhiQubitInfo:
        return EhiQubitInfo(
            is_communication=is_communication, decoherence_rate=decoherence_rate
        )

    @classmethod
    def perfect_qubit(cls, is_communication: bool) -> EhiQubitInfo:
        return cls.decoherence_qubit(
            is_communication=is_communication, decoherence_rate=0
        )

    @classmethod
    def decoherence_gates(
            cls,
            duration: float,
            instructions: List[Type[NetQASMInstruction]],
            decoherence: float,
    ) -> List[EhiGateInfo]:
        return [
            EhiGateInfo(instruction=instr, duration=duration, decoherence=decoherence)
            for instr in instructions
        ]

    @classmethod
    def perfect_gates(
            cls, duration: float, instructions: List[Type[NetQASMInstruction]]
    ) -> List[EhiGateInfo]:
        return cls.decoherence_gates(
            duration=duration, instructions=instructions, decoherence=0
        )

    @classmethod
    def perfect_uniform(
            cls,
            num_qubits,
            flavour: Type[Flavour],
            single_instructions: List[Type[NetQASMInstruction]],
            single_duration: float,
            two_instructions: List[Type[NetQASMInstruction]],
            two_duration: float,
            all_qubit_instructions: List[Type[NetQASMInstruction]] = None,
            all_qubit_duration: float = 0,
            latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        if all_qubit_instructions is None:
            return cls.fully_uniform(
                num_qubits=num_qubits,
                flavour=flavour,
                qubit_info=cls.perfect_qubit(is_communication=True),
                single_gate_infos=cls.perfect_gates(
                    single_duration, single_instructions
                ),
                two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
                latencies=latencies,
            )
        else:
            all_qubit_gate_infos = cls.perfect_gates(
                all_qubit_duration, all_qubit_instructions
            )
            return cls.fully_uniform(
                num_qubits=num_qubits,
                flavour=flavour,
                qubit_info=cls.perfect_qubit(is_communication=True),
                single_gate_infos=cls.perfect_gates(
                    single_duration, single_instructions
                ),
                two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
                all_qubit_gate_infos=all_qubit_gate_infos,
                latencies=latencies,
            )

    @classmethod
    def fully_uniform(
            cls,
            num_qubits,
            qubit_info: EhiQubitInfo,
            flavour: Type[Flavour],
            single_gate_infos: List[EhiGateInfo],
            two_gate_infos: List[EhiGateInfo],
            all_qubit_gate_infos: List[EhiGateInfo] = None,
            latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        q_infos = {i: qubit_info for i in range(num_qubits)}
        sg_infos = {i: single_gate_infos for i in range(num_qubits)}
        mg_infos = {}
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    multi = MultiQubit([i, j])
                    mg_infos[multi] = two_gate_infos

        if latencies is None:
            latencies = EhiLatencies.all_zero()

        return EhiNodeInfo(
            q_infos,
            flavour,
            sg_infos,
            mg_infos,
            latencies,
            all_qubit_gate_infos=copy.deepcopy(all_qubit_gate_infos),
        )

    @classmethod
    def perfect_star(
            cls,
            num_qubits: int,
            flavour: Type[Flavour],
            comm_instructions: List[Type[NetQASMInstruction]],
            comm_duration: float,
            mem_instructions: List[Type[NetQASMInstruction]],
            mem_duration: float,
            two_instructions: List[Type[NetQASMInstruction]],
            two_duration: float,
            latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        comm_qubit_info = cls.perfect_qubit(is_communication=True)
        mem_qubit_info = cls.perfect_qubit(is_communication=False)
        comm_gate_infos = cls.perfect_gates(comm_duration, comm_instructions)
        mem_gate_infos = cls.perfect_gates(mem_duration, mem_instructions)
        two_gate_infos = cls.perfect_gates(two_duration, two_instructions)

        q_infos = {0: comm_qubit_info}
        for i in range(1, num_qubits):
            q_infos[i] = mem_qubit_info

        sg_infos = {0: comm_gate_infos}
        for i in range(1, num_qubits):
            sg_infos[i] = mem_gate_infos

        mg_infos = {}
        for i in range(1, num_qubits):
            mg_infos[MultiQubit([0, i])] = two_gate_infos

        if latencies is None:
            latencies = EhiLatencies.all_zero()
        return EhiNodeInfo(q_infos, flavour, sg_infos, mg_infos, latencies)

    @classmethod
    def generic_t1t2_star(
            cls,
            num_qubits: int,
            flavour: Type[Flavour],
            comm_decoherence: float,
            mem_decoherence: float,
            comm_instructions: List[Type[NetQASMInstruction]],
            comm_duration: float,
            comm_instr_decoherence: float,
            mem_instructions: List[Type[NetQASMInstruction]],
            mem_duration: float,
            mem_instr_decoherence: float,
            two_instructions: List[Type[NetQASMInstruction]],
            two_duration: float,
            two_instr_decoherence: float,
            latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        comm_qubit_info = cls.decoherence_qubit(
            is_communication=True, decoherence_rate=comm_decoherence
        )
        mem_qubit_info = cls.decoherence_qubit(
            is_communication=False, decoherence_rate=mem_decoherence
        )
        comm_gate_infos = cls.decoherence_gates(
            comm_duration, comm_instructions, comm_instr_decoherence
        )
        mem_gate_infos = cls.decoherence_gates(
            mem_duration, mem_instructions, mem_instr_decoherence
        )
        two_gate_infos = cls.decoherence_gates(
            two_duration, two_instructions, two_instr_decoherence
        )

        q_infos = {0: comm_qubit_info}
        for i in range(1, num_qubits):
            q_infos[i] = mem_qubit_info

        sg_infos = {0: comm_gate_infos}
        for i in range(1, num_qubits):
            sg_infos[i] = mem_gate_infos

        mg_infos = {}
        for i in range(1, num_qubits):
            mg_infos[MultiQubit([0, i])] = two_gate_infos

        if latencies is None:
            latencies = EhiLatencies.all_zero()
        return EhiNodeInfo(q_infos, flavour, sg_infos, mg_infos, latencies)


@dataclass(frozen=True)
class UnitModule:
    """Description of virtual memory space for programs. Target for a compiler.

    Simply wraps around a EhiNodeInfo object and provides convenience methods.

    Unit Modules should be used as the interface for compilers and schedulers,
    as well as the program itself. This object does not contain information about
    runtime values (i.e. qubit mappings); this is managed by the Memory Manager.
    Only the Memory Manager should use an EhiNodeInfo object itself,
    namely the object that represents the full quantum memory space of the node."""

    info: EhiNodeInfo

    def is_communication(self, qubit_id: int) -> bool:
        return self.info.qubit_infos[qubit_id].is_communication

    @classmethod
    def from_ehi(cls, ehi: EhiNodeInfo, qubit_ids: List[int]) -> UnitModule:
        """Get a subset of an EhiNodeInfo"""
        for id in qubit_ids:
            if id not in ehi.qubit_infos:
                raise ValueError(f"Qubit ID {id} not in EhiNodeInfo")

        qubit_infos = {i: ehi.qubit_infos[i] for i in qubit_ids}
        single_gate_infos = {i: ehi.single_gate_infos[i] for i in qubit_ids}
        multi_gate_infos = {
            ids: info
            for (ids, info) in ehi.multi_gate_infos.items()
            if all(id in qubit_ids for id in ids.qubit_ids)
        }
        all_qubit_gate_infos: Optional[List[EhiGateInfo]] = None
        if len(ehi.qubit_infos.keys()) == len(qubit_ids):
            all_qubit_gate_infos = ehi.all_qubit_gate_infos

        return UnitModule(
            info=EhiNodeInfo(
                qubit_infos,
                ehi.flavour,
                single_gate_infos,
                multi_gate_infos,
                ehi.latencies,
                all_qubit_gate_infos=all_qubit_gate_infos,
            )
        )

    @classmethod
    def from_full_ehi(cls, ehi: EhiNodeInfo) -> UnitModule:
        """Use the full EhiNodeInfo"""
        return UnitModule(info=copy.deepcopy(ehi))

    def get_all_qubit_ids(self) -> List[int]:
        return list(self.info.qubit_infos.keys())


@dataclass
class EhiNetworkTimebin:
    nodes: FrozenSet[int]
    batch_ids: Dict[int, int]  # node ID -> batch ids


@dataclass
class EhiNetworkSchedule:
    bin_length: int
    first_bin: int

    bin_pattern: List[EhiNetworkTimebin] | List[List[EhiNetworkTimebin]]
    repeat_period: int

    length_of_qc_blocks: Dict[Tuple[int, int, int, int], float] | None = None

    # (node_id1, batch_id 1, node_id2, batch_id 1) -> float (length of QC / PGA)
    # TODO: Allow for session PGA/QC to change lengths during the schedule?
    # TODO: Is the key the correct format here, or better to use EhiNetworkTimebin?

    def next_bin(self, time: int, future: bool = False) -> Tuple[int, EhiNetworkTimebin]:
        """
        future: Optional bool. Defaults to False, if set to true, then returns the next time bin in the future,
                i.e. if currently at start of a time bin, then return the next one.
        """
        global_offset = time - self.first_bin

        # print(time)

        # Get the start of the current iteration of the repeating pattern.
        curr_pattern_index = floor(global_offset / self.repeat_period)
        curr_pattern_start = curr_pattern_index * self.repeat_period + self.first_bin

        # Get relative time within the pattern.
        time_since_pattern_start = global_offset - curr_pattern_start

        # Get the index of the next bin within the pattern.
        # It could be that we're already in the last bin. Then the next bin
        # is the first bin of the next pattern repetition.
        next_bin_index = ceil(time_since_pattern_start / self.bin_length)

        if time_since_pattern_start % self.bin_length == 0 and future:  # If we are on a border and
            next_bin_index += 1

        if next_bin_index >= len(self.bin_pattern):
            next_bin_start = curr_pattern_start + self.repeat_period
            next_bin = self.bin_pattern[0]
        else:
            next_bin_start = next_bin_index * self.bin_length + curr_pattern_start
            next_bin = self.bin_pattern[next_bin_index]
        return next_bin_start, next_bin

    def next_specific_bin(self, time: int, bin: EhiNetworkTimebin) -> int:
        bin_index: Optional[int] = None
        first_bin_index: Optional[int] = None

        current_bin_index = (time % self.repeat_period) // self.bin_length + (
            1 if time % self.bin_length != 0 else 0)  # Gets the index of the next bin in the future/present, i.e. if we are part of the way through a bin then the next possible bin is the next one. This will prevent needlessly jumping to the next schedule from taking the bin currently in as the "next" occurrence. It will still return the current bin if queried on a boundary.

        if current_bin_index >= len(self.bin_pattern):
            current_bin_index = 0  # If we don't have enough bins to fill the repeat_period, then if the query time is in the gap between the end of the schedule and the end of the repeat period, then the next bin is the first, i.e. index 0.

        for i, pat_bin in enumerate(self.bin_pattern):
            if bin in pat_bin if isinstance(pat_bin,
                                            list) else bin == pat_bin:  # For parallel operations each element in the bin pattern is a list of bins, so need to check inclusion rather than equality, whilst maintaining compatibility with single bin-per-slot patterns
                if first_bin_index is None:
                    first_bin_index = i
                if bin_index is None and i >= current_bin_index:
                    bin_index = i
                    break  # Breaks after the first bin found in the future/present

        if bin_index is None:
            if first_bin_index is not None:
                bin_index = first_bin_index  # If found a bin in the past / next period then take that as the bin index
            else:
                raise ValueError(f"No index found for bin {bin}")

        bin_rel_to_pat_start = bin_index * self.bin_length

        # TODO: merge below code with that in the `next_bin()` method
        global_offset = time - self.first_bin
        curr_pattern_index = floor(global_offset / self.repeat_period)
        curr_pattern_start = curr_pattern_index * self.repeat_period + self.first_bin
        time_since_pattern_start = global_offset - curr_pattern_start

        # print('|'.join(str(x) for x in [time, bin, current_bin_index, bin_index,bin_rel_to_pat_start - time_since_pattern_start if bin_rel_to_pat_start >= time_since_pattern_start else self.repeat_period - time_since_pattern_start + bin_rel_to_pat_start]))  # For debugging purposes.

        if bin_rel_to_pat_start >= time_since_pattern_start:
            return bin_rel_to_pat_start - time_since_pattern_start
        else:
            return self.repeat_period - time_since_pattern_start + bin_rel_to_pat_start


@dataclass
class EhiNetworkInfo:
    nodes: Dict[int, str]  # node ID -> node name

    # (node A ID, node B ID) -> link info
    # for a pair (a, b) there exists no separate (b, a) info (it is the same)
    links: Dict[FrozenSet[int], EhiLinkInfo]

    network_schedule: Optional[EhiNetworkSchedule] = None

    @classmethod
    def only_nodes(cls, nodes: Dict[int, str]) -> EhiNetworkInfo:
        return EhiNetworkInfo(nodes, {})

    @classmethod
    def fully_connected(
            cls, nodes: Dict[int, str], info: EhiLinkInfo
    ) -> EhiNetworkInfo:
        links: Dict[FrozenSet[int], EhiLinkInfo] = {}
        for n1, n2 in itertools.combinations(nodes.keys(), 2):
            node_link = frozenset([n1, n2])
            links[node_link] = info
        return EhiNetworkInfo(nodes, links)

    @classmethod
    def perfect_fully_connected(
            cls, nodes: Dict[int, str], duration: float
    ) -> EhiNetworkInfo:
        link = EhiLinkInfo(duration=duration, fidelity=1.0)
        return cls.fully_connected(nodes, link)

    def get_node_id(self, name: str) -> int:
        for id, node_name in self.nodes.items():
            if node_name == name:
                return id
        raise ValueError(f"Node with name {name} not found")

    def add_link(self, node1_id: int, node2_id: int, link_info: EhiLinkInfo) -> None:
        if node1_id not in self.nodes:
            raise ValueError(f"Node with ID {node1_id} not found")
        if node2_id not in self.nodes:
            raise ValueError(f"Node with ID {node2_id} not found")
        if node1_id == node2_id:
            raise ValueError("Cannot add link between same node")
        node_link = frozenset([node1_id, node2_id])
        if node_link in self.links:
            raise ValueError(
                f"Link between nodes {node1_id} and {node2_id} already exists"
            )
        self.links[node_link] = link_info

    def get_all_node_names(self) -> List[str]:
        return list(self.nodes.values())

    def get_link(self, node_id1: int, node_id2: int) -> EhiLinkInfo:
        node_link = frozenset([node_id1, node_id2])
        try:
            return self.links[node_link]
        except KeyError:
            raise ValueError(f"No link between nodes {node_id1} and {node_id2}")
