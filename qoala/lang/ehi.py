from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, FrozenSet, List, Optional, Tuple, Type

from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour

from qoala.lang.common import MultiQubit


@dataclass(frozen=True)
class EhiQubitInfo:
    """
    Information about a single qubit. It stores whether the qubit is a communication qubit and its decoherence rate.
    Each qubit in a multi qubit system will have its own EhiQubitInfo which makes it flexible.

    :param is_communication: Whether the qubit is a communication qubit which is important for NV hardware.
    :param decoherence_rate: The decoherence rate of the qubit. This is given as a rate per second.
    """

    is_communication: bool
    decoherence_rate: float  # rate per second


@dataclass(frozen=True)
class EhiGateInfo:
    """
    Information about a single gate. It stores the NetQASM instruction, the duration of the gate and the decoherence.

    :param instruction: The NetQASM instruction that describes the operation of the gate.
    :param duration: The duration of the gate in ns.
    :param decoherence: The decoherence rate of the gate. This is given as a rate per second, for all qubits.
    """

    instruction: Type[NetQASMInstruction]
    duration: float  # ns
    decoherence: float  # rate per second, for all qubits


@dataclass(frozen=True)
class EhiLinkInfo:
    """
    Information about a link between two nodes in the network. It stores the duration and
    fidelity of the entanglement distribution.

    :param duration: The duration of the entanglement distribution in ns.
    :param fidelity: The fidelity of the entanglement distribution. This is a value between 0 and 1.
    """

    duration: float  # ns
    fidelity: float


@dataclass(frozen=True)
class EhiLatencies:
    """
    Describes the latencies of a node in the network. The latencies include the duration of classical
    and quantum operations, the processing time for messages coming from remote nodes and messages between
    the node scheduler and the processor schedulers.

    :param host_instr_time: The duration of classical host instructions in ns.
    :param qnos_instr_time: The duration of quantum instructions in the Qnos processor in ns.
    :param host_peer_latency: The processing time for classical messages from remote nodes in ns.
    :param internal_sched_latency: The processing time for messages between the node scheduler
    and the processor schedulers in ns.
    """

    host_instr_time: float  # duration of classical Host instr execution (CL)
    qnos_instr_time: float  # duration of classical Qnos instr execution (QL)
    host_peer_latency: float  # processing time for Host messages from remote node (CC)
    internal_sched_latency: float  # processing time for messaging between node scheduler and processor schedulers

    @classmethod
    def all_zero(cls) -> EhiLatencies:
        """
        Convenience method to create a EhiLatencies object with all latencies set to 0. Using this method instead of
        setting all latencies to 0 manually is recommended.
        """
        return EhiLatencies(0, 0, 0, 0)


@dataclass(frozen=True)
class EhiNodeInfo:
    """
    Information about a node in the network. It stores information about the qubits, gates and latencies of the node.
    Additionally, the flavour of the node is stored which describes the set of NetQASM instructions that the node can
    execute.

    :param qubit_infos: A dictionary that maps qubit IDs to EhiQubitInfo objects. Each qubit in the node has its own
    EhiQubitInfo object which gives a chance to describe each qubit individually.
    :param flavour: The flavour of the node which describes the set of NetQASM instructions that the node can execute.
    :param single_gate_infos: A dictionary that maps qubit IDs to a list of EhiGateInfo objects. Therefore, each qubit
    has its own list of possible single qubit gates.
    :param multi_gate_infos: A dictionary that maps a MultiQubit object to a list of EhiGateInfo objects. Therefore,
    each qubit has its own list of possible multi qubit gates.
    :param latencies: The latencies of the node. Since it is determined for each node individually, different nodes in
    the network can have different latencies.
    """

    qubit_infos: Dict[int, EhiQubitInfo]  # qubit ID -> info

    flavour: Type[
        Flavour
    ]  # set of NetQASM instrs, no info about which qubits can do what instr
    single_gate_infos: Dict[int, List[EhiGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[EhiGateInfo]
    ]  # ordered qubit ID list -> gates
    latencies: EhiLatencies
    all_qubit_gate_infos: Optional[List[EhiGateInfo]] = (
        None  # gates that are applied to all qubits
    )

    def find_single_gate(
        self, qubit_id: int, instr: Type[NetQASMInstruction]
    ) -> Optional[EhiGateInfo]:
        """
        Find a single qubit gate for a given qubit ID and NetQASM instruction.

        :param qubit_id: The ID of the qubit.
        :param instr: The NetQASM instruction.
        :return: Single qubit gate if the gate is found for the given qubit ID and NetQASM instruction.
        Otherwise, None is returned.
        """
        if qubit_id not in self.single_gate_infos:
            return None
        for info in self.single_gate_infos[qubit_id]:
            if info.instruction == instr:
                return info
        return None

    def find_multi_gate(
        self, qubit_ids: List[int], instr: Type[NetQASMInstruction]
    ) -> Optional[EhiGateInfo]:
        """
        Find a multi qubit gate for a given NetQASM instruction and list of qubit IDs.

        :param qubit_ids: The list of qubit IDs. Note that the order of the qubit IDs is important.
        :param instr: The NetQASM instruction.
        :return: Multi qubit gate if the gate is found for the given NetQASM instruction and list of
        qubit IDs. Otherwise, None is returned.
        """
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
    """
    Convenience class to create EhiNodeInfo objects. It provides methods to create perfect qubits and gates as well
    as qubits and gates with decoherence. Additionally, it provides methods to create a fully uniform node with
    perfect qubits and gates as well as a star shaped node with perfect qubits and gates. Finally, it provides a
    method to create a generic star shaped node with qubits and gates with decoherence.
    """

    @classmethod
    def decoherence_qubit(
        cls, is_communication: bool, decoherence_rate: float
    ) -> EhiQubitInfo:
        """
        Creates a qubit with decoherence.

        :param is_communication: Whether the qubit is a communication qubit which is important for NV hardware.
        :param decoherence_rate: The decoherence rate of the qubit. This is given as a rate per second.
        :return: Qubit with decoherence.
        """
        return EhiQubitInfo(
            is_communication=is_communication, decoherence_rate=decoherence_rate
        )

    @classmethod
    def perfect_qubit(cls, is_communication: bool) -> EhiQubitInfo:
        """
        Creates a perfect qubit. Perfect qubits are qubits without decoherence.

        :param is_communication: Whether the qubit is a communication qubit which is important for NV hardware.
        :return: Qubit with no decoherence.
        """
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
        """
        Creates a list of gates for given list of instructions which has provided decoherence and duration.

        :param duration: The duration of the gates in ns.
        :param instructions: The list of NetQASM instructions.
        :param decoherence: The decoherence rate of the gates. This is given as a rate per second, for all qubits.

        :return: A list of EhiGateInfo objects for given list of instructions, duration and decoherence.
        """
        return [
            EhiGateInfo(instruction=instr, duration=duration, decoherence=decoherence)
            for instr in instructions
        ]

    @classmethod
    def perfect_gates(
        cls, duration: float, instructions: List[Type[NetQASMInstruction]]
    ) -> List[EhiGateInfo]:
        """
        Creates a list of perfect gates for given list of instructions and duration. Perfect gates are gates without
        decoherence.

        :param duration: The duration of the gates in ns.
        :param instructions: The list of NetQASM instructions.

        :return: A list of EhiGateInfo objects for given list of instructions, duration.
        """
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
        all_qubit_instructions: Optional[List[Type[NetQASMInstruction]]] = None,
        all_qubit_duration: float = 0,
        latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        """
        Builds a fully uniform node with perfect qubits and gates. Fully uniform means that all qubits are same and
        all gates are available for all qubits.

        :param num_qubits: The number of qubits in the node.
        :param flavour: The flavour of the node which describes the set of NetQASM instructions that the node can
        execute.
        :param single_instructions: The list of NetQASM instructions for single qubit gates.
        :param single_duration: The duration of single qubit gates in ns.
        :param two_instructions: The list of NetQASM instructions for multi qubit gates.
        :param two_duration: The duration of multi qubit gates in ns.
        :param latencies: The latencies of the node.

        :return: A fully uniform node with perfect qubits and gates.
        """
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
        all_qubit_gate_infos: Optional[List[EhiGateInfo]] = None,
        latencies: Optional[EhiLatencies] = None,
    ) -> EhiNodeInfo:
        """
        Builds a fully uniform node with given qubits and gates. Fully uniform means that all qubits are same and
        all gates are available for all qubits.

        :param num_qubits: The number of qubits in the node.
        :param qubit_info: The information about the qubits. All qubits in the node will have the same information.
        :param flavour: The flavour of the node which describes the set of NetQASM instructions that the node can
        execute.
        :param single_gate_infos: The list of single qubit gate information. All qubits in the node will have the same
        list of single qubit gates.
        :param two_gate_infos: The list of multi qubit gate information. All qubits in the node will have the same
        list of multi qubit gates.
        :param latencies: The latencies of the node.

        :return: A fully uniform node.
        """
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
        """
        Builds a star shaped node with perfect qubits and gates. This shape is used for the NV hardware. It has one
        communication qubit and all other qubits are memory qubits.

        :param num_qubits: The number of qubits in the node.
        :param flavour: The flavour of the node which describes the set of NetQASM instructions that the node can
        execute.
        :param comm_instructions: The list of NetQASM instructions for communication qubit.
        :param comm_duration: The duration of communication qubit gates in ns.
        :param mem_instructions: The list of NetQASM instructions for memory qubits.
        :param mem_duration: The duration of memory qubit gates in ns.
        :param two_instructions: The list of NetQASM instructions for multi qubit gates. Multi qubit gates are used only
        between the communication qubit and memory qubits.
        :param two_duration: The duration of multi qubit gates in ns.
        :param latencies: The latencies of the node.

        :return: A star shaped node with perfect qubits and gates.
        """
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
        """
        Builds a star shaped node with qubits and gates with decoherence. This shape is used for the NV hardware. It has
        one communication qubit and all other qubits are memory qubits.

        :param num_qubits: The number of qubits in the node.
        :param flavour: The flavour of the node which describes the set of NetQASM instructions that the node can
        execute.
        :param comm_decoherence: The decoherence rate of the communication qubit. This is given as a rate per second.
        :param mem_decoherence: The decoherence rate of the memory qubits. This is given as a rate per second.
        :param comm_instructions: The list of NetQASM instructions for communication qubit.
        :param comm_duration: The duration of communication qubit gates in ns.
        :param comm_instr_decoherence: The decoherence rate of the communication qubit gates. This is given as a rate
        per second.
        :param mem_instructions: The list of NetQASM instructions for memory qubits.
        :param mem_duration: The duration of memory qubit gates in ns.
        :param mem_instr_decoherence: The decoherence rate of the memory qubit gates. This is given as a rate per
        second.
        :param two_instructions: The list of NetQASM instructions for multi qubit gates. Multi qubit gates are used only
        between the communication qubit and memory qubits.
        :param two_duration: The duration of multi qubit gates in ns.
        :param two_instr_decoherence: The decoherence rate of the multi qubit gates. This is given as a rate per second.
        :param latencies: The latencies of the node.

        :return: A star shaped node with qubits and gates with decoherence.
        """
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
    namely the object that represents the full quantum memory space of the node.

    :param info: node that this unit module represents
    """

    info: EhiNodeInfo

    def is_communication(self, qubit_id: int) -> bool:
        """
        Checks whether a qubit with given qubit id is a communication qubit.

        :param qubit_id: The ID of the qubit.
        :return: True if the qubit is a communication qubit, False otherwise.

        """
        return self.info.qubit_infos[qubit_id].is_communication

    @classmethod
    def from_ehi(cls, ehi: EhiNodeInfo, qubit_ids: List[int]) -> UnitModule:
        """
        Creates a UnitModule object from a EhiNodeInfo object and a list of qubit IDs. It basically creates a subset of
        the EhiNodeInfo object for given list of qubit IDs.

        :param ehi: The EhiNodeInfo object.
        :param qubit_ids: The list of qubit IDs.

        :return: A UnitModule object that represents a subset of the EhiNodeInfo object for given list of qubit IDs.
        """
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
        """
        Creates a UnitModule object from a EhiNodeInfo object using all of its information.

        :param ehi: The EhiNodeInfo object.

        :return: A UnitModule object that represents the given EhiNodeInfo object.
        """
        return UnitModule(info=copy.deepcopy(ehi))

    def get_all_qubit_ids(self) -> List[int]:
        """
        Returns a list of all qubit IDs in the UnitModule.

        :return: A list of all qubit IDs in the UnitModule.
        """
        return list(self.info.qubit_infos.keys())


@dataclass
class EhiNetworkTimebin:
    nodes: FrozenSet[int]
    pids: Dict[int, int]  # node ID -> PID


@dataclass
class ExplicitTimebin:
    """An EhiNetworkTimebin with explicit start and end time"""

    bin: EhiNetworkTimebin
    start: float
    end: float


@dataclass
class EhiNetworkSchedule:
    bin_length: int
    first_bin: int

    bin_pattern: List[EhiNetworkTimebin]
    repeat_period: int

    def _curr_pattern_start(self, time: int) -> int:
        global_offset = time - self.first_bin

        # Get the start of the current iteration of the repeating pattern.
        curr_pattern_index = floor(global_offset / self.repeat_period)
        return curr_pattern_index * self.repeat_period + self.first_bin

    def current_bin(self, time: int) -> Optional[ExplicitTimebin]:
        # Get relative time within the pattern.
        curr_pattern_start = self._curr_pattern_start(time)
        time_since_pattern_start = time - curr_pattern_start

        # Get the index of the current bin within the pattern.
        curr_bin_index = floor(time_since_pattern_start / self.bin_length)

        # It may be that we're currently not in any bin.
        if curr_bin_index >= len(self.bin_pattern):
            return None
        # Else, find the current bin.
        curr_bin_start = curr_bin_index * self.bin_length + curr_pattern_start
        curr_bin_end = curr_bin_start + self.bin_length
        curr_bin = self.bin_pattern[curr_bin_index]
        return ExplicitTimebin(curr_bin, curr_bin_start, curr_bin_end)

    def next_bin(self, time: int) -> ExplicitTimebin:
        # Get relative time within the pattern.
        curr_pattern_start = self._curr_pattern_start(time)
        time_since_pattern_start = time - curr_pattern_start

        # Get the index of the next bin within the pattern.
        # It could be that we're already in the last bin. Then the next bin
        # is the first bin of the next pattern repetition.
        next_bin_index = ceil(time_since_pattern_start / self.bin_length)

        if next_bin_index >= len(self.bin_pattern):
            next_bin_start = curr_pattern_start + self.repeat_period
            next_bin = self.bin_pattern[0]
        else:
            next_bin_start = next_bin_index * self.bin_length + curr_pattern_start
            next_bin = self.bin_pattern[next_bin_index]

        next_bin_end = next_bin_start + self.bin_length
        return ExplicitTimebin(next_bin, next_bin_start, next_bin_end)

    def next_specific_bin(self, time: int, bin: EhiNetworkTimebin) -> int:
        bin_index: Optional[int] = None

        for i, pat_bin in enumerate(self.bin_pattern):
            if bin == pat_bin:
                bin_index = i
        if bin_index is None:
            raise ValueError

        bin_rel_to_pat_start = bin_index * self.bin_length

        # TODO: merge below code with that in the `next_bin()` method
        global_offset = time - self.first_bin
        curr_pattern_index = floor(global_offset / self.repeat_period)
        curr_pattern_start = curr_pattern_index * self.repeat_period + self.first_bin
        time_since_pattern_start = time - curr_pattern_start
        if bin_rel_to_pat_start >= time_since_pattern_start:
            return bin_rel_to_pat_start - time_since_pattern_start
        else:
            return self.repeat_period - time_since_pattern_start + bin_rel_to_pat_start


@dataclass
class EhiNetworkInfo:
    """
    Information about the network. It stores information about the nodes and links in the network as well as the
    network schedule.

    :param nodes: Nodes in the networks stored as a dictionary that maps node IDs to node names.
    :param links: Links between the pair of nodes in the network as a dictionary that maps a pair of node IDs to
    EhiLinkInfo objects. Therefore, each link has its own EhiLinkInfo object. So different pair of nodes can have
    different EhiLinkInfo objects. Note that for a pair of nodes (a, b) there exists no separate (b, a) info
    (it is the same).
    :param network_schedule: The network schedule of the network.
    """

    nodes: Dict[int, str]  # node ID -> node name

    # (node A ID, node B ID) -> link info
    # for a pair (a, b) there exists no separate (b, a) info (it is the same)
    links: Dict[FrozenSet[int], EhiLinkInfo]

    network_schedule: Optional[EhiNetworkSchedule] = None

    @classmethod
    def only_nodes(cls, nodes: Dict[int, str]) -> EhiNetworkInfo:
        """
        Convenience method to create a EhiNetworkInfo object with only nodes and no links.

        :param nodes: Nodes in the network.
        :return: A network with only nodes and no links.
        """
        return EhiNetworkInfo(nodes, {})

    @classmethod
    def fully_connected(
        cls, nodes: Dict[int, str], info: EhiLinkInfo
    ) -> EhiNetworkInfo:
        """
        Convenience method to create a fully connected network with given nodes and link information. A fully connected
        network means that each pair of nodes has a link between them with the same link information.

        :param nodes: Nodes in the network.
        :param info: Link information between each pair of nodes.
        :return: A fully connected network with given nodes and link information.
        """
        links: Dict[FrozenSet[int], EhiLinkInfo] = {}
        for n1, n2 in itertools.combinations(nodes.keys(), 2):
            node_link = frozenset([n1, n2])
            links[node_link] = info
        return EhiNetworkInfo(nodes, links)

    @classmethod
    def perfect_fully_connected(
        cls, nodes: Dict[int, str], duration: float
    ) -> EhiNetworkInfo:
        """
        Convenience method to create a perfect fully connected network with given nodes and link information.
        A fully connected network means that each pair of nodes has a link between them with the same link information
        and perfect means that the link has 1.0 fidelity.

        :param nodes: Nodes in the network.
        :param duration: The duration of the entanglement distribution in ns.
        :return: A perfect fully connected network with given nodes and link information.
        """
        link = EhiLinkInfo(duration=duration, fidelity=1.0)
        return cls.fully_connected(nodes, link)

    def get_node_id(self, name: str) -> int:
        """
        Returns the node ID for a given node name if the node exists in the network.

        :param name: The name of the node.
        :return: The node ID for a given node name.
        :raises ValueError: If the node does not exist in the network.
        """
        for id, node_name in self.nodes.items():
            if node_name == name:
                return id
        raise ValueError(f"Node with name {name} not found")

    def add_link(self, node1_id: int, node2_id: int, link_info: EhiLinkInfo) -> None:
        """
        Adds a link between two nodes in the network with given link information.

        :param node1_id: The ID of the first node.
        :param node2_id: The ID of the second node.
        :param link_info: The link information.
        :raises ValueError: If one of the nodes does not exist in the network or if the link is between the same node.
        """
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
        """
        Returns a list of all node names in the network.

        :return: A list of all node names in the network.
        """
        return list(self.nodes.values())

    def get_link(self, node_id1: int, node_id2: int) -> EhiLinkInfo:
        """
        Returns the link information between the give pair of nodes.

        :param node_id1: The ID of the first node.
        :param node_id2: The ID of the second node.
        :return: The link information between the give pair of nodes.
        :raises ValueError: If there is no link between the given pair of nodes.
        """
        node_link = frozenset([node_id1, node_id2])
        try:
            return self.links[node_link]
        except KeyError:
            raise ValueError(f"No link between nodes {node_id1} and {node_id2}")
