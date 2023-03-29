from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour

from qoala.lang.common import MultiQubit


@dataclass(eq=True, frozen=True)
class ExposedQubitInfo:
    is_communication: bool
    decoherence_rate: float  # rate per second


@dataclass(eq=True, frozen=True)
class ExposedGateInfo:
    instruction: Type[NetQASMInstruction]
    duration: float  # ns
    decoherence: float  # rate per second, for all qubits


@dataclass(eq=True, frozen=True)
class ExposedLinkInfo:
    duration: float  # ns
    fidelity: float


@dataclass(eq=True, frozen=True)
class EhiLatencies:
    host_instr_time: float  # duration of classical Host instr execution (CL)
    qnos_instr_time: float  # duration of classical Qnos instr execution (QL)
    host_peer_latency: float  # processing time for Host messages from remote node (CC)

    @classmethod
    def all_zero(cls) -> EhiLatencies:
        return EhiLatencies(0, 0, 0)


@dataclass(eq=True, frozen=True)
class ExposedHardwareInfo:
    """Hardware made available to offline compiler."""

    qubit_infos: Dict[int, ExposedQubitInfo]  # qubit ID -> info

    flavour: Type[
        Flavour
    ]  # set of NetQASM instrs, no info about which qubits can do what instr
    single_gate_infos: Dict[int, List[ExposedGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[ExposedGateInfo]
    ]  # ordered qubit ID list -> gates

    latencies: EhiLatencies


class EhiBuilder:
    @classmethod
    def decoherence_qubit(
        cls, is_communication: bool, decoherence_rate: float
    ) -> ExposedQubitInfo:
        return ExposedQubitInfo(
            is_communication=is_communication, decoherence_rate=decoherence_rate
        )

    @classmethod
    def perfect_qubit(cls, is_communication: bool) -> ExposedQubitInfo:
        return cls.decoherence_qubit(
            is_communication=is_communication, decoherence_rate=0
        )

    @classmethod
    def decoherence_gates(
        cls,
        duration: float,
        instructions: List[Type[NetQASMInstruction]],
        decoherence: float,
    ) -> List[ExposedGateInfo]:
        return [
            ExposedGateInfo(
                instruction=instr, duration=duration, decoherence=decoherence
            )
            for instr in instructions
        ]

    @classmethod
    def perfect_gates(
        cls, duration: float, instructions: List[Type[NetQASMInstruction]]
    ) -> List[ExposedGateInfo]:
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
        latencies: Optional[EhiLatencies] = None,
    ) -> ExposedHardwareInfo:
        return cls.fully_uniform(
            num_qubits=num_qubits,
            flavour=flavour,
            qubit_info=cls.perfect_qubit(is_communication=True),
            single_gate_infos=cls.perfect_gates(single_duration, single_instructions),
            two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
            latencies=latencies,
        )

    @classmethod
    def fully_uniform(
        cls,
        num_qubits,
        qubit_info: ExposedQubitInfo,
        flavour: Type[Flavour],
        single_gate_infos: List[ExposedGateInfo],
        two_gate_infos: List[ExposedGateInfo],
        latencies: Optional[EhiLatencies] = None,
    ) -> ExposedHardwareInfo:
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
        return ExposedHardwareInfo(q_infos, flavour, sg_infos, mg_infos, latencies)

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
    ) -> ExposedHardwareInfo:
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
        return ExposedHardwareInfo(q_infos, flavour, sg_infos, mg_infos, latencies)

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
    ) -> ExposedHardwareInfo:
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
        return ExposedHardwareInfo(q_infos, flavour, sg_infos, mg_infos, latencies)


@dataclass(eq=True, frozen=True)
class UnitModule:
    """Description of virtual memory space for programs. Target for a compiler.

    Simply wraps around a ExposedHardwareInfo object and provides convenience methods.

    Unit Modules should be used as the interface for compilers and schedulers,
    as well as the program itself. This object does not contain information about
    runtime values (i.e. qubit mappings); this is managed by the Memory Manager.
    Only the Memory Manager should use an ExposedHardwareInfo object itself,
    namely the object that represents the full quantum memory space of the node."""

    info: ExposedHardwareInfo

    def is_communication(self, qubit_id: int) -> bool:
        return self.info.qubit_infos[qubit_id].is_communication

    @classmethod
    def from_ehi(cls, ehi: ExposedHardwareInfo, qubit_ids: List[int]) -> UnitModule:
        """Get a subset of an ExposedHardwareInfo"""
        qubit_infos = {i: ehi.qubit_infos[i] for i in qubit_ids}
        single_gate_infos = {i: ehi.single_gate_infos[i] for i in qubit_ids}
        multi_gate_infos = {
            ids: info
            for (ids, info) in ehi.multi_gate_infos.items()
            if all(id in qubit_ids for id in ids.qubit_ids)
        }

        return UnitModule(
            info=ExposedHardwareInfo(
                qubit_infos,
                ehi.flavour,
                single_gate_infos,
                multi_gate_infos,
                ehi.latencies,
            )
        )

    @classmethod
    def from_full_ehi(cls, ehi: ExposedHardwareInfo) -> UnitModule:
        """Use the full ExposedHardwareInfo"""
        qubit_ids = [i for i in ehi.qubit_infos.keys()]
        return cls.from_ehi(ehi, qubit_ids)

    def get_all_qubit_ids(self) -> List[int]:
        return list(self.info.qubit_infos.keys())
