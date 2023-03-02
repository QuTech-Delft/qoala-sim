# Low-level Hardware Info. Expressed using NetSquid concepts and objects.
from abc import ABC, abstractmethod
from ast import Mult
from dataclasses import dataclass
from inspect import isclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CZ,
    INSTR_H,
    INSTR_INIT,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.instructions import Instruction as NetSquidInstruction
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)

from qoala.lang.common import MultiQubit

# Config Interface


class LhiQubitConfigInterface(ABC):
    @abstractmethod
    def to_is_communication(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class LhiGateConfigInterface(ABC):
    @abstractmethod
    def to_instruction(self) -> Type[NetSquidInstruction]:
        raise NotImplementedError

    @abstractmethod
    def to_duration(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class LhiTopologyConfigInterface(ABC):
    @abstractmethod
    def get_qubit_configs(self) -> Dict[int, LhiQubitConfigInterface]:
        raise NotImplementedError

    @abstractmethod
    def get_single_gate_configs(self) -> Dict[int, List[LhiGateConfigInterface]]:
        raise NotImplementedError

    @abstractmethod
    def get_multi_gate_configs(
        self,
    ) -> Dict[MultiQubit, List[LhiGateConfigInterface]]:
        raise NotImplementedError


# Data classes


@dataclass
class LhiQubitInfo:
    is_communication: bool
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


@dataclass
class LhiGateInfo:
    instruction: Type[NetSquidInstruction]
    duration: int  # ns
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


@dataclass
class LhiTopology:
    qubit_infos: Dict[int, LhiQubitInfo]  # qubit ID -> info
    single_gate_infos: Dict[int, List[LhiGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[LhiGateInfo]
    ]  # ordered qubit ID list -> gates

    def find_single_gate(
        self, qubit_id: int, instr: Type[NetSquidInstruction]
    ) -> Optional[LhiGateInfo]:
        if qubit_id not in self.single_gate_infos:
            return None
        for info in self.single_gate_infos[qubit_id]:
            if info.instruction == instr:
                return info
        return None

    def find_multi_gate(
        self, qubit_ids: List[int], instr: Type[NetSquidInstruction]
    ) -> Optional[LhiGateInfo]:
        multi = MultiQubit(qubit_ids)
        if multi not in self.multi_gate_infos:
            return None
        for info in self.multi_gate_infos[multi]:
            if info.instruction == instr:
                return info
        return None


# Convenience methods.


class LhiTopologyBuilder:
    """Convenience methods for creating a Topology object."""

    @classmethod
    def from_config(cls, cfg: LhiTopologyConfigInterface) -> LhiTopology:
        qubit_infos: Dict[int, LhiQubitInfo] = {}
        for i, cfg_info in cfg.get_qubit_configs().items():
            qubit_infos[i] = LhiQubitInfo(
                is_communication=cfg_info.to_is_communication(),
                error_model=cfg_info.to_error_model(),
                error_model_kwargs=cfg_info.to_error_model_kwargs(),
            )

        single_gate_infos: Dict[int, List[LhiGateInfo]] = {}
        for i, cfg_infos in cfg.get_single_gate_configs().items():
            single_gate_infos[i] = [
                LhiGateInfo(
                    instruction=info.to_instruction(),
                    duration=info.to_duration(),
                    error_model=info.to_error_model(),
                    error_model_kwargs=info.to_error_model_kwargs(),
                )
                for info in cfg_infos
            ]

        multi_gate_infos: Dict[MultiQubit, List[LhiGateInfo]] = {}
        for ids, cfg_infos in cfg.get_multi_gate_configs().items():
            multi_gate_infos[ids] = [
                LhiGateInfo(
                    instruction=info.to_instruction(),
                    duration=info.to_duration(),
                    error_model=info.to_error_model(),
                    error_model_kwargs=info.to_error_model_kwargs(),
                )
                for info in cfg_infos
            ]

        return LhiTopology(
            qubit_infos=qubit_infos,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
        )

    @classmethod
    def t1t2_qubit(cls, is_communication: bool, t1: float, t2: float) -> LhiQubitInfo:
        return LhiQubitInfo(
            is_communication=is_communication,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": t1, "T2": t2},
        )

    @classmethod
    def perfect_qubit(cls, is_communication: bool) -> LhiQubitInfo:
        return cls.t1t2_qubit(is_communication=is_communication, t1=0, t2=0)

    @classmethod
    def depolar_gates(
        cls, duration: int, instructions: List[NetSquidInstruction], depolar_rate: float
    ) -> List[LhiGateInfo]:
        return [
            LhiGateInfo(
                instruction=instr,
                duration=duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": depolar_rate},
            )
            for instr in instructions
        ]

    @classmethod
    def perfect_gates(
        cls, duration: int, instructions: List[NetSquidInstruction]
    ) -> List[LhiGateInfo]:
        return cls.depolar_gates(
            duration=duration, instructions=instructions, depolar_rate=0
        )

    @classmethod
    def perfect_uniform_default_gates(cls, num_qubits) -> LhiTopology:
        # TODO: test this and update default values
        return cls.perfect_uniform(
            num_qubits=num_qubits,
            single_instructions=[
                INSTR_INIT,
                INSTR_X,
                INSTR_Y,
                INSTR_Z,
                INSTR_H,
                INSTR_ROT_X,
                INSTR_ROT_Y,
                INSTR_ROT_Z,
                INSTR_MEASURE,
            ],
            single_duration=5e3,
            two_instructions=[INSTR_CNOT, INSTR_CZ],
            two_duration=100e3,
        )

    @classmethod
    def perfect_uniform(
        cls,
        num_qubits,
        single_instructions: List[NetSquidInstruction],
        single_duration: int,
        two_instructions: List[NetSquidInstruction],
        two_duration: int,
    ) -> LhiTopology:
        return cls.fully_uniform(
            num_qubits=num_qubits,
            qubit_info=cls.perfect_qubit(is_communication=True),
            single_gate_infos=cls.perfect_gates(single_duration, single_instructions),
            two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
        )

    @classmethod
    def fully_uniform(
        cls,
        num_qubits,
        qubit_info: LhiQubitInfo,
        single_gate_infos: List[LhiGateInfo],
        two_gate_infos: List[LhiGateInfo],
    ) -> LhiTopology:
        q_infos = {i: qubit_info for i in range(num_qubits)}
        sg_infos = {i: single_gate_infos for i in range(num_qubits)}
        mg_infos = {}
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    multi = MultiQubit([i, j])
                    mg_infos[multi] = two_gate_infos
        return LhiTopology(q_infos, sg_infos, mg_infos)

    @classmethod
    def perfect_star(
        cls,
        num_qubits: int,
        comm_instructions: List[NetSquidInstruction],
        comm_duration: int,
        mem_instructions: List[NetSquidInstruction],
        mem_duration: int,
        two_instructions: List[NetSquidInstruction],
        two_duration: int,
    ) -> LhiTopology:
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

        return LhiTopology(q_infos, sg_infos, mg_infos)

    @classmethod
    def generic_t1t2_star(
        cls,
        num_qubits: int,
        comm_t1: float,
        comm_t2: float,
        mem_t1: float,
        mem_t2: float,
        comm_instructions: List[NetSquidInstruction],
        comm_duration: int,
        comm_instr_depolar_rate: float,
        mem_instructions: List[NetSquidInstruction],
        mem_duration: int,
        mem_instr_depolar_rate: float,
        two_instructions: List[NetSquidInstruction],
        two_duration: int,
        two_instr_depolar_rate: float,
    ) -> LhiTopology:
        comm_qubit_info = cls.t1t2_qubit(is_communication=True, t1=comm_t1, t2=comm_t2)
        mem_qubit_info = cls.t1t2_qubit(is_communication=False, t1=mem_t1, t2=mem_t2)

        comm_gate_infos = cls.depolar_gates(
            comm_duration, comm_instructions, comm_instr_depolar_rate
        )
        mem_gate_infos = cls.depolar_gates(
            mem_duration, mem_instructions, mem_instr_depolar_rate
        )
        two_gate_infos = cls.depolar_gates(
            two_duration, two_instructions, two_instr_depolar_rate
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

        return LhiTopology(q_infos, sg_infos, mg_infos)
