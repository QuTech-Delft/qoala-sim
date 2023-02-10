# Low-level Hardware Info. Expressed using NetSquid concepts and objects.
from abc import ABC, abstractmethod
from ast import Mult
from dataclasses import dataclass
from operator import is_
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CZ,
    INSTR_INIT,
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

# Config Interface


@dataclass(eq=True, frozen=True)
class MultiQubit:
    qubit_ids: List[int]

    def __hash__(self) -> int:
        return hash(tuple(self.qubit_ids))


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
    def perfect_qubit(cls, is_communication: bool) -> LhiQubitInfo:
        return LhiQubitInfo(
            is_communication=is_communication,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": 0, "T2": 0},
        )

    @classmethod
    def perfect_gates(
        cls, duration: int, instructions: List[NetSquidInstruction]
    ) -> List[LhiGateInfo]:
        return [
            LhiGateInfo(
                instruction=instr,
                duration=duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": 0},
            )
            for instr in instructions
        ]

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
    def build_star_generic_perfect(
        cls,
        num_qubits: int,
    ) -> LhiTopology:
        single_gate_duration = 1_000
        multi_gate_duration = 10_000

        qubit_infos: Dict[int, LhiQubitInfo] = {}
        qubit_infos[0] = LhiQubitInfo(
            is_communication=True,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": 0, "T2": 0},
        )
        for i in range(1, num_qubits):
            qubit_infos[i] = LhiQubitInfo(
                is_communication=False,
                error_model=T1T2NoiseModel,
                error_model_kwargs={"T1": 0, "T2": 0},
            )

        single_gates = [
            LhiGateInfo(
                instruction=instr,
                duration=single_gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={
                    "depolar_rate": 0,
                    "time_independent": True,
                },
            )
            for instr in [INSTR_X, INSTR_Y, INSTR_Z, INSTR_INIT]
        ]
        multi_gates = [
            LhiGateInfo(
                instruction=instr,
                duration=multi_gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={
                    "depolar_rate": 0,
                    "time_independent": True,
                },
            )
            for instr in [INSTR_CNOT, INSTR_CZ]
        ]

        single_gate_infos: Dict[int, LhiGateInfo] = {}
        for i in range(num_qubits):
            single_gate_infos[i] = single_gates

        multi_gate_infos: Dict[MultiQubit, LhiGateInfo] = {}
        for i in range(1, num_qubits):
            multi_gate_infos[(0, i)] = multi_gates

        return LhiTopology(
            qubit_infos=qubit_infos,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
        )

    @classmethod
    def build_star_generic_t1t2(
        cls,
        num_qubits: int,
        t1: int,
        t2: int,
        gate_duration: int,
        depolar_rate: float,
    ) -> LhiTopology:
        qubit_infos: Dict[int, LhiQubitInfo] = {}
        qubit_infos[0] = LhiQubitInfo(
            is_communication=True,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": t1, "T2": t2},
        )
        for i in range(1, num_qubits):
            qubit_infos[i] = LhiQubitInfo(
                is_communication=False,
                error_model=T1T2NoiseModel,
                error_model_kwargs={"T1": t1, "T2": t2},
            )

        single_gates = [
            LhiGateInfo(
                instruction=instr,
                duration=gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={
                    "depolar_rate": depolar_rate,
                    "time_independent": True,
                },
            )
            for instr in [INSTR_X, INSTR_Y, INSTR_Z, INSTR_INIT]
        ]
        multi_gates = [
            LhiGateInfo(
                instruction=instr,
                duration=gate_duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={
                    "depolar_rate": depolar_rate,
                    "time_independent": True,
                },
            )
            for instr in [INSTR_CNOT, INSTR_CZ]
        ]

        single_gate_infos: Dict[int, LhiGateInfo] = {}
        for i in range(num_qubits):
            single_gate_infos[i] = single_gates

        multi_gate_infos: Dict[MultiQubit, LhiGateInfo] = {}
        for i in range(1, num_qubits):
            multi_gate_infos[(0, i)] = multi_gates

        return LhiTopology(
            qubit_infos=qubit_infos,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
        )
