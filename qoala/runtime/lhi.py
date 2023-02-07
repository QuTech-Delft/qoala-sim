# Low-level Hardware Info. Expressed using NetSquid concepts and objects.
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

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


@dataclass(eq=True, frozen=True)
class MultiQubit:
    qubit_ids: List[int]

    def __hash__(self) -> int:
        return hash(tuple(self.qubit_ids))


@dataclass
class LhiQubitInfo:
    is_communication: bool
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


class LhiQubitConfigInterface:
    @abstractmethod
    def to_is_communication(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class LhiGateInfo:
    instruction: Type[NetSquidInstruction]
    duration: int  # ns
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


class LhiGateConfigInterface:
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


class LhiTopologyConfigInterface:
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


@dataclass
class LhiTopology:
    qubit_infos: Dict[int, LhiQubitInfo]  # qubit ID -> info
    single_gate_infos: Dict[int, List[LhiGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[LhiGateInfo]
    ]  # ordered qubit ID list -> gates


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
