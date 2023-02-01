# Low-level Hardware Info. Expressed using NetSquid concepts and objects.
import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour
from netsquid.components.instructions import Instruction as NetSquidInstruction
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)

from qoala.lang.target import ExposedGateInfo, ExposedHardwareInfo, ExposedQubitInfo


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
    gate_infos: Dict[List[int], LhiGateInfo]  # order qubit ID list -> info


class NativeToFlavourInterface(abc.ABC):
    """Interface for mapping NetSquid instructions to NetQASM instructions of a
    particular flavour."""

    # TODO: make sure all instructions mapped by the same interface use the same
    # flavour.
    # TODO: make sure we can control which flavour is used.

    @abc.abstractmethod
    def map(self, ns_instr: Type[NetSquidInstruction]) -> Type[NetQASMInstruction]:
        pass


class LhiConverter:
    @classmethod
    def error_model_to_rate(
        cls, model: Type[QuantumErrorModel], model_kwargs: Dict[str, Any]
    ) -> float:
        if model == DepolarNoiseModel:
            return model_kwargs["depolar_rate"]
        elif model == T1T2NoiseModel:
            return model_kwargs["T1"]  # TODO use T2 somehow
        else:
            raise RuntimeError("Unsupported LHI Error model")

    @classmethod
    def qubit_info_to_ehi(cls, info: LhiQubitInfo) -> ExposedQubitInfo:
        return ExposedQubitInfo(
            is_communication=info.is_communication,
            decoherence_rate=cls.error_model_to_rate(
                info.error_model, info.error_model_kwargs
            ),
        )

    @classmethod
    def gate_info_to_ehi(
        cls, info: LhiGateInfo, ntf: NativeToFlavourInterface
    ) -> ExposedGateInfo:
        instr = ntf.map(info.instruction)
        duration = info.duration
        decoherence = cls.error_model_to_rate(info.error_model, info.error_model_kwargs)
        return ExposedGateInfo(
            instruction=instr, duration=duration, decoherence=decoherence
        )

    @classmethod
    def to_ehi(cls, topology: LhiTopology) -> ExposedHardwareInfo:
        pass
