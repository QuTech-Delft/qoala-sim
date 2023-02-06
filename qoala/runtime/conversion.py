import abc
from typing import Any, Dict, Type

from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour
from netsquid.components.instructions import Instruction as NetSquidInstruction
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)

from qoala.lang.ehi import ExposedGateInfo, ExposedHardwareInfo, ExposedQubitInfo
from qoala.runtime.lhi import LhiGateInfo, LhiQubitInfo, LhiTopology


class NativeToFlavourInterface(abc.ABC):
    @abc.abstractmethod
    def flavour(self) -> Type[Flavour]:
        raise NotImplementedError

    @abc.abstractmethod
    def map(self, ns_instr: Type[NetSquidInstruction]) -> Type[NetQASMInstruction]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        raise NotImplementedError


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
    def to_ehi(
        cls, topology: LhiTopology, ntf: NativeToFlavourInterface
    ) -> ExposedHardwareInfo:
        qubit_infos = [cls.qubit_info_to_ehi(qi) for qi in topology.qubit_infos]
        single_gate_infos = {
            id: [cls.gate_info_to_ehi(gi, ntf) for gi in gis]
            for (id, gis) in topology.single_gate_infos.items()
        }
        multi_gate_infos = {
            ids: [cls.gate_info_to_ehi(gi, ntf) for gi in gis]
            for (ids, gis) in topology.multi_gate_infos.items()
        }
        return ExposedHardwareInfo(
            qubit_infos=qubit_infos,
            flavour=ntf.flavour(),
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
        )
