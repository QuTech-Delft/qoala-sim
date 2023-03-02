import abc
from typing import Any, Dict, Type

from netqasm.lang.instr import core, nv, vanilla
from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour, NVFlavour, VanillaFlavour
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_CZ,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
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


class GenericToVanillaInterface(NativeToFlavourInterface):
    _MAP: Dict[Type[NetSquidInstruction], Type[NetQASMInstruction]] = {
        INSTR_INIT: core.InitInstruction,
        INSTR_X: vanilla.GateXInstruction,
        INSTR_Y: vanilla.GateYInstruction,
        INSTR_Z: vanilla.GateZInstruction,
        INSTR_H: vanilla.GateHInstruction,
        INSTR_ROT_X: vanilla.RotXInstruction,
        INSTR_ROT_Y: vanilla.RotYInstruction,
        INSTR_ROT_Z: vanilla.RotZInstruction,
        INSTR_CNOT: vanilla.CnotInstruction,
        INSTR_CZ: vanilla.CphaseInstruction,
        INSTR_MEASURE: core.MeasInstruction,
    }

    def flavour(self) -> Type[Flavour]:
        return VanillaFlavour

    def map(self, ns_instr: Type[NetSquidInstruction]) -> Type[NetQASMInstruction]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        return self._MAP[ns_instr]


class NvToNvInterface(NativeToFlavourInterface):
    _MAP: Dict[Type[NetSquidInstruction], Type[NetQASMInstruction]] = {
        INSTR_INIT: core.InitInstruction,
        INSTR_ROT_X: nv.RotXInstruction,
        INSTR_ROT_Y: nv.RotYInstruction,
        INSTR_ROT_Z: nv.RotZInstruction,
        INSTR_CXDIR: nv.ControlledRotXInstruction,
        INSTR_CYDIR: nv.ControlledRotYInstruction,
        INSTR_MEASURE: core.MeasInstruction,
    }

    def flavour(self) -> Type[Flavour]:
        return NVFlavour

    def map(self, ns_instr: Type[NetSquidInstruction]) -> Type[NetQASMInstruction]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        return self._MAP[ns_instr]


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
        qubit_infos = {
            id: cls.qubit_info_to_ehi(qi) for (id, qi) in topology.qubit_infos.items()
        }
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
