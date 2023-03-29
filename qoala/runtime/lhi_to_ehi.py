import abc
from typing import Any, Dict, Optional, Tuple, Type

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
from netsquid_magic.state_delivery_sampler import (
    DepolariseWithFailureStateSamplerFactory,
    PerfectStateSamplerFactory,
)

from qoala.lang.ehi import (
    EhiLatencies,
    ExposedGateInfo,
    ExposedHardwareInfo,
    ExposedLinkInfo,
    ExposedQubitInfo,
    NetworkEhi,
)
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiGateInfo,
    LhiLatencies,
    LhiLinkInfo,
    LhiQubitInfo,
    LhiTopology,
    NetworkLhi,
)
from qoala.util.constants import prob_max_mixed_to_fidelity


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
        INSTR_MEASURE_INSTANT: core.MeasInstruction,
    }

    def flavour(self) -> Type[Flavour]:
        return VanillaFlavour  # type: ignore

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
        return NVFlavour  # type: ignore

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
            return model_kwargs["depolar_rate"]  # type: ignore
        elif model == T1T2NoiseModel:
            # TODO use T2 somehow
            return model_kwargs["T1"]  # type: ignore
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
        cls,
        topology: LhiTopology,
        ntf: NativeToFlavourInterface,
        latencies: Optional[LhiLatencies] = None,
    ) -> ExposedHardwareInfo:
        if latencies is None:
            latencies = LhiLatencies.all_zero()

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
        flavour = ntf.flavour()

        ehi_latencies = EhiLatencies(
            latencies.host_instr_time,
            latencies.qnos_instr_time,
            latencies.host_peer_latency,
        )

        return ExposedHardwareInfo(
            qubit_infos=qubit_infos,
            flavour=flavour,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
            latencies=ehi_latencies,
        )

    @classmethod
    def link_info_to_ehi(cls, info: LhiLinkInfo) -> ExposedLinkInfo:
        if info.sampler_factory == PerfectStateSamplerFactory:
            return ExposedLinkInfo(duration=info.state_delay, fidelity=1.0)
        elif info.sampler_factory == DepolariseWithFailureStateSamplerFactory:
            expected_gen_duration = (
                info.sampler_kwargs["cycle_time"] / info.sampler_kwargs["prob_success"]
            )
            duration = expected_gen_duration + info.state_delay
            fidelity = prob_max_mixed_to_fidelity(info.sampler_kwargs["prob_max_mixed"])
            return ExposedLinkInfo(duration=duration, fidelity=fidelity)
        else:
            raise NotImplementedError

    @classmethod
    def network_to_ehi(cls, info: NetworkLhi) -> NetworkEhi:
        links: Dict[Tuple[int, int], ExposedLinkInfo] = {}
        for ([n1, n2], link_info) in info.links.items():
            ehi_link = cls.link_info_to_ehi(link_info)
            links[(n1, n2)] = ehi_link
        return NetworkEhi(links)
