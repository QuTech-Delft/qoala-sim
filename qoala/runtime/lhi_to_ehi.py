from typing import Any, Dict, FrozenSet, Optional, Type

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
    EhiGateInfo,
    EhiLatencies,
    EhiLinkInfo,
    EhiNetworkInfo,
    EhiNetworkSchedule,
    EhiNetworkTimebin,
    EhiNodeInfo,
    EhiQubitInfo,
)
from qoala.runtime.lhi import (
    LhiGateInfo,
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiNetworkSchedule,
    LhiNetworkTimebin,
    LhiQubitInfo,
    LhiTopology,
)
from qoala.runtime.ntf import NtfInterface
from qoala.util.math import prob_max_mixed_to_fidelity


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
    def qubit_info_to_ehi(cls, info: LhiQubitInfo) -> EhiQubitInfo:
        return EhiQubitInfo(
            is_communication=info.is_communication,
            decoherence_rate=cls.error_model_to_rate(
                info.error_model, info.error_model_kwargs
            ),
        )

    @classmethod
    def gate_info_to_ehi(cls, info: LhiGateInfo, ntf: NtfInterface) -> EhiGateInfo:
        # TODO: deal with mapping to multiple gates
        instr = ntf.native_to_netqasm(info.instruction)[0]  # (!)
        duration = info.duration
        decoherence = cls.error_model_to_rate(info.error_model, info.error_model_kwargs)
        return EhiGateInfo(
            instruction=instr, duration=duration, decoherence=decoherence
        )

    @classmethod
    def to_ehi(
        cls,
        topology: LhiTopology,
        ntf: NtfInterface,
        latencies: Optional[LhiLatencies] = None,
    ) -> EhiNodeInfo:
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
        if topology.all_qubit_gate_infos is None:
            all_qubit_gate_infos = []
        else:
            all_qubit_gate_infos = [
                cls.gate_info_to_ehi(gi, ntf) for gi in topology.all_qubit_gate_infos
            ]

        flavour = ntf.flavour()

        ehi_latencies = EhiLatencies(
            latencies.host_instr_time,
            latencies.qnos_instr_time,
            latencies.host_peer_latency,
            latencies.internal_sched_latency,
        )

        return EhiNodeInfo(
            qubit_infos=qubit_infos,
            flavour=flavour,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
            all_qubit_gate_infos=all_qubit_gate_infos,
            latencies=ehi_latencies,
        )

    @classmethod
    def link_info_to_ehi(cls, info: LhiLinkInfo) -> EhiLinkInfo:
        if info.sampler_factory == PerfectStateSamplerFactory:
            return EhiLinkInfo(duration=info.state_delay, fidelity=1.0)
        elif info.sampler_factory == DepolariseWithFailureStateSamplerFactory:
            expected_gen_duration = (
                info.sampler_kwargs["cycle_time"] / info.sampler_kwargs["prob_success"]
            )
            duration = expected_gen_duration + info.state_delay
            fidelity = prob_max_mixed_to_fidelity(
                2, info.sampler_kwargs["prob_max_mixed"]
            )
            return EhiLinkInfo(duration=duration, fidelity=fidelity)
        else:
            raise NotImplementedError

    @classmethod
    def network_to_ehi(cls, info: LhiNetworkInfo) -> EhiNetworkInfo:
        links: Dict[FrozenSet[int], EhiLinkInfo] = {}
        for (node_link, link_info) in info.links.items():
            ehi_link = cls.link_info_to_ehi(link_info)
            links[node_link] = ehi_link
        return EhiNetworkInfo(info.nodes, links)

    @classmethod
    def timebin_to_ehi(cls, bin: LhiNetworkTimebin) -> EhiNetworkTimebin:
        return EhiNetworkTimebin(bin.nodes, bin.pids)

    @classmethod
    def netschedule_to_ehi(cls, schedule: LhiNetworkSchedule) -> EhiNetworkSchedule:
        return EhiNetworkSchedule(
            bin_length=schedule.bin_length,
            first_bin=schedule.first_bin,
            bin_pattern=[cls.timebin_to_ehi(bin) for bin in schedule.bin_pattern],
            repeat_period=schedule.repeat_period,
        )
