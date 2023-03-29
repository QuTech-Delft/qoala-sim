import pytest
from netqasm.lang.instr import core, nv
from netqasm.lang.instr.flavour import NVFlavour
from netsquid.components.instructions import (
    INSTR_CXDIR,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
)

from qoala.lang.common import MultiQubit
from qoala.lang.ehi import ExposedGateInfo, ExposedQubitInfo
from qoala.runtime.config import DepolariseSamplerConfig, LinkConfig
from qoala.runtime.lhi import LhiLatencies, LhiLinkInfo, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NvToNvInterface
from qoala.util.constants import prob_max_mixed_to_fidelity


def test_topology_to_ehi():
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=2,
        single_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
        single_duration=5e3,
        two_instructions=[INSTR_CXDIR],
        two_duration=100e3,
    )

    latencies = LhiLatencies(
        host_qnos_latency=1,
        host_instr_time=2,
        qnos_instr_time=3,
        host_peer_latency=4,
        netstack_peer_latency=5,
    )

    interface = NvToNvInterface()
    ehi = LhiConverter.to_ehi(topology, interface, latencies)

    assert ehi.qubit_infos == {
        0: ExposedQubitInfo(is_communication=True, decoherence_rate=0),
        1: ExposedQubitInfo(is_communication=True, decoherence_rate=0),
    }

    assert ehi.flavour == NVFlavour

    single_gates = [
        ExposedGateInfo(instr, 5e3, 0)
        for instr in [
            core.InitInstruction,
            nv.RotXInstruction,
            nv.RotYInstruction,
            nv.RotZInstruction,
            core.MeasInstruction,
        ]
    ]
    assert ehi.single_gate_infos == {0: single_gates, 1: single_gates}

    multi_gates = [ExposedGateInfo(nv.ControlledRotXInstruction, 100e3, 0)]

    assert ehi.multi_gate_infos == {
        MultiQubit([0, 1]): multi_gates,
        MultiQubit([1, 0]): multi_gates,
    }

    assert ehi.latencies.host_instr_time == 2
    assert ehi.latencies.qnos_instr_time == 3
    assert ehi.latencies.host_peer_latency == 4


def test_link_info_to_ehi_perfect():
    cfg = LinkConfig.perfect_config(state_delay=1200)
    lhi_info = LhiLinkInfo.from_config(cfg)
    ehi_info = LhiConverter.link_info_to_ehi(lhi_info)

    assert ehi_info.duration == 1200
    assert ehi_info.fidelity == 1.0


def test_link_info_to_ehi_depolarise():
    state_delay = 500
    cycle_time = 10
    prob_max_mixed = 0.3
    prob_success = 0.1

    cfg = LinkConfig(
        state_delay=state_delay,
        sampler_config_cls="DepolariseSamplerConfig",
        sampler_config=DepolariseSamplerConfig(
            cycle_time=cycle_time,
            prob_max_mixed=prob_max_mixed,
            prob_success=prob_success,
        ),
    )
    lhi_info = LhiLinkInfo.from_config(cfg)

    ehi_info = LhiConverter.link_info_to_ehi(lhi_info)

    expected_duration = (cycle_time / prob_success) + state_delay
    expected_fidelity = prob_max_mixed_to_fidelity(prob_max_mixed)
    assert ehi_info.duration == pytest.approx(expected_duration)
    assert ehi_info.fidelity == pytest.approx(expected_fidelity)


if __name__ == "__main__":
    test_topology_to_ehi()
    test_link_info_to_ehi_perfect()
    test_link_info_to_ehi_depolarise()
