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
from qoala.lang.ehi import EhiGateInfo, EhiQubitInfo
from qoala.runtime.config import DepolariseSamplerConfig, LinkConfig
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.ntf import NvNtf
from qoala.util.math import prob_max_mixed_to_fidelity


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
        host_instr_time=2,
        qnos_instr_time=3,
        host_peer_latency=4,
    )

    interface = NvNtf()
    ehi = LhiConverter.to_ehi(topology, interface, latencies)

    assert ehi.qubit_infos == {
        0: EhiQubitInfo(is_communication=True, decoherence_rate=0),
        1: EhiQubitInfo(is_communication=True, decoherence_rate=0),
    }

    assert ehi.flavour == NVFlavour

    single_gates = [
        EhiGateInfo(instr, 5e3, 0)
        for instr in [
            core.InitInstruction,
            nv.RotXInstruction,
            nv.RotYInstruction,
            nv.RotZInstruction,
            core.MeasInstruction,
        ]
    ]
    assert ehi.single_gate_infos == {0: single_gates, 1: single_gates}

    multi_gates = [EhiGateInfo(nv.ControlledRotXInstruction, 100e3, 0)]

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
    expected_fidelity = prob_max_mixed_to_fidelity(2, prob_max_mixed)
    assert ehi_info.duration == pytest.approx(expected_duration)
    assert ehi_info.fidelity == pytest.approx(expected_fidelity)


def test_network_to_ehi():
    depolar_link = LhiLinkInfo.depolarise(
        cycle_time=10, prob_max_mixed=0.2, prob_success=0.5, state_delay=2000
    )
    perfect_link = LhiLinkInfo.perfect(1000)
    nodes = {0: "node0", 1: "node1"}
    lhi_network = LhiNetworkInfo(
        nodes=nodes, links={(0, 1): depolar_link, (1, 3): perfect_link}
    )

    ehi_network = LhiConverter.network_to_ehi(lhi_network)
    expected_duration_0_1 = (10 / 0.5) + 2000
    expected_fidelty_0_1 = prob_max_mixed_to_fidelity(2, 0.2)

    ehi_link_0_1 = ehi_network.get_link(0, 1)
    assert ehi_link_0_1.duration == pytest.approx(expected_duration_0_1)
    assert ehi_link_0_1.fidelity == pytest.approx(expected_fidelty_0_1)

    ehi_link_1_3 = ehi_network.get_link(1, 3)
    assert ehi_link_1_3.duration == 1000
    assert ehi_link_1_3.fidelity == 1.0


if __name__ == "__main__":
    test_topology_to_ehi()
    test_link_info_to_ehi_perfect()
    test_link_info_to_ehi_depolarise()
    test_network_to_ehi()
