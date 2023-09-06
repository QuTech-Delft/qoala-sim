import pytest
from netqasm.lang.instr import core, nv, trapped_ion
from netqasm.lang.instr.flavour import NVFlavour, TrappedIonFlavour
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
from qoala.runtime.instructions import INSTR_ROT_X_ALL
from qoala.runtime.lhi import (
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiTopologyBuilder,
)
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.ntf import NvNtf, TrappedIonNtf
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


def test_topology_to_ehi_trapped_ion():
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=2,
        single_instructions=[
            INSTR_INIT,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
        single_duration=5e3,
        two_instructions=[],
        two_duration=0,
        all_qubit_instructions=[INSTR_ROT_X_ALL],
        all_qubit_duration=50e3,
    )

    latencies = LhiLatencies(
        host_instr_time=2,
        qnos_instr_time=3,
        host_peer_latency=4,
    )

    interface = TrappedIonNtf()
    ehi = LhiConverter.to_ehi(topology, interface, latencies)

    assert ehi.qubit_infos == {
        0: EhiQubitInfo(is_communication=True, decoherence_rate=0),
        1: EhiQubitInfo(is_communication=True, decoherence_rate=0),
    }

    assert ehi.flavour == TrappedIonFlavour

    single_gates = [
        EhiGateInfo(instr, 5e3, 0)
        for instr in [
            core.InitInstruction,
            trapped_ion.RotZInstruction,
            core.MeasInstruction,
        ]
    ]
    assert ehi.single_gate_infos == {0: single_gates, 1: single_gates}

    assert ehi.multi_gate_infos == {
        MultiQubit([0, 1]): [],
        MultiQubit([1, 0]): [],
    }

    all_qubit_gates = [EhiGateInfo(trapped_ion.AllQubitsRotXInstruction, 50e3, 0)]
    assert all_qubit_gates == ehi.all_qubit_gate_infos

    assert ehi.latencies.host_instr_time == 2
    assert ehi.latencies.qnos_instr_time == 3
    assert ehi.latencies.host_peer_latency == 4


def test_qubit_to_ehi():
    lhi_qubit = LhiTopologyBuilder.t1t2_qubit(is_communication=False, t1=3, t2=4)
    ehi_qubit = LhiConverter.qubit_info_to_ehi(lhi_qubit)

    assert not ehi_qubit.is_communication
    # currently it converts T1 into decoherence rate
    assert ehi_qubit.decoherence_rate == 3

    lhi_qubit2 = LhiTopologyBuilder.perfect_qubit(is_communication=True)
    ehi_qubit2 = LhiConverter.qubit_info_to_ehi(lhi_qubit2)

    assert ehi_qubit2.is_communication
    # currently it converts T1 into decoherence rate
    assert ehi_qubit2.decoherence_rate == 0


def test_gate_to_ehi():
    instructions = [INSTR_INIT, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_MEASURE]
    lhi_gates = LhiTopologyBuilder.perfect_gates(
        duration=5,
        instructions=instructions,
    )

    nvNtf = NvNtf()
    ehi_gates = [
        LhiConverter.gate_info_to_ehi(lhi_gate, nvNtf) for lhi_gate in lhi_gates
    ]

    assert ehi_gates == [
        EhiGateInfo(
            nvNtf.native_to_netqasm(lhi_gate.instruction)[0],
            lhi_gate.duration,
            0,
        )
        for lhi_gate in lhi_gates
    ]

    lhi_gates2 = LhiTopologyBuilder.depolar_gates(
        duration=5, instructions=instructions, depolar_rate=0.5
    )
    ehi_gates2 = [
        LhiConverter.gate_info_to_ehi(lhi_gate, nvNtf) for lhi_gate in lhi_gates2
    ]

    assert ehi_gates2 == [
        EhiGateInfo(
            nvNtf.native_to_netqasm(lhi_gate.instruction)[0],
            lhi_gate.duration,
            0.5,
        )
        for lhi_gate in lhi_gates2
    ]


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
        nodes=nodes,
        links={frozenset([0, 1]): depolar_link, frozenset([1, 3]): perfect_link},
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
    test_topology_to_ehi_trapped_ion()
    test_qubit_to_ehi()
    test_gate_to_ehi()
    test_link_info_to_ehi_perfect()
    test_link_info_to_ehi_depolarise()
    test_network_to_ehi()
