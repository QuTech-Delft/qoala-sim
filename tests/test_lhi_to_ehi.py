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
from qoala.runtime.lhi import LhiLatencies, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NvToNvInterface


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

    interface = NvToNvInterface()
    ehi = LhiConverter.to_ehi(topology, interface)

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


def test_latencies_to_ehi():
    lhi_latencies = LhiLatencies(
        host_qnos_latency=1,
        host_instr_time=2,
        qnos_instr_time=3,
        host_peer_latency=4,
        qnos_peer_latency=5,
    )

    ehi_latencies = LhiConverter.lhi_latencies_to_ehi(lhi_latencies)

    assert ehi_latencies.host_qnos_latency == 1
    assert ehi_latencies.host_instr_time == 2
    assert ehi_latencies.qnos_instr_time == 3
    assert ehi_latencies.host_peer_latency == 4
    assert ehi_latencies.qnos_peer_latency == 5


if __name__ == "__main__":
    test_topology_to_ehi()
    test_latencies_to_ehi()
