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
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter, NvToNvInterface


def test1():
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


if __name__ == "__main__":
    test1()
