from typing import List

from netqasm.lang.instr import core, nv, vanilla
from netqasm.lang.instr.flavour import Flavour, NVFlavour

from qoala.lang.common import MultiQubit
from qoala.lang.ehi import (
    ExposedGateInfo,
    ExposedHardwareInfo,
    ExposedQubitInfo,
    UnitModule,
)


def qubit_info() -> ExposedQubitInfo:
    return ExposedQubitInfo(is_communication=True, decoherence_rate=0)


def single_gates() -> List[ExposedGateInfo]:
    return [
        ExposedGateInfo(instr, 5e3, 0)
        for instr in [
            core.InitInstruction,
            nv.RotXInstruction,
            nv.RotYInstruction,
            nv.RotZInstruction,
            core.MeasInstruction,
        ]
    ]


def multi_gates() -> List[ExposedGateInfo]:
    return [ExposedGateInfo(nv.ControlledRotXInstruction, 100e3, 0)]


def create_ehi() -> ExposedHardwareInfo:
    num_qubits = 3

    qubit_infos = {i: qubit_info() for i in range(num_qubits)}

    flavour = NVFlavour

    single_gate_infos = {i: single_gates() for i in range(num_qubits)}

    multi_gate_infos = {
        MultiQubit([0, 1]): multi_gates(),
        MultiQubit([1, 0]): multi_gates(),
    }

    return ExposedHardwareInfo(
        qubit_infos, flavour, single_gate_infos, multi_gate_infos
    )


def test_1_qubit():
    ehi = create_ehi()

    for i in range(3):
        um = UnitModule.from_ehi(ehi, qubit_ids=[i])

        assert um.info.qubit_infos == {i: qubit_info()}
        assert um.info.single_gate_infos == {i: single_gates()}
        assert um.info.multi_gate_infos == {}


def test_2_qubits():
    ehi = create_ehi()

    um01 = UnitModule.from_ehi(ehi, qubit_ids=[0, 1])

    assert um01.info.qubit_infos == {0: qubit_info(), 1: qubit_info()}
    assert um01.info.single_gate_infos == {0: single_gates(), 1: single_gates()}
    assert um01.info.multi_gate_infos == {
        MultiQubit([0, 1]): multi_gates(),
        MultiQubit([1, 0]): multi_gates(),
    }

    um02 = UnitModule.from_ehi(ehi, qubit_ids=[0, 2])

    assert um02.info.qubit_infos == {0: qubit_info(), 2: qubit_info()}
    assert um02.info.single_gate_infos == {0: single_gates(), 2: single_gates()}
    assert um02.info.multi_gate_infos == {}

    um12 = UnitModule.from_ehi(ehi, qubit_ids=[1, 2])

    assert um12.info.qubit_infos == {1: qubit_info(), 2: qubit_info()}
    assert um12.info.single_gate_infos == {1: single_gates(), 2: single_gates()}
    assert um12.info.multi_gate_infos == {}


if __name__ == "__main__":
    test_1_qubit()
    test_2_qubits()
