from typing import List

from netqasm.lang.instr import core, nv
from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.common import MultiQubit
from qoala.lang.ehi import (
    EhiBuilder,
    EhiGateInfo,
    EhiLatencies,
    EhiLinkInfo,
    EhiNetworkInfo,
    EhiNodeInfo,
    EhiQubitInfo,
    UnitModule,
)


def qubit_info() -> EhiQubitInfo:
    return EhiQubitInfo(is_communication=True, decoherence_rate=0)


def single_gates() -> List[EhiGateInfo]:
    return [
        EhiGateInfo(instr, 5e3, 0)
        for instr in [
            core.InitInstruction,
            nv.RotXInstruction,
            nv.RotYInstruction,
            nv.RotZInstruction,
            core.MeasInstruction,
        ]
    ]


def multi_gates() -> List[EhiGateInfo]:
    return [EhiGateInfo(nv.ControlledRotXInstruction, 100e3, 0)]


def create_ehi() -> EhiNodeInfo:
    num_qubits = 3

    qubit_infos = {i: qubit_info() for i in range(num_qubits)}

    flavour = NVFlavour

    single_gate_infos = {i: single_gates() for i in range(num_qubits)}

    multi_gate_infos = {
        MultiQubit([0, 1]): multi_gates(),
        MultiQubit([1, 0]): multi_gates(),
    }

    return EhiNodeInfo(
        qubit_infos,
        flavour,
        single_gate_infos,
        multi_gate_infos,
        EhiLatencies.all_zero(),
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


def test_full():
    ehi = create_ehi()
    um = UnitModule.from_full_ehi(ehi)

    assert um.info.qubit_infos == {i: qubit_info() for i in range(3)}
    assert um.info.single_gate_infos == {i: single_gates() for i in range(3)}
    assert um.info.multi_gate_infos == {
        MultiQubit([0, 1]): multi_gates(),
        MultiQubit([1, 0]): multi_gates(),
    }


def test_perfect_qubit():
    qubit_info = EhiBuilder.perfect_qubit(is_communication=True)
    assert qubit_info.is_communication
    assert qubit_info.decoherence_rate == 0


def test_decoherence_qubit():
    qubit_info = EhiBuilder.decoherence_qubit(
        is_communication=True, decoherence_rate=0.3
    )
    assert qubit_info.is_communication
    assert qubit_info.decoherence_rate == 0.3


def test_perfect_gates():
    duration = 5e3
    instructions = [nv.RotXInstruction, nv.RotYInstruction]
    gate_infos = EhiBuilder.perfect_gates(duration, instructions)

    assert gate_infos == [
        EhiGateInfo(instruction=instr, duration=duration, decoherence=0)
        for instr in instructions
    ]


def test_decoherence_gates():
    duration = 5e3
    instructions = [nv.RotXInstruction, nv.RotYInstruction]
    gate_infos = EhiBuilder.decoherence_gates(duration, instructions, decoherence=0.2)

    assert gate_infos == [
        EhiGateInfo(instruction=instr, duration=duration, decoherence=0.2)
        for instr in instructions
    ]


def test_perfect_uniform():
    num_qubits = 3
    single_qubit_instructions = [nv.RotXInstruction, nv.RotYInstruction]
    single_gate_duration = 5e3
    two_qubit_instructions = [nv.ControlledRotXInstruction]
    two_gate_duration = 2e5
    ehi = EhiBuilder.perfect_uniform(
        num_qubits,
        NVFlavour,
        single_qubit_instructions,
        single_gate_duration,
        two_qubit_instructions,
        two_gate_duration,
    )

    assert ehi.qubit_infos == {
        i: EhiQubitInfo(is_communication=True, decoherence_rate=0)
        for i in range(num_qubits)
    }

    assert ehi.single_gate_infos == {
        i: [
            EhiGateInfo(instruction=instr, duration=single_gate_duration, decoherence=0)
            for instr in single_qubit_instructions
        ]
        for i in range(num_qubits)
    }

    assert ehi.multi_gate_infos == {
        MultiQubit([i, j]): [
            EhiGateInfo(instruction=instr, duration=two_gate_duration, decoherence=0)
            for instr in two_qubit_instructions
        ]
        for i in range(num_qubits)
        for j in range(num_qubits)
        if i != j
    }


def test_build_fully_uniform():
    qubit_info = EhiQubitInfo(is_communication=True, decoherence_rate=0.01)
    single_gate_infos = [
        EhiGateInfo(instruction=instr, duration=5e3, decoherence=0.02)
        for instr in [nv.RotXInstruction, nv.RotYInstruction]
    ]
    two_gate_infos = [
        EhiGateInfo(
            instruction=nv.ControlledRotXInstruction, duration=2e4, decoherence=0.05
        )
    ]
    ehi = EhiBuilder.fully_uniform(
        num_qubits=3,
        flavour=NVFlavour,
        qubit_info=qubit_info,
        single_gate_infos=single_gate_infos,
        two_gate_infos=two_gate_infos,
    )

    assert len(ehi.qubit_infos) == 3
    for i in range(3):
        assert ehi.qubit_infos[i].is_communication
        single_gates = [info.instruction for info in ehi.single_gate_infos[i]]
        assert nv.RotXInstruction in single_gates
        assert nv.RotYInstruction in single_gates

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            multi = MultiQubit([i, j])
            gates = [info.instruction for info in ehi.multi_gate_infos[multi]]
            assert nv.ControlledRotXInstruction in gates


def test_perfect_star():
    num_qubits = 3
    comm_instructions = [nv.RotXInstruction, nv.RotYInstruction, nv.RotZInstruction]
    comm_duration = 5e3
    mem_instructions = [nv.RotXInstruction, nv.RotYInstruction]
    mem_duration = 1e4
    two_instructions = [nv.ControlledRotXInstruction, nv.ControlledRotYInstruction]
    two_duration = 2e5
    ehi = EhiBuilder.perfect_star(
        num_qubits,
        NVFlavour,
        comm_instructions,
        comm_duration,
        mem_instructions,
        mem_duration,
        two_instructions,
        two_duration,
    )

    assert ehi.qubit_infos[0] == EhiBuilder.perfect_qubit(is_communication=True)

    for i in range(1, num_qubits):
        assert ehi.qubit_infos[i] == EhiBuilder.perfect_qubit(is_communication=False)

    assert ehi.single_gate_infos[0] == EhiBuilder.perfect_gates(
        comm_duration, [nv.RotXInstruction, nv.RotYInstruction, nv.RotZInstruction]
    )
    for i in range(1, num_qubits):
        assert ehi.single_gate_infos[i] == EhiBuilder.perfect_gates(
            mem_duration, [nv.RotXInstruction, nv.RotYInstruction]
        )

    for i in range(1, num_qubits):
        assert ehi.multi_gate_infos[MultiQubit([0, i])] == EhiBuilder.perfect_gates(
            two_duration, [nv.ControlledRotXInstruction, nv.ControlledRotYInstruction]
        )


def test_generic_t1t2_star():
    num_qubits = 3
    comm_decoherence = 0.05
    mem_decoherence = 0.01
    comm_instructions = [nv.RotXInstruction, nv.RotYInstruction, nv.RotZInstruction]
    comm_duration = 5e3
    comm_instr_decoherence = 0.3
    mem_instructions = [nv.RotXInstruction, nv.RotYInstruction]
    mem_duration = 1e4
    mem_instr_decoherence = 0.4
    two_instructions = [nv.ControlledRotXInstruction, nv.ControlledRotYInstruction]
    two_duration = 2e5
    two_instr_decoherence = 0.5
    ehi = EhiBuilder.generic_t1t2_star(
        num_qubits,
        NVFlavour,
        comm_decoherence,
        mem_decoherence,
        comm_instructions,
        comm_duration,
        comm_instr_decoherence,
        mem_instructions,
        mem_duration,
        mem_instr_decoherence,
        two_instructions,
        two_duration,
        two_instr_decoherence,
    )

    assert ehi.qubit_infos[0] == EhiBuilder.decoherence_qubit(True, comm_decoherence)

    for i in range(1, num_qubits):
        assert ehi.qubit_infos[i] == EhiBuilder.decoherence_qubit(
            False, mem_decoherence
        )

    assert ehi.single_gate_infos[0] == EhiBuilder.decoherence_gates(
        comm_duration,
        [nv.RotXInstruction, nv.RotYInstruction, nv.RotZInstruction],
        comm_instr_decoherence,
    )
    for i in range(1, num_qubits):
        assert ehi.single_gate_infos[i] == EhiBuilder.decoherence_gates(
            mem_duration,
            [nv.RotXInstruction, nv.RotYInstruction],
            mem_instr_decoherence,
        )

    for i in range(1, num_qubits):
        assert ehi.multi_gate_infos[MultiQubit([0, i])] == EhiBuilder.decoherence_gates(
            two_duration,
            [nv.ControlledRotXInstruction, nv.ControlledRotYInstruction],
            two_instr_decoherence,
        )


def test_find_gates():
    num_qubits = 3
    comm_decoherence = 0.05
    mem_decoherence = 0.01
    comm_instructions = [nv.RotXInstruction, nv.RotYInstruction, nv.RotZInstruction]
    comm_duration = 5e3
    comm_instr_decoherence = 0.3
    mem_instructions = [nv.RotXInstruction, nv.RotYInstruction]
    mem_duration = 1e4
    mem_instr_decoherence = 0.4
    two_instructions = [nv.ControlledRotXInstruction, nv.ControlledRotYInstruction]
    two_duration = 2e5
    two_instr_decoherence = 0.5
    ehi = EhiBuilder.generic_t1t2_star(
        num_qubits,
        NVFlavour,
        comm_decoherence,
        mem_decoherence,
        comm_instructions,
        comm_duration,
        comm_instr_decoherence,
        mem_instructions,
        mem_duration,
        mem_instr_decoherence,
        two_instructions,
        two_duration,
        two_instr_decoherence,
    )
    for i in range(3):
        assert ehi.find_single_gate(i, nv.RotXInstruction) is not None
        assert ehi.find_single_gate(i, nv.RotYInstruction) is not None
        if i == 0:
            assert ehi.find_single_gate(i, nv.RotZInstruction) is not None
        else:
            assert ehi.find_single_gate(i, nv.RotZInstruction) is None

    for i in range(1, 3):
        assert ehi.find_multi_gate([0, i], nv.ControlledRotXInstruction) is not None
        assert ehi.find_multi_gate([0, i], nv.ControlledRotYInstruction) is not None
        assert ehi.find_multi_gate([i, 0], nv.ControlledRotYInstruction) is None


def test_network_ehi():
    nodes = {0: "node0", 1: "node1", 2: "node2"}
    network_ehi = EhiNetworkInfo.perfect_fully_connected(nodes=nodes, duration=1000)

    assert network_ehi.get_link(0, 1) == EhiLinkInfo(duration=1000, fidelity=1.0)
    assert network_ehi.get_link(0, 2) == EhiLinkInfo(duration=1000, fidelity=1.0)
    assert network_ehi.get_link(1, 2) == EhiLinkInfo(duration=1000, fidelity=1.0)


if __name__ == "__main__":
    test_1_qubit()
    test_2_qubits()
    test_full()
    test_perfect_qubit()
    test_decoherence_qubit()
    test_perfect_gates()
    test_decoherence_gates()
    test_perfect_uniform()
    test_build_fully_uniform()
    test_perfect_star()
    test_generic_t1t2_star()
    test_find_gates()
    test_network_ehi()
