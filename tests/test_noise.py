from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import netsquid as ns
import pytest
from netqasm.lang.parsing import parse_text_subroutine
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_H,
    INSTR_I,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel
from netsquid.components.qprocessor import PhysicalInstruction, QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.nodes import Node
from netsquid.qubits import ketstates, qubitapi

from qoala.lang.ehi import UnitModule
from qoala.lang.program import LocalRoutine, ProgramMeta, QoalaProgram
from qoala.lang.routine import RoutineMetadata
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.ntf import GenericNtf
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.memmgr import MemoryManager
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos import (
    GenericProcessor,
    QnosComponent,
    QnosInterface,
    QnosLatencies,
    QnosProcessor,
)
from qoala.util.math import has_max_mixed_state, has_state, prob_max_mixed_to_fidelity
from qoala.util.tests import netsquid_run, netsquid_wait


def uniform_qdevice_noisy_qubits(num_qubits: int) -> QDevice:
    qubit_info = LhiTopologyBuilder.t1t2_qubit(is_communication=True, t1=1, t2=1)
    single_gate_infos = LhiTopologyBuilder.perfect_gates(
        duration=1e4,
        instructions=[
            INSTR_INIT,
            INSTR_X,
            INSTR_Y,
            INSTR_Z,
            INSTR_H,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
    )
    two_gate_infos = LhiTopologyBuilder.perfect_gates(
        duration=1e5,
        instructions=[
            INSTR_CNOT,
        ],
    )

    topology = LhiTopologyBuilder.fully_uniform(
        num_qubits=num_qubits,
        qubit_info=qubit_info,
        single_gate_infos=single_gate_infos,
        two_gate_infos=two_gate_infos,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


def setup_noisy_components(num_qubits: int) -> Tuple[QnosProcessor, UnitModule]:
    qdevice = uniform_qdevice_noisy_qubits(num_qubits)
    ehi = LhiConverter.to_ehi(qdevice.topology, ntf=GenericNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    qnos_comp = QnosComponent(node=qdevice._node)
    memmgr = MemoryManager(qdevice._node.name, qdevice)
    interface = QnosInterface(qnos_comp, qdevice, memmgr)
    processor = GenericProcessor(interface, latencies=QnosLatencies.all_zero())
    return processor, unit_module


def create_program(
    subroutines: Optional[Dict[str, LocalRoutine]] = None,
    meta: Optional[ProgramMeta] = None,
) -> QoalaProgram:
    if subroutines is None:
        subroutines = {}
    if meta is None:
        meta = ProgramMeta.empty("prog")
    return QoalaProgram(blocks=[], local_routines=subroutines, meta=meta)


def create_process(
    pid: int, program: QoalaProgram, unit_module: UnitModule
) -> QoalaProcess:
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=ProgramInput({}),
        unit_module=unit_module,
    )
    mem = ProgramMemory(pid=pid)

    process = QoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
    )
    return process


def create_process_with_subrt(
    pid: int,
    subrt_text: str,
    unit_module: UnitModule,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> QoalaProcess:
    if uses is None:
        uses = []
    if keeps is None:
        keeps = []
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata(qubit_use=uses, qubit_keep=keeps)
    iqoala_subrt = LocalRoutine("subrt", subrt, return_vars=[], metadata=metadata)
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines={"subrt": iqoala_subrt}, meta=meta)
    return create_process(pid, program, unit_module)


def set_new_subroutine(process: QoalaProcess, subrt_text: str) -> None:
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata.use_none()
    iqoala_subrt = LocalRoutine("subrt", subrt, return_vars=[], metadata=metadata)
    program = process.prog_instance.program
    program.local_routines["subrt"] = iqoala_subrt


def test_depolarizing_decoherence():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    q = qubitapi.create_qubits(1)[0]
    qubitapi.assign_qstate(q, ketstates.s1)
    assert has_state(q, ketstates.s1)
    ns.qubits.delay_depolarize(q, 1e6, delay=1e9)
    assert has_max_mixed_state(q)


def test_depolarizing_decoherence_qprocessor():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    phys_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=1e3),
        PhysicalInstruction(INSTR_X, duration=1e3),
        PhysicalInstruction(INSTR_I, duration=1e10),
        PhysicalInstruction(INSTR_Z, duration=1),
    ]
    mem_noise_models = [DepolarNoiseModel(1e10)]
    processor = QuantumProcessor(
        "processor",
        num_positions=1,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    assert ns.sim_time() == 0
    processor.execute_instruction(INSTR_INIT, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e3
    q = processor.peek([0])[0]
    assert has_state(q, ketstates.s0)

    processor.execute_instruction(INSTR_X, [0])
    ns.sim_run()
    assert ns.sim_time() == 2e3
    q = processor.peek([0])[0]
    assert has_state(q, ketstates.s1)

    processor.execute_instruction(INSTR_I, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e10 + 2e3
    q = processor.peek([0])[0]
    # The I instruction made the simulator jump a whole 1e10 ns forward.
    # However, NetSquid does not apply any decoherence noise since any noise
    # is applied by the gate execution itself (in this case: no noise).
    # So, still state |1> is expected.
    assert has_state(q, ketstates.s1)

    netsquid_wait(1e10)
    assert ns.sim_time() == 2e10 + 2e3
    q = processor.peek([0])[0]
    # Now we waited another 1e10 ns, but since it was not because of executing
    # an instruction, decoherence *is* applied.
    assert has_max_mixed_state(q)


def test_depolarizing_decoherence_qprocessor_2():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    phys_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=1e3),
        PhysicalInstruction(INSTR_X, duration=1e3),
        PhysicalInstruction(INSTR_I, duration=1e10),
        PhysicalInstruction(INSTR_Z, duration=1),
    ]
    mem_noise_models = [DepolarNoiseModel(1e10)]
    processor = QuantumProcessor(
        "processor",
        num_positions=1,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    prog = QuantumProgram()
    prog.apply(INSTR_INIT, [0])
    prog.apply(INSTR_X, [0])
    processor.execute_program(prog)
    ns.sim_run()

    # Wait 1e10 ns and then do an I instruction which takes another 1e10 ns.
    netsquid_wait(1e10)

    # NOTE: executing the following code would result in *not* applying any
    # decoherence noise, and the state would still be in |1> !
    # This is because decoherence noise is only applied when accessing the qubit
    # (like `peek` or executing an instruction), and the noise itself depends on
    # the difference between current time and 'last access time' of the qubit.
    # A `peek` with skip_noise=True will hence update the 'last access time' without
    # applying noise, and therefore the following I instruction won't apply any noise
    # either (since the 'last access time' is the same as the current time).
    # Without the following statement (i.e. as it is commented out like now),
    # the I instruction would first result in applying decoherence noise
    # (since last access time is not yet updated). Then, potentially some noise
    # is applied because of the I instruction itself (in this example: no noise).
    #
    # processor.peek([0], skip_noise=True)[0]

    prog = QuantumProgram()
    prog.apply(INSTR_I, [0])
    processor.execute_program(prog)
    ns.sim_run()
    assert ns.sim_time() == 2e10 + 2e3

    # Just before executing the I instruction, the decoherence noise (over 1e10 ns)
    # was applied, since the 'last access time' of the qubit was 1e10 ns ago.
    q = processor.peek([0])[0]
    assert has_max_mixed_state(q)


def test_t1t2_decoherence_qprocessor():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    phys_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=1e3),
        PhysicalInstruction(INSTR_X, duration=1e3),
        PhysicalInstruction(INSTR_I, duration=1e10),
        PhysicalInstruction(INSTR_Z, duration=1),
    ]
    mem_noise_models = [T1T2NoiseModel(T1=10e6, T2=1e6)]
    processor = QuantumProcessor(
        "processor",
        num_positions=1,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    assert ns.sim_time() == 0
    processor.execute_instruction(INSTR_INIT, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e3
    q = processor.peek([0])[0]
    assert has_state(q, ketstates.s0)

    processor.execute_instruction(INSTR_X, [0])
    ns.sim_run()
    assert ns.sim_time() == 2e3
    q = processor.peek([0])[0]
    assert has_state(q, ketstates.s1)

    processor.execute_instruction(INSTR_I, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e10 + 2e3
    q = processor.peek([0])[0]
    # The I instruction made the simulator jump a whole 1e10 ns forward.
    # However, NetSquid does not apply any decoherence noise since any noise
    # is applied by the gate execution itself (in this case: no noise).
    # So, still state |1> is expected.
    assert has_state(q, ketstates.s1)

    netsquid_wait(1e10)
    assert ns.sim_time() == 2e10 + 2e3
    q = processor.peek([0])[0]
    # Now we waited another 1e10 ns, but since it was not because of executing
    # an instruction, decoherence *is* applied.
    assert has_state(q, ketstates.s0)


def test_decoherence_in_subroutine():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    num_qubits = 3
    processor, unit_module = setup_noisy_components(num_qubits)

    subrt = """
    set Q0 0
    init Q0
    x Q0
    """

    process = create_process_with_subrt(0, subrt, unit_module, uses=[0], keeps=[0])
    processor._interface.memmgr.add_process(process)
    processor._interface.memmgr.allocate(process.pid, 0)
    netsquid_run(
        processor.assign_local_routine(process, "subrt", MemAddr(0), MemAddr(1))
    )

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)

    # Qubit should be in |1>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s1)

    netsquid_wait(1e9)
    qubit = processor.qdevice.get_local_qubit(phys_id)
    # Qubit should be fully dephased and amplitude-dampened.
    assert has_state(qubit, ketstates.s0)


def test_gate_noise():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    depolar_prob = 1

    gate_noise = DepolarNoiseModel(depolar_rate=depolar_prob, time_independent=True)
    phys_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=1e3),
        PhysicalInstruction(INSTR_I, duration=1e6, quantum_noise_model=gate_noise),
    ]
    # mem_noise_models = [DepolarNoiseModel(1e10)]
    processor = QuantumProcessor(
        "processor",
        num_positions=1,
        # mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    assert ns.sim_time() == 0
    processor.execute_instruction(INSTR_INIT, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e3
    q = processor.peek([0])[0]
    assert has_state(q, ketstates.s0)

    netsquid_wait(1e10)
    assert ns.sim_time() == 1e10 + 1e3
    q = processor.peek([0])[0]
    # No decoherence, so should still be |0>.
    assert has_state(q, ketstates.s0)

    processor.execute_instruction(INSTR_I, [0])
    ns.sim_run()
    assert ns.sim_time() == 1e10 + 1e3 + 1e6
    q = processor.peek([0])[0]
    # Gate should have applied noise.
    assert has_max_mixed_state(q)

    fidelity = qubitapi.fidelity(q, ketstates.s0, squared=True)
    assert fidelity == pytest.approx(prob_max_mixed_to_fidelity(1, depolar_prob))


def test_two_gate_noise():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    depolar_prob = 1

    gate_noise = DepolarNoiseModel(depolar_rate=depolar_prob, time_independent=True)
    phys_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=1e3),
        PhysicalInstruction(INSTR_CNOT, duration=1e6, quantum_noise_model=gate_noise),
    ]
    processor = QuantumProcessor(
        "processor",
        num_positions=2,
        phys_instructions=phys_instructions,
    )
    assert ns.sim_time() == 0
    processor.execute_instruction(INSTR_INIT, [0])
    ns.sim_run()
    processor.execute_instruction(INSTR_INIT, [1])
    ns.sim_run()
    assert ns.sim_time() == 2e3
    q0 = processor.peek([0])[0]
    q1 = processor.peek([1])[0]
    assert has_state(q0, ketstates.s0)
    assert has_state(q1, ketstates.s0)

    processor.execute_instruction(INSTR_CNOT, [0, 1])
    ns.sim_run()
    assert ns.sim_time() == 2e3 + 1e6
    q0 = processor.peek([0])[0]
    q1 = processor.peek([1])[0]
    print(qubitapi.reduced_dm([q0, q1]))
    # Gate should have applied noise.
    fidelity = qubitapi.fidelity([q0, q1], ketstates.b00, squared=True)
    assert fidelity == pytest.approx(prob_max_mixed_to_fidelity(2, depolar_prob))


if __name__ == "__main__":
    test_depolarizing_decoherence()
    test_depolarizing_decoherence_qprocessor()
    test_depolarizing_decoherence_qprocessor_2()
    test_t1t2_decoherence_qprocessor()
    test_decoherence_in_subroutine()
    test_gate_noise()
    test_two_gate_noise()
