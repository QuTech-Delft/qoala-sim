from __future__ import annotations

from typing import Dict, Optional, Tuple

import netsquid as ns
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

from qoala.lang.program import IqoalaProgram, LocalRoutine, ProgramMeta
from qoala.lang.routine import RoutineMetadata
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface, LhiConverter
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.schedule import ProgramTaskList
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.memmgr import MemoryManager
from qoala.sim.memory import ProgramMemory
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.sim.qmem import UnitModule
from qoala.sim.qnoscomp import QnosComponent
from qoala.sim.qnosinterface import QnosInterface, QnosLatencies
from qoala.sim.qnosprocessor import GenericProcessor, QnosProcessor
from qoala.util.tests import has_max_mixed_state, has_state, netsquid_run, netsquid_wait


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
    ehi = LhiConverter.to_ehi(qdevice.topology, ntf=GenericToVanillaInterface())
    unit_module = UnitModule.from_full_ehi(ehi)
    qnos_comp = QnosComponent(node=qdevice._node)
    memmgr = MemoryManager(qdevice._node.name, qdevice)
    interface = QnosInterface(qnos_comp, qdevice, memmgr)
    processor = GenericProcessor(interface, latencies=QnosLatencies.all_zero())
    return (processor, unit_module)


def create_program(
    subroutines: Optional[Dict[str, LocalRoutine]] = None,
    meta: Optional[ProgramMeta] = None,
) -> IqoalaProgram:
    if subroutines is None:
        subroutines = {}
    if meta is None:
        meta = ProgramMeta.empty("prog")
    return IqoalaProgram(instructions=[], local_routines=subroutines, meta=meta)


def create_process(
    pid: int, program: IqoalaProgram, unit_module: UnitModule
) -> IqoalaProcess:
    instance = ProgramInstance(
        pid=pid,
        program=program,
        inputs=ProgramInput({}),
        tasks=ProgramTaskList.empty(program),
    )
    mem = ProgramMemory(pid=pid, unit_module=unit_module)

    process = IqoalaProcess(
        prog_instance=instance,
        prog_memory=mem,
        csockets={},
        epr_sockets=program.meta.epr_sockets,
        result=ProgramResult(values={}),
        active_routines={},
    )
    return process


def create_process_with_subrt(
    pid: int, subrt_text: str, unit_module: UnitModule
) -> IqoalaProcess:
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata.use_none()
    iqoala_subrt = LocalRoutine("subrt", subrt, return_map={}, metadata=metadata)
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines={"subrt": iqoala_subrt}, meta=meta)
    return create_process(pid, program, unit_module)


def set_new_subroutine(process: IqoalaProcess, subrt_text: str) -> None:
    subrt = parse_text_subroutine(subrt_text)
    metadata = RoutineMetadata.use_none()
    iqoala_subrt = LocalRoutine("subrt", subrt, return_map={}, metadata=metadata)
    program = process.prog_instance.program
    program.local_routines["subrt"] = iqoala_subrt


def execute_process(processor: GenericProcessor, process: IqoalaProcess) -> int:
    subroutines = process.prog_instance.program.local_routines
    process.instantiate_routine("subrt", 0, {})
    netqasm_instructions = subroutines["subrt"].subroutine.instructions

    instr_count = 0

    instr_idx = 0
    while instr_idx < len(netqasm_instructions):
        instr_count += 1
        instr_idx = netsquid_run(
            processor.assign_routine_instr(process, "subrt", instr_idx)
        )
    return instr_count


def test_depolarizing_decoherence():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    q = qubitapi.create_qubits(1)[0]
    qubitapi.assign_qstate(q, ketstates.s1)
    print(ns.sim_time())
    assert has_state(q, ketstates.s1)
    ns.qubits.delay_depolarize(q, 1e6, delay=1e9)
    print(ns.sim_time())
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
    print(ns.sim_time())
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
    qalloc Q0
    init Q0
    x Q0
    """

    process = create_process_with_subrt(0, subrt, unit_module)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)

    # Qubit should be in |1>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s1)

    netsquid_wait(1e9)
    qubit = processor.qdevice.get_local_qubit(phys_id)
    # Qubit should be fully dephased and amplitude-dampened.
    assert has_state(qubit, ketstates.s0)


if __name__ == "__main__":
    test_depolarizing_decoherence()
    test_depolarizing_decoherence_qprocessor()
    test_depolarizing_decoherence_qprocessor_2()
    test_t1t2_decoherence_qprocessor()
    test_decoherence_in_subroutine()
