from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import netsquid as ns
import pytest
from netqasm.lang.instr.flavour import (
    Flavour,
    NVFlavour,
    TrappedIonFlavour,
    VanillaFlavour,
)
from netqasm.lang.parsing import parse_text_subroutine
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.nodes import Node
from netsquid.qubits import ketstates

from qoala.lang.ehi import UnitModule
from qoala.lang.program import ProgramMeta, QoalaProgram
from qoala.lang.routine import LocalRoutine, RoutineMetadata
from qoala.runtime.config import NvParams, TopologyConfig
from qoala.runtime.lhi import LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.ntf import GenericNtf, NvNtf, TrappedIonNtf
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.memmgr import AllocError, MemoryManager, NotAllocatedError
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice
from qoala.sim.qnos import (
    GenericProcessor,
    NVProcessor,
    QnosComponent,
    QnosInterface,
    QnosLatencies,
    QnosProcessor,
)
from qoala.sim.qnos.qnosprocessor import (
    IonTrapProcessor,
    UnsupportedNetqasmInstructionError,
)
from qoala.util.math import has_multi_state, has_state
from qoala.util.tests import netsquid_run


def perfect_uniform_qdevice(num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.perfect_uniform(
        num_qubits=num_qubits,
        single_instructions=[
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
        single_duration=5e3,
        two_instructions=[INSTR_CNOT],
        two_duration=100e3,
    )
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


def perfect_nv_star_qdevice(num_qubits: int) -> QDevice:
    cfg = TopologyConfig.from_nv_params(num_qubits, NvParams())
    topology = LhiTopologyBuilder.from_config(cfg)
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


def perfect_trapped_ion_qdevice(num_qubits: int) -> QDevice:
    topology = LhiTopologyBuilder.trapped_ion_default_perfect_gates(num_qubits)
    processor = build_qprocessor_from_topology(name="processor", topology=topology)
    node = Node(name="alice", qmemory=processor)
    return QDevice(node=node, topology=topology)


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
    flavour: Flavour,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> QoalaProcess:
    if uses is None:
        uses = []
    if keeps is None:
        keeps = []
    subrt = parse_text_subroutine(subrt_text, flavour=flavour)
    metadata = RoutineMetadata(uses, keeps)
    iqoala_subrt = LocalRoutine("subrt", subrt, return_vars=[], metadata=metadata)
    meta = ProgramMeta.empty("alice")
    meta.epr_sockets = {0: "bob"}
    program = create_program(subroutines={"subrt": iqoala_subrt}, meta=meta)
    return create_process(pid, program, unit_module)


def create_process_with_vanilla_subrt(
    pid: int,
    subrt_text: str,
    unit_module: UnitModule,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> QoalaProcess:
    return create_process_with_subrt(
        pid, subrt_text, unit_module, VanillaFlavour(), uses, keeps
    )


def create_process_with_nv_subrt(
    pid: int,
    subrt_text: str,
    unit_module: UnitModule,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> QoalaProcess:
    return create_process_with_subrt(
        pid, subrt_text, unit_module, NVFlavour(), uses, keeps
    )


def create_process_with_trapped_ion_subrt(
    pid: int,
    subrt_text: str,
    unit_module: UnitModule,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> QoalaProcess:
    return create_process_with_subrt(
        pid, subrt_text, unit_module, TrappedIonFlavour(), uses, keeps
    )


def set_new_subroutine(
    process: QoalaProcess,
    subrt_text: str,
    flavour: Flavour,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> None:
    if uses is None:
        uses = []
    if keeps is None:
        keeps = []
    subrt = parse_text_subroutine(subrt_text, flavour=flavour)
    metadata = RoutineMetadata(uses, keeps)
    iqoala_subrt = LocalRoutine("subrt", subrt, return_vars=[], metadata=metadata)
    program = process.prog_instance.program
    program.local_routines["subrt"] = iqoala_subrt


def set_new_vanilla_subroutine(
    process: QoalaProcess,
    subrt_text: str,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> None:
    set_new_subroutine(process, subrt_text, VanillaFlavour(), uses, keeps)


def set_new_nv_subroutine(
    process: QoalaProcess,
    subrt_text: str,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> None:
    set_new_subroutine(process, subrt_text, NVFlavour(), uses, keeps)


def set_new_trapped_ion_subroutine(
    process: QoalaProcess,
    subrt_text: str,
    uses: Optional[List[int]] = None,
    keeps: Optional[List[int]] = None,
) -> None:
    set_new_subroutine(process, subrt_text, TrappedIonFlavour(), uses, keeps)


def execute_process(processor: GenericProcessor, process: QoalaProcess) -> int:
    all_routines = process.program.local_routines
    routine = all_routines["subrt"]
    for virt_id in routine.metadata.qubit_use:
        if processor.interface.memmgr.phys_id_for(process.pid, virt_id) is None:
            processor.interface.memmgr.allocate(process.pid, virt_id)

    # input/result arrays not used
    # TODO: add tests that do use these
    inputs = process.inputs.values
    processor.instantiate_routine(process, routine, inputs, MemAddr(0), MemAddr(0))

    netqasm_instructions = routine.subroutine.instructions

    instr_count = 0

    instr_idx = 0
    while instr_idx < len(netqasm_instructions):
        instr_count += 1
        instr_idx = netsquid_run(
            processor.assign_routine_instr(process, "subrt", instr_idx)
        )
    return instr_count


def execute_multiple_processes(
    processor: GenericProcessor, processes: List[QoalaProcess]
) -> None:
    for proc in processes:
        all_routines = proc.program.local_routines
        routine = all_routines["subrt"]
        for virt_id in routine.metadata.qubit_use:
            if processor.interface.memmgr.phys_id_for(proc.pid, virt_id) is None:
                processor.interface.memmgr.allocate(proc.pid, virt_id)

        # input/result arrays not used
        # TODO: add tests that do use these
        inputs = proc.inputs.values
        processor.instantiate_routine(proc, routine, inputs, MemAddr(0), MemAddr(0))
        netqasm_instructions = routine.subroutine.instructions
        for i in range(len(netqasm_instructions)):
            netsquid_run(processor.assign_routine_instr(proc, "subrt", i))


def setup_components_generic(
    num_qubits: int, latencies: QnosLatencies = QnosLatencies.all_zero()
) -> Tuple[QnosProcessor, UnitModule]:
    qdevice = perfect_uniform_qdevice(num_qubits)
    ehi = LhiConverter.to_ehi(qdevice.topology, ntf=GenericNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    qnos_comp = QnosComponent(node=qdevice._node)
    memmgr = MemoryManager(qdevice._node.name, qdevice)
    interface = QnosInterface(qnos_comp, qdevice, memmgr)
    processor = GenericProcessor(interface, latencies)
    return processor, unit_module


def setup_components_nv_star(
    num_qubits: int, latencies: QnosLatencies = QnosLatencies.all_zero()
) -> Tuple[QnosProcessor, UnitModule]:
    qdevice = perfect_nv_star_qdevice(num_qubits)
    ehi = LhiConverter.to_ehi(qdevice.topology, ntf=NvNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    qnos_comp = QnosComponent(node=qdevice._node)
    memmgr = MemoryManager(qdevice._node.name, qdevice)
    interface = QnosInterface(qnos_comp, qdevice, memmgr)
    processor = NVProcessor(interface, latencies)
    return processor, unit_module


def setup_components_trapped_ion(
    num_qubits: int, latencies: QnosLatencies = QnosLatencies.all_zero()
) -> Tuple[QnosProcessor, UnitModule]:
    qdevice = perfect_trapped_ion_qdevice(num_qubits)
    ehi = LhiConverter.to_ehi(qdevice.topology, ntf=TrappedIonNtf())
    unit_module = UnitModule.from_full_ehi(ehi)
    qnos_comp = QnosComponent(node=qdevice._node)
    memmgr = MemoryManager(qdevice._node.name, qdevice)
    interface = QnosInterface(qnos_comp, qdevice, memmgr)
    processor = IonTrapProcessor(interface, latencies)
    return processor, unit_module


def test_init_qubit():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    subrt = """
    set Q0 0
    init Q0
    """

    process = create_process_with_vanilla_subrt(0, subrt, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)


def test_init_qubit_with_latencies():
    ns.sim_reset()

    instr_time = 1e3

    # TODO: improve this
    # Value is copied from hardcoded implementation of `perfect_uniform_qdevice`.
    gate_time = 5e3

    num_qubits = 3
    processor, unit_module = setup_components_generic(
        num_qubits, latencies=QnosLatencies(qnos_instr_time=instr_time)
    )

    subrt = """
    set Q0 0
    init Q0
    """

    process = create_process_with_vanilla_subrt(0, subrt, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process)

    assert ns.sim_time() == 0
    execute_process(processor, process)
    assert ns.sim_time() == instr_time + 1 * gate_time

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)


def test_init_not_allocated():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    subrt = """
    set Q0 0
    init Q0
    """

    # set "uses" to empty list so virt ID 0 is not allocated
    process = create_process_with_vanilla_subrt(0, subrt, unit_module, uses=[])
    processor._interface.memmgr.add_process(process)

    with pytest.raises(NotAllocatedError):
        execute_process(processor, process)


def test_single_gates_generic():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    subrt = """
    set Q0 0
    init Q0
    x Q0
    """

    process = create_process_with_vanilla_subrt(0, subrt, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)

    # Qubit should be in |1>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s1)

    # New subroutine: apply X.
    subrt = """
    x Q0
    """
    set_new_vanilla_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be back in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply Z.
    subrt = """
    z Q0
    """
    set_new_vanilla_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should still be in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply PI/2 Y-rotation.
    # pi/2 = 8 / 2^4 * pi
    subrt = """
    rot_y Q0 8 4
    """
    set_new_vanilla_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be in |+>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.h0)

    # New subroutine: apply -PI/2 Z-rotation.
    # -pi/2 = 24 / 2^4 * pi
    subrt = """
    rot_z Q0 24 4
    """
    set_new_vanilla_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be in |-i>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.y1)

    # New subroutine: apply S on qubit 0 (comm). Should not be allowed.
    subrt = """
    s Q0
    """
    set_new_vanilla_subroutine(process, subrt, [0], [0])
    with pytest.raises(UnsupportedNetqasmInstructionError):
        execute_process(processor, process)


def test_single_gates_multiple_qubits_generic():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    # Initialize q0 and q1. Apply X on q0 and Z on q1.
    subrt = """
    set Q0 0
    init Q0
    set Q1 1
    init Q1
    x Q0
    z Q1
    """

    process = create_process_with_vanilla_subrt(0, subrt, unit_module, [0, 1], [0, 1])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id0 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)
    # Check if qubit with virt ID 1 has been initialized.
    phys_id1 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=1)

    # Virtual qubit 0 should be in |1>
    qubit0 = processor.qdevice.get_local_qubit(phys_id0)
    assert has_state(qubit0, ketstates.s1)

    # Virtual qubit 1 should be in |0>
    qubit1 = processor.qdevice.get_local_qubit(phys_id1)
    assert has_state(qubit1, ketstates.s0)

    # New subroutine: apply X to q0 and Y to q1
    subrt = """
    x Q0
    y Q1
    """
    set_new_vanilla_subroutine(process, subrt, [0, 1], [0, 1])
    execute_process(processor, process)

    # q0 should be back in |0>
    qubit0 = processor.qdevice.get_local_qubit(phys_id0)
    assert has_state(qubit0, ketstates.s0)

    # q1 should be in |1>
    qubit1 = processor.qdevice.get_local_qubit(phys_id1)
    assert has_state(qubit1, ketstates.s1)

    # New subroutine: init q2, apply Y-rotation of PI/2 on q0
    # pi/2 = 8 / 2^4 * pi
    subrt = """
    set Q2 2
    init Q2
    rot_y Q0 8 4
    """
    set_new_vanilla_subroutine(process, subrt, [2], [2])
    execute_process(processor, process)

    # q0 should be in |+>
    qubit0 = processor.qdevice.get_local_qubit(phys_id0)
    assert has_state(qubit0, ketstates.h0)

    # q1 should still be in |1>
    qubit1 = processor.qdevice.get_local_qubit(phys_id1)
    assert has_state(qubit1, ketstates.s1)

    # Check if qubit with virt ID 2 has been initialized.
    phys_id2 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=2)

    # q2 should be in |0>
    qubit2 = processor.qdevice.get_local_qubit(phys_id2)
    assert has_state(qubit2, ketstates.s0)


def test_two_qubit_gates_generic():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    # Initialize q0 and q1. Apply CNOT between q0 and q1.
    subrt = """
    set Q0 0
    init Q0
    set Q1 1
    init Q1
    cnot Q0 Q1
    """

    process = create_process_with_vanilla_subrt(0, subrt, unit_module, [0, 1], [0, 1])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id0 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)
    # Check if qubit with virt ID 1 has been initialized.
    phys_id1 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=1)

    # Virtual qubit 0 should be in |0>
    q0 = processor.qdevice.get_local_qubit(phys_id0)
    assert has_state(q0, ketstates.s0)

    # Virtual qubit 1 should be in |0>
    q1 = processor.qdevice.get_local_qubit(phys_id1)
    assert has_state(q1, ketstates.s0)

    # New subroutine: apply H to q0 and again CNOT between q0 and q1
    subrt = """
    h Q0
    cnot Q0 Q1
    """
    set_new_vanilla_subroutine(process, subrt, [0, 1], [0, 1])
    execute_process(processor, process)

    # q0 and q1 should be maximally entangled
    [q0, q1] = processor.qdevice.get_local_qubits([phys_id0, phys_id1])
    # TODO: fix fidelity calculation with mixed states
    # assert has_max_mixed_state(q0)
    assert has_multi_state([q0, q1], ketstates.b00)

    # New subroutine: apply CNOT between q1 and q0
    subrt = """
    cnot Q1 Q0
    """
    set_new_vanilla_subroutine(process, subrt, [0, 1], [0, 1])
    execute_process(processor, process)

    # q0 should be |0>
    # q1 should be |+>
    [q0, q1] = processor.qdevice.get_local_qubits([phys_id0, phys_id1])
    assert has_state(q0, ketstates.s0)
    assert has_state(q1, ketstates.h0)


def test_multiple_processes_generic():
    num_qubits = 3
    processor, unit_module = setup_components_generic(num_qubits)

    # Process 0: initialize q0.
    subrt0 = """
    set Q0 0
    init Q0
    """

    # Process 1: initialize q0.
    subrt1 = """
    set Q0 0
    init Q0
    """

    process0 = create_process_with_vanilla_subrt(0, subrt0, unit_module, [0], [0])
    process1 = create_process_with_vanilla_subrt(1, subrt1, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process0)
    processor._interface.memmgr.add_process(process1)
    execute_multiple_processes(processor, [process0, process1])

    # Check if qubit with virt ID 0 has been initialized for process 0
    proc0_phys_id0 = processor._interface.memmgr.phys_id_for(pid=0, virt_id=0)
    # Should be mapped to phys ID 0
    assert proc0_phys_id0 == 0

    # Check if qubit with virt ID 0 has been initialized for process 1
    proc1_phys_id0 = processor._interface.memmgr.phys_id_for(pid=1, virt_id=0)
    # Should be mapped to phys ID 1
    assert proc1_phys_id0 == 1

    # Process 0 virt qubit 0 should be in |0>
    proc0_q0 = processor.qdevice.get_local_qubit(proc0_phys_id0)
    assert has_state(proc0_q0, ketstates.s0)

    # Process 1 virt qubit 0 should be in |0>
    proc1_q0 = processor.qdevice.get_local_qubit(proc1_phys_id0)
    assert has_state(proc1_q0, ketstates.s0)

    # New subroutine for process 0: apply X to q0 and initialize q1
    subrt0 = """
    x Q0
    set Q1 1
    init Q1
    """
    set_new_vanilla_subroutine(process0, subrt0, [0, 1], [0, 1])

    # New subroutine for process 0: apply H to q0
    subrt1 = """
    h Q0
    """
    set_new_vanilla_subroutine(process1, subrt1, [0], [0])
    execute_multiple_processes(processor, [process0, process1])

    # Check if qubit with virt ID 1 has been initialized for process 0
    proc0_phys_id1 = processor._interface.memmgr.phys_id_for(pid=0, virt_id=1)
    # Should be mapped to phys ID 2
    assert proc0_phys_id1 == 2

    # Process 0 virt qubit 0 should be in |1>
    proc0_q0 = processor.qdevice.get_local_qubit(proc0_phys_id0)
    assert has_state(proc0_q0, ketstates.s1)

    # Process 0 virt qubit 1 should be in |0>
    proc0_q1 = processor.qdevice.get_local_qubit(proc0_phys_id1)
    assert has_state(proc0_q1, ketstates.s0)

    # Process 1 virt qubit 0 should be in |+>
    proc1_q0 = processor.qdevice.get_local_qubit(proc1_phys_id0)
    assert has_state(proc1_q0, ketstates.h0)

    # New subroutine for process 1: alloc q1
    subrt1 = """
    set Q1 1
    init Q1
    """
    set_new_vanilla_subroutine(process1, subrt1, [1], [1])
    # Should raise an AllocError since no physical qubits left.
    with pytest.raises(AllocError):
        execute_multiple_processes(processor, [process0, process1])

    # Free q0 for process 0
    processor.interface.memmgr.free(process0.pid, 0)
    # Try again same subroutine for process 1
    execute_process(processor, process1)

    # Check that qubit with virt ID 0 has been freed for process 0
    assert processor._interface.memmgr.phys_id_for(pid=0, virt_id=0) is None

    # Check that qubit with virt ID 1 for process 0 is still mapped to phys ID 2
    assert processor._interface.memmgr.phys_id_for(pid=0, virt_id=1) == 2

    # Check that qubit with virt ID 0 for process 1 is still mapped to phys ID 1
    assert processor._interface.memmgr.phys_id_for(pid=1, virt_id=0) == 1

    # Check that qubit with virt ID 1 for process 1 is now mapped to phys ID 0
    assert processor._interface.memmgr.phys_id_for(pid=1, virt_id=1) == 0

    # Check that physical qubit 0 has been reset to |0>
    # (because of initializing virt qubit 1 in process 1)
    phys_qubit_0 = processor.qdevice.get_local_qubit(0)
    assert has_state(phys_qubit_0, ketstates.s0)


def test_single_gates_nv_comm():
    num_qubits = 3

    # Set up a NV processor where
    # - comm qubit can only do ROT_X and ROT_Y
    # - mem qubit can do ROT_X, ROT_Y, ROT_Z
    processor, unit_module = setup_components_nv_star(num_qubits)

    subrt = """
    set Q0 0
    init Q0
    rot_x Q0 16 4
    """

    process = create_process_with_nv_subrt(0, subrt, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)

    # Qubit should be in |1>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s1)

    # New subroutine: apply rot_x on qubit 0 (comm).
    subrt = """
    rot_x Q0 16 4
    """
    set_new_nv_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be back in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply rot_z on qubit 0 (comm), decomposed as rot_x and rot_y.
    subrt = """
    rot_x Q0 24 4
    rot_y Q0 16 4
    rot_x Q0 8 4
    """
    set_new_nv_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should still be in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply PI/2 Y-rotation.
    # pi/2 = 8 / 2^4 * pi
    subrt = """
    rot_y Q0 8 4
    """
    set_new_nv_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be in |+>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.h0)

    # New subroutine: apply -PI/2 Z-rotation, decomposed as X-rotations and Y-rotation.
    # -pi/2 = 24 / 2^4 * pi
    subrt = """
    rot_x Q0 24 4
    rot_y Q0 24 4
    rot_x Q0 8 4
    """
    set_new_nv_subroutine(process, subrt, [0], [0])
    execute_process(processor, process)

    # Qubit should be in |-i>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.y1)


def test_single_gates_nv_mem():
    num_qubits = 3

    # Set up a NV processor where
    # - comm qubit can only do ROT_X and ROT_Y
    # - mem qubit can do ROT_X, ROT_Y, ROT_Z
    processor, unit_module = setup_components_nv_star(num_qubits)

    subrt = """
    set Q1 1
    init Q1
    rot_x Q1 16 4
    """

    process = create_process_with_nv_subrt(0, subrt, unit_module, [1], [1])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 1 has been initialized.
    phys_id = processor._interface.memmgr.phys_id_for(process.pid, virt_id=1)

    # Qubit should be in |1>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s1)

    # New subroutine: apply rot_x on qubit 1 (mem).
    subrt = """
    rot_x Q1 16 4
    """
    set_new_nv_subroutine(process, subrt, [1], [1])
    execute_process(processor, process)

    # Qubit should be back in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply rot_z on qubit 1 (mem). Should be allowed.
    subrt = """
    rot_z Q1 16 4
    """
    set_new_nv_subroutine(process, subrt, [1], [1])
    execute_process(processor, process)

    # Qubit should still be in |0>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.s0)

    # New subroutine: apply PI/2 Y-rotation.
    # pi/2 = 8 / 2^4 * pi
    subrt = """
    rot_y Q1 8 4
    """
    set_new_nv_subroutine(process, subrt, [1], [1])
    execute_process(processor, process)

    # Qubit should be in |+>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.h0)

    # New subroutine: apply -PI/2 Z-rotation.
    # -pi/2 = 24 / 2^4 * pi
    subrt = """
    rot_z Q1 24 4
    """
    set_new_nv_subroutine(process, subrt, [1], [1])
    execute_process(processor, process)

    # Qubit should be in |-i>
    qubit = processor.qdevice.get_local_qubit(phys_id)
    assert has_state(qubit, ketstates.y1)


def test_two_qubit_gates_nv():
    ns.sim_reset()
    num_qubits = 3
    processor, unit_module = setup_components_nv_star(num_qubits)

    # Initialize q0 and q1. Apply CROT_X (angle 0) between q0 and q1.
    subrt = """
    set Q0 0
    init Q0
    set Q1 1
    init Q1
    crot_x Q0 Q1 0 4
    """

    process = create_process_with_nv_subrt(0, subrt, unit_module, [0, 1], [0, 1])
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    # Check if qubit with virt ID 0 has been initialized.
    phys_id0 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=0)
    # Check if qubit with virt ID 1 has been initialized.
    phys_id1 = processor._interface.memmgr.phys_id_for(process.pid, virt_id=1)

    # Virtual qubit 0 should be in |0>
    q0 = processor.qdevice.get_local_qubit(phys_id0)
    assert has_state(q0, ketstates.s0)

    # Virtual qubit 1 should be in |0>
    q1 = processor.qdevice.get_local_qubit(phys_id1)
    assert has_state(q1, ketstates.s0)

    # New subroutine: apply CNOT between q0 and q1.
    # Should not be allowed because CNOT is not in NV flavour.
    subrt = """
    cnot Q0 Q1
    """
    # parse as vanilla just to check that NVProcessor cannot handle it
    # (parsing as NV will already give a parsing error)
    set_new_vanilla_subroutine(process, subrt, [0, 1], [0, 1])
    with pytest.raises(
        NotImplementedError
    ):  # NVProcessor does not implement _interpret_single_qubit_instr
        execute_process(processor, process)

    # New subroutine: apply H to q0 and CNOT between q0 and q1, decomposed using NV gates.
    subrt = """
    rot_y Q0 8 4
    rot_x Q0 16 4
    crot_x Q0 Q1 8 4
    rot_x Q0 24 4
    rot_y Q0 24 4
    rot_x Q0 8 4
    rot_x Q1 24 4
    """
    set_new_nv_subroutine(process, subrt, [0, 1], [0, 1])
    execute_process(processor, process)

    # q0 and q1 should be maximally entangled
    [q0, q1] = processor.qdevice.get_local_qubits([phys_id0, phys_id1])
    assert has_multi_state([q0, q1], ketstates.b00)

    # New subroutine: apply CNOT between q1 and q0, decomposed using NV gates.
    subrt = """
    rot_y Q0 8 4
    rot_x Q0 16 4
    rot_y Q1 8 4
    crot_x Q0 Q1 8 4
    rot_x Q0 24 4
    rot_y Q0 24 4
    rot_x Q0 8 4
    rot_y Q0 8 4
    rot_x Q0 16 4
    rot_x Q1 24 4
    rot_y Q1 24 4
    """
    set_new_nv_subroutine(process, subrt, [0, 1], [0, 1])
    execute_process(processor, process)

    # q0 should be |0>
    # q1 should be |+>
    [q0, q1] = processor.qdevice.get_local_qubits([phys_id0, phys_id1])
    assert has_state(q0, ketstates.s0)
    assert has_state(q1, ketstates.h0)


def test_multiple_processes_nv_alloc_error():
    num_qubits = 3
    processor, unit_module = setup_components_nv_star(num_qubits)

    # Process 0: initialize q0.
    subrt0 = """
    set Q0 0
    init Q0
    """

    # Process 1: initialize q0.
    subrt1 = """
    set Q0 0
    init Q0
    """

    process0 = create_process_with_nv_subrt(0, subrt0, unit_module, [0], [0])
    process1 = create_process_with_nv_subrt(1, subrt1, unit_module, [0], [0])
    processor._interface.memmgr.add_process(process0)
    processor._interface.memmgr.add_process(process1)

    # Should raise an AllocError: both processes allocate their virtual comm qubit
    # but there is only one physical comm qubit available.
    with pytest.raises(AllocError):
        execute_multiple_processes(processor, [process0, process1])


def test_initialize_all():
    num_qubits = 3
    processor, unit_module = setup_components_trapped_ion(num_qubits)

    subrt = """
    init_all
    """
    process = create_process_with_trapped_ion_subrt(
        0, subrt, unit_module, [0, 1, 2], [0, 1, 2]
    )
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)

    assert has_state(qubit0, ketstates.s0)
    assert has_state(qubit1, ketstates.s0)
    assert has_state(qubit2, ketstates.s0)


def test_rotate_all():
    num_qubits = 3
    processor, unit_module = setup_components_trapped_ion(num_qubits)

    subrt = """
        init_all
        rot_x_all 4 2
        """
    process = create_process_with_trapped_ion_subrt(
        0, subrt, unit_module, [0, 1, 2], [0, 1, 2]
    )
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)
    assert has_state(qubit0, ketstates.s1)
    assert has_state(qubit1, ketstates.s1)
    assert has_state(qubit2, ketstates.s1)

    subrt = """
    init_all
    rot_y_all 8 4
    """
    set_new_trapped_ion_subroutine(process, subrt, [0, 1, 2], [0, 1, 2])
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)
    assert has_state(qubit0, ketstates.h0)
    assert has_state(qubit1, ketstates.h0)
    assert has_state(qubit2, ketstates.h0)

    subrt = """
    rot_z_all 24 4
    """

    set_new_trapped_ion_subroutine(process, subrt, [0, 1, 2], [0, 1, 2])
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)

    assert has_state(qubit0, ketstates.y1)
    assert has_state(qubit1, ketstates.y1)
    assert has_state(qubit2, ketstates.y1)


def test_bichromatic():
    num_qubits = 3
    processor, unit_module = setup_components_trapped_ion(num_qubits)

    subrt = """
        init_all
        bichromatic 8 4
        """
    process = create_process_with_trapped_ion_subrt(
        0, subrt, unit_module, [0, 1, 2], [0, 1, 2]
    )
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)

    assert qubit0 is not None
    assert qubit1 is not None
    assert qubit2 is not None


def test_measure_all():
    num_qubits = 3
    processor, unit_module = setup_components_trapped_ion(num_qubits)

    subrt = """
            set Q0 0
            init_all
            meas_all Q0 @output
            """
    process = create_process_with_trapped_ion_subrt(
        0,
        subrt,
        unit_module,
        [0, 1, 2],
    )
    process.shared_mem.allocate_lr_out(3)
    processor._interface.memmgr.add_process(process)
    execute_process(processor, process)

    qubit0 = processor.qdevice.get_local_qubit(0)
    qubit1 = processor.qdevice.get_local_qubit(1)
    qubit2 = processor.qdevice.get_local_qubit(2)

    assert process.shared_mem.read_lr_out(MemAddr(0), 3) == [0, 0, 0]
    assert qubit0 is not None
    assert qubit1 is not None
    assert qubit2 is not None


if __name__ == "__main__":
    test_init_qubit()
    test_init_qubit_with_latencies()
    test_init_not_allocated()
    test_single_gates_generic()
    test_single_gates_multiple_qubits_generic()
    test_two_qubit_gates_generic()
    test_multiple_processes_generic()
    test_single_gates_nv_comm()
    test_single_gates_nv_mem()
    test_two_qubit_gates_nv()
    test_multiple_processes_nv_alloc_error()
    test_initialize_all()
    test_rotate_all()
    test_bichromatic()
