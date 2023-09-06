from __future__ import annotations

from typing import List, Set, Tuple

import pytest

from qoala.lang.ehi import EhiBuilder, UnitModule
from qoala.lang.program import ProgramMeta, QoalaProgram
from qoala.runtime.memory import ProgramMemory
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice


class MockQDevice(QDevice):
    def __init__(self, typ: str = "star") -> None:
        self._typ = typ

    def get_all_qubit_ids(self) -> Set[int]:
        return {0, 1}

    def get_comm_qubit_ids(self) -> Set[int]:
        if self._typ == "star":
            return {0}
        else:
            assert self._typ == "uniform"
            return {0, 1}

    def get_non_comm_qubit_ids(self) -> Set[int]:
        if self._typ == "star":
            return {1}
        else:
            assert self._typ == "uniform"
            return {}

    def set_mem_pos_in_use(self, id: int, in_use: bool) -> None:
        pass


def create_process(pid: int, unit_module: UnitModule) -> QoalaProcess:
    program = QoalaProgram(
        blocks=[],
        local_routines={},
        meta=ProgramMeta.empty("prog"),
    )
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


def create_unit_module_star() -> UnitModule:
    ehi = EhiBuilder.perfect_star(
        num_qubits=2,
        flavour=None,
        comm_instructions=[],
        comm_duration=0,
        mem_instructions=[],
        mem_duration=0,
        two_instructions=[],
        two_duration=0,
    )
    return UnitModule.from_full_ehi(ehi)


def create_unit_module_uniform() -> UnitModule:
    ehi = EhiBuilder.perfect_uniform(
        num_qubits=2,
        flavour=None,
        single_instructions=[],
        single_duration=0,
        two_instructions=[],
        two_duration=0,
    )
    return UnitModule.from_full_ehi(ehi)


def setup_manager(typ: str = "star") -> Tuple[int, MemoryManager]:
    if typ == "star":
        um = create_unit_module_star()
        qdevice = MockQDevice()
        mgr = MemoryManager("alice", qdevice)
    else:
        assert typ == "uniform"
        um = create_unit_module_uniform()
        qdevice = MockQDevice(typ="uniform")
        mgr = MemoryManager("alice", qdevice)

    process = create_process(0, um)
    mgr.add_process(process)
    pid = process.pid

    return pid, mgr


def setup_manager_multiple_processes(
    num_processes: int, typ: str = "star"
) -> Tuple[List[int], MemoryManager]:
    if typ == "star":
        um = create_unit_module_star()
        qdevice = MockQDevice()
        mgr = MemoryManager("alice", qdevice)
    else:
        assert typ == "uniform"
        um = create_unit_module_uniform()
        qdevice = MockQDevice(typ="uniform")
        mgr = MemoryManager("alice", qdevice)

    pids: List[int] = []

    for i in range(num_processes):
        process = create_process(i, um)
        mgr.add_process(process)
        pids.append(process.pid)

    return pids, mgr


def test_alloc_free_0():
    pid, mgr = setup_manager()

    assert mgr.phys_id_for(pid, 0) is None
    assert mgr.phys_id_for(pid, 1) is None
    assert mgr.virt_id_for(pid, 0) is None
    assert mgr.virt_id_for(pid, 1) is None

    with pytest.raises(RuntimeError):
        assert mgr.phys_id_for(pid, 2) is None

    with pytest.raises(RuntimeError):
        assert mgr.virt_id_for(pid, 2) is None

    mgr.allocate(pid, 0)
    assert mgr.phys_id_for(pid, 0) == 0
    assert mgr.phys_id_for(pid, 1) is None
    assert mgr.virt_id_for(pid, 0) == 0
    assert mgr.virt_id_for(pid, 1) is None

    mgr.free(pid, 0)
    assert mgr.phys_id_for(pid, 0) is None
    assert mgr.phys_id_for(pid, 1) is None
    assert mgr.virt_id_for(pid, 0) is None
    assert mgr.virt_id_for(pid, 1) is None

    with pytest.raises(AssertionError):
        mgr.free(pid, 5)


def test_alloc_free_0_1():
    pid, mgr = setup_manager()

    assert mgr.phys_id_for(pid, 0) is None
    assert mgr.phys_id_for(pid, 1) is None
    assert mgr.virt_id_for(pid, 0) is None
    assert mgr.virt_id_for(pid, 1) is None

    mgr.allocate(pid, 0)
    mgr.allocate(pid, 1)
    assert mgr.phys_id_for(pid, 0) == 0
    assert mgr.phys_id_for(pid, 1) == 1
    assert mgr.virt_id_for(pid, 0) == 0
    assert mgr.virt_id_for(pid, 1) == 1

    mgr.free(pid, 0)
    assert mgr.phys_id_for(pid, 0) is None
    assert mgr.phys_id_for(pid, 1) == 1
    assert mgr.virt_id_for(pid, 0) is None
    assert mgr.virt_id_for(pid, 1) == 1


def test_alloc_non_existing():
    pid, mgr = setup_manager()

    with pytest.raises(AllocError):
        mgr.allocate(pid, 2)


def test_alloc_already_allocated():
    pid, mgr = setup_manager()

    mgr.allocate(pid, 1)

    with pytest.raises(AllocError):
        mgr.allocate(pid, 1)


def test_free_alreay_freed():
    pid, mgr = setup_manager()

    with pytest.raises(AllocError):
        mgr.free(pid, 0)


def test_get_unmapped_qubit():
    pid, mgr = setup_manager()

    assert mgr.get_unmapped_non_comm_qubit(pid) == 1
    mgr.allocate(pid, 0)
    assert mgr.get_unmapped_non_comm_qubit(pid) == 1
    mgr.allocate(pid, 1)
    with pytest.raises(AllocError):
        mgr.get_unmapped_non_comm_qubit(pid)
    mgr.free(pid, 1)
    assert mgr.get_unmapped_non_comm_qubit(pid) == 1


def test_alloc_multiple_processes():
    [pid0, pid1], mgr = setup_manager_multiple_processes(2)

    assert mgr.phys_id_for(pid0, 0) is None
    assert mgr.phys_id_for(pid0, 1) is None
    assert mgr.virt_id_for(pid0, 0) is None
    assert mgr.virt_id_for(pid0, 1) is None
    assert mgr.phys_id_for(pid1, 0) is None
    assert mgr.phys_id_for(pid1, 1) is None
    assert mgr.virt_id_for(pid1, 0) is None
    assert mgr.virt_id_for(pid1, 1) is None

    mgr.allocate(pid0, 0)
    # Should allocate phys ID 0 for virt ID 0 of pid0
    assert mgr.phys_id_for(pid0, 0) == 0
    assert mgr.virt_id_for(pid0, 0) == 0

    with pytest.raises(AllocError):
        # Should try to allocate phys ID 0, but it's not available
        mgr.allocate(pid1, 0)

    mgr.allocate(pid1, 1)
    # Should have mapping:
    #   phys ID 0 : (pid0, virt0)
    #   phys ID 1 : (pid1, virt1)
    assert mgr.phys_id_for(pid0, 0) == 0
    assert mgr.phys_id_for(pid0, 1) is None
    assert mgr.virt_id_for(pid0, 0) == 0
    assert mgr.virt_id_for(pid0, 1) is None
    assert mgr.phys_id_for(pid1, 0) is None
    assert mgr.phys_id_for(pid1, 1) == 1
    assert mgr.virt_id_for(pid1, 0) is None
    assert mgr.virt_id_for(pid1, 1) == 1


def test_alloc_multiple_processes_same_virt_id():
    [pid0, pid1], mgr = setup_manager_multiple_processes(2, typ="uniform")

    mgr.allocate(pid0, 0)
    # Should allocate phys ID 0 for virt ID 0 of pid0
    assert mgr.phys_id_for(pid0, 0) == 0
    assert mgr.virt_id_for(pid0, 0) == 0

    # Should allocate phys ID 1 for virt ID 0 of pid1
    mgr.allocate(pid1, 0)
    assert mgr.phys_id_for(pid1, 0) == 1
    assert mgr.virt_id_for(pid1, 1) == 0

    with pytest.raises(AllocError):
        mgr.allocate(pid1, 1)


if __name__ == "__main__":
    test_alloc_free_0()
    test_alloc_free_0_1()
    test_alloc_non_existing()
    test_alloc_already_allocated()
    test_free_alreay_freed()
    test_get_unmapped_qubit()
    test_alloc_multiple_processes()
    test_alloc_multiple_processes_same_virt_id()
