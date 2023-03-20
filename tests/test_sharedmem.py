import pytest

from qoala.runtime.program import LocalRoutineParams
from qoala.runtime.sharedmem import MemAddr, SharedMemoryManager, SharedMemWriteError


def test1():
    mgr = SharedMemoryManager()

    params = LocalRoutineParams()

    # Not allocated yet
    with pytest.raises(SharedMemWriteError):
        mgr.write_lr_in(MemAddr(0), params)

    addr = mgr.allocate_lr_in()
    mgr.write_lr_in(addr, params)
    assert len(mgr._lr_in._memory) == 1
    assert mgr._lr_in._memory[addr] == params

    assert mgr.read_lr_in(addr) == params

    # Wrong type
    with pytest.raises(AssertionError):
        mgr.write_cr_in(addr, params)


if __name__ == "__main__":
    test1()
