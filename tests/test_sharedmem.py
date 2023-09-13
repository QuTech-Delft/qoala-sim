import pytest

from qoala.runtime.sharedmem import (
    MemAddr,
    SharedMemIllegalRegionError,
    SharedMemNotAllocatedError,
    SharedMemory,
    SharedMemReadError,
)


def test1():
    mgr = SharedMemory()

    params = [0, 1]

    # Not allocated yet
    with pytest.raises(SharedMemNotAllocatedError):
        mgr.write_lr_in(MemAddr(0), params)

    addr = mgr.allocate_lr_in(2)
    mgr.write_lr_in(addr, params)
    assert len(mgr._arrays._memory) == 1
    assert mgr._arrays._memory[addr] == params

    assert mgr._lr_in_addrs == [addr]

    assert mgr.read_lr_in(addr, 2) == params

    assert mgr.read_lr_in(addr, 1) == params[0:1]
    assert mgr.read_lr_in(addr, 1, offset=1) == params[1:]

    # Beyond end of array
    with pytest.raises(SharedMemReadError):
        assert mgr.read_lr_in(addr, 1, offset=2)

    # Too long
    with pytest.raises(SharedMemReadError):
        assert mgr.read_lr_in(addr, 3, offset=0)

    # Wrong type
    with pytest.raises(SharedMemIllegalRegionError):
        mgr.write_cr_in(addr, params)

    addr2 = mgr.allocate_rr_in(3)
    assert len(mgr._arrays._memory) == 2
    params2 = [0, 1, 2]

    with pytest.raises(SharedMemIllegalRegionError):
        assert mgr.write_lr_in(addr2, params2)

    mgr.write_rr_in(addr2, params2)

    assert mgr._arrays._memory[addr2] == params2

    assert mgr._rr_in_addrs == [addr2]

    assert mgr._lr_in_addrs == [addr]
    assert mgr._addr_counter == 2

    assert mgr.read_rr_in(addr2, 3) == params2

    assert mgr.read_rr_in(addr2, 2) == params2[0:2]
    assert mgr.read_rr_in(addr2, 2, offset=1) == params2[1:]

    with pytest.raises(SharedMemIllegalRegionError):
        assert mgr.read_lr_in(addr2, 3, offset=0)


if __name__ == "__main__":
    test1()
