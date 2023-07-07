from dataclasses import dataclass
from typing import Dict, List, Optional


class SharedMemWriteError(Exception):
    pass


class SharedMemReadError(Exception):
    pass


class SharedMemNotAllocatedError(Exception):
    pass


class SharedMemIllegalRegionError(Exception):
    pass


@dataclass(frozen=True)
class MemAddr:
    addr: int


class NetQASMArrays:
    def __init__(self) -> None:
        self._next_addr: int = 0
        self._memory: Dict[MemAddr, List[Optional[int]]] = {}

    @property
    def raw_memory(self) -> Dict[MemAddr, List[Optional[int]]]:
        return self._memory

    def allocate(self, addr: MemAddr, size: int) -> None:
        assert addr not in self._memory
        self._memory[addr] = [None] * size

    def write(self, addr: MemAddr, data: List[int], offset: int) -> None:
        if addr not in self._memory:
            raise SharedMemWriteError(f"Address {addr} is not in the memory.")
        if len(self._memory[addr]) < offset + len(data):
            raise SharedMemWriteError(
                f"Address size of the data + offset is larger than the \
                allocated space in the memory for address {addr}."
            )
        for i in range(len(data)):
            self._memory[addr][i + offset] = data[i]

    def read(self, addr: MemAddr, size: int, offset: int) -> List[int]:
        if addr not in self._memory:
            raise SharedMemReadError(f"Address {addr} is not in the memory.")
        if len(self._memory[addr]) < offset + size:
            raise SharedMemReadError(
                f"Size + offset is larger than the \
                allocated space in the memory for address {addr}."
            )
        array = self._memory[addr][offset : offset + size]
        assert all(v is not None for v in array)
        return array  # type: ignore


class SharedMemory:
    def __init__(self) -> None:
        self._arrays = NetQASMArrays()

        self._rr_in_addrs: List[MemAddr] = []
        self._rr_out_addrs: List[MemAddr] = []
        self._cr_in_addrs: List[MemAddr] = []
        self._lr_in_addrs: List[MemAddr] = []
        self._lr_out_addrs: List[MemAddr] = []

        self._addr_counter: int = 0

    @property
    def raw_arrays(self) -> NetQASMArrays:
        return self._arrays

    def _allocate(self, size: int) -> MemAddr:
        addr = MemAddr(self._addr_counter)
        self._addr_counter += 1
        self._arrays.allocate(addr, size)
        return addr

    def allocate_rr_in(self, size: int) -> MemAddr:
        addr = self._allocate(size)
        self._rr_in_addrs.append(addr)
        return addr

    def write_rr_in(self, addr: MemAddr, data: List[int], offset: int = 0) -> None:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._rr_in_addrs:
            raise SharedMemIllegalRegionError
        self._arrays.write(addr, data, offset)

    def read_rr_in(self, addr: MemAddr, size: int, offset: int = 0) -> List[int]:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._rr_in_addrs:
            raise SharedMemIllegalRegionError
        data = self._arrays.read(addr, size, offset)
        return data

    def allocate_rr_out(self, size: int) -> MemAddr:
        addr = self._allocate(size)
        self._rr_out_addrs.append(addr)
        return addr

    def write_rr_out(self, addr: MemAddr, data: List[int], offset: int = 0) -> None:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._rr_out_addrs:
            raise SharedMemIllegalRegionError
        self._arrays.write(addr, data, offset)

    def read_rr_out(self, addr: MemAddr, size: int, offset: int = 0) -> List[int]:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._rr_out_addrs:
            raise SharedMemIllegalRegionError
        data = self._arrays.read(addr, size, offset)
        return data

    def allocate_cr_in(self, size: int) -> MemAddr:
        addr = self._allocate(size)
        self._cr_in_addrs.append(addr)
        return addr

    def write_cr_in(self, addr: MemAddr, data: List[int], offset: int = 0) -> None:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._cr_in_addrs:
            raise SharedMemIllegalRegionError
        self._arrays.write(addr, data, offset)

    def read_cr_in(self, addr: MemAddr, size: int, offset: int = 0) -> List[int]:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._cr_in_addrs:
            raise SharedMemIllegalRegionError
        data = self._arrays.read(addr, size, offset)
        return data

    def allocate_lr_in(self, size: int) -> MemAddr:
        addr = self._allocate(size)
        self._lr_in_addrs.append(addr)
        return addr

    def _check_allocated(self, addr: MemAddr) -> bool:
        # "hack": make use of fact that addr counter is never decreased
        return addr.addr < self._addr_counter

    def write_lr_in(self, addr: MemAddr, data: List[int], offset: int = 0) -> None:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._lr_in_addrs:
            raise SharedMemIllegalRegionError
        self._arrays.write(addr, data, offset)

    def read_lr_in(self, addr: MemAddr, size: int, offset: int = 0) -> List[int]:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._lr_in_addrs:
            raise SharedMemIllegalRegionError
        data = self._arrays.read(addr, size, offset)
        return data

    def allocate_lr_out(self, size: int) -> MemAddr:
        addr = self._allocate(size)
        self._lr_out_addrs.append(addr)
        return addr

    def write_lr_out(self, addr: MemAddr, data: List[int], offset: int = 0) -> None:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._lr_out_addrs:
            raise SharedMemIllegalRegionError
        self._arrays.write(addr, data, offset)

    def read_lr_out(self, addr: MemAddr, size: int, offset: int = 0) -> List[int]:
        if not self._check_allocated(addr):
            raise SharedMemNotAllocatedError
        if addr not in self._lr_out_addrs:
            raise SharedMemIllegalRegionError
        data = self._arrays.read(addr, size, offset)
        return data
