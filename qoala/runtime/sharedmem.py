from dataclasses import dataclass
from typing import Any, Dict, Optional

from qoala.runtime.program import (
    CallbackRoutineParams,
    LocalRoutineParams,
    LocalRoutineResult,
    RequestRoutineParams,
    RequestRoutineResult,
)


class SharedMemWriteError(Exception):
    pass


class SharedMemReadError(Exception):
    pass


@dataclass(eq=True, frozen=True)
class MemAddr:
    addr: int


class SharedMemoryRegion:
    """Allocation is represented by adding a key to the _memory dict.
    Writing is represented by writing a value to a _memory dict entry."""

    def __init__(self) -> None:
        self._next_addr: int = 0
        self._memory: Dict[MemAddr, Optional[Any]] = {}

    def allocate(self) -> MemAddr:
        addr = MemAddr(self._next_addr)
        self._next_addr += 1
        self._memory[addr] = None
        return addr

    def write(self, addr: MemAddr, data: Any) -> None:
        if addr not in self._memory:
            raise SharedMemWriteError
        self._memory[addr] = data

    def read(self, addr: MemAddr) -> Any:
        if addr not in self._memory:
            raise SharedMemReadError
        return self._memory[addr]


class SharedMemoryManager:
    def __init__(self) -> None:
        self._rr_in = SharedMemoryRegion()
        self._rr_out = SharedMemoryRegion()
        self._cr_in = SharedMemoryRegion()
        self._lr_in = SharedMemoryRegion()
        self._lr_out = SharedMemoryRegion()

    def allocate_rr_in(self) -> MemAddr:
        return self._rr_in.allocate()

    def write_rr_in(self, addr: MemAddr, params: RequestRoutineParams) -> None:
        assert isinstance(params, RequestRoutineParams)
        self._rr_in.write(addr, params)

    def read_rr_in(self, addr: MemAddr) -> RequestRoutineParams:
        data = self._rr_in.read(addr)
        assert isinstance(data, RequestRoutineParams)
        return data

    def allocate_rr_out(self) -> MemAddr:
        return self._rr_out.allocate()

    def write_rr_out(self, addr: MemAddr, result: RequestRoutineResult) -> None:
        assert isinstance(result, RequestRoutineResult)
        self._rr_out.write(addr, result)

    def read_rr_out(self, addr: MemAddr) -> RequestRoutineResult:
        data = self._rr_out.read(addr)
        assert isinstance(data, RequestRoutineResult)
        return data

    def allocate_cr_in(self) -> MemAddr:
        return self._cr_in.allocate()

    def write_cr_in(self, addr: MemAddr, params: CallbackRoutineParams) -> None:
        assert isinstance(params, CallbackRoutineParams)
        self._cr_in.write(addr, params)

    def read_cr_in(self, addr: MemAddr) -> CallbackRoutineParams:
        data = self._cr_in.read(addr)
        assert isinstance(data, CallbackRoutineParams)
        return data

    def allocate_lr_in(self) -> MemAddr:
        return self._lr_in.allocate()

    def write_lr_in(self, addr: MemAddr, params: LocalRoutineParams) -> None:
        assert isinstance(params, LocalRoutineParams)
        self._lr_in.write(addr, params)

    def read_lr_in(self, addr: MemAddr) -> LocalRoutineParams:
        data = self._lr_in.read(addr)
        assert isinstance(data, LocalRoutineParams)
        return data

    def allocate_lr_out(self) -> MemAddr:
        return self._lr_out.allocate()

    def write_lr_out(self, addr: MemAddr, result: LocalRoutineResult) -> None:
        assert isinstance(result, LocalRoutineResult)
        self._lr_out.write(addr, result)

    def read_lr_out(self, addr: MemAddr) -> LocalRoutineResult:
        data = self._lr_out.read(addr)
        assert isinstance(data, LocalRoutineResult)
        return data
