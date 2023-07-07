from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from qoala.runtime.sharedmem import MemAddr


@dataclass
class Message:
    src_pid: int
    dst_pid: int
    content: Any


@dataclass
class LrCallTuple:
    routine_name: str
    input_addr: MemAddr
    result_addr: MemAddr


@dataclass
class RrCallTuple:
    routine_name: str
    input_addr: MemAddr
    result_addr: MemAddr
    cb_input_addrs: List[MemAddr]
    cb_output_addrs: List[MemAddr]

    @classmethod
    def no_alloc(cls, name: str) -> RrCallTuple:
        return RrCallTuple(name, MemAddr(0), MemAddr(0), [], [])
