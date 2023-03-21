from dataclasses import dataclass
from typing import Any

from qoala.runtime.sharedmem import MemAddr


@dataclass
class Message:
    content: Any


@dataclass
class LrCallTuple:
    routine_name: str
    input_addr: MemAddr
    result_addr: MemAddr
