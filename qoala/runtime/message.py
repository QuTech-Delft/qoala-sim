from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    content: Any


@dataclass
class LrCallTuple:
    routine_name: str
    input_addr: int
    result_addr: int
