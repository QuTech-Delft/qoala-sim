from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

from qoala.sim.requests import EprType


class CallbackType(Enum):
    SEQUENTIAL = 0
    WAIT_ALL = auto()


@dataclass
class RequestRoutine:
    request: IqoalaRequest
    callback_type: CallbackType
    callback: Optional[str]  # Local Routine name


class EprRole(Enum):
    CREATE = 0
    RECEIVE = auto()


@dataclass(eq=True, frozen=True)
class IqoalaRequest:
    name: str
    remote_id: int
    epr_socket_id: int
    num_pairs: int
    virt_ids: List[int]
    timeout: float
    fidelity: float
    typ: EprType
    role: EprRole
    result_array_addr: int  # TODO remove when implementing proper shared memory

    def serialize(self) -> str:
        s = f"REQUEST {self.name}"
        s += f"remote_id: {self.remote_id}"
        s += f"epr_socket_id: {self.epr_socket_id}"
        s += f"num_pairs: {self.num_pairs}"
        s += f"virt_ids: {','.join(self.virt_ids)}"
        s += f"timeout: {self.timeout}"
        s += f"fidelity: {self.fidelity}"
        s += f"typ: {self.typ.name}"
        s += f"role: {self.role}"
        s += f"result_array_addr: {self.result_array_addr}"
        return s

    def __str__(self) -> str:
        return self.serialize()
