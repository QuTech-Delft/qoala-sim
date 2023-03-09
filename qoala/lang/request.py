from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from qoala.sim.requests import EprCreateRole, T_NetstackRequest


@dataclass
class RequestMemoryUsage:
    mapping: Dict[int, int]  # pair index -> virt qubit ID

    @classmethod
    def all_in(cls, num_pairs: int, virt_id: int) -> RequestMemoryUsage:
        return RequestMemoryUsage(mapping={i: virt_id for i in range(num_pairs)})


class IqoalaRequest:
    def __init__(
        self,
        name: str,
        role: EprCreateRole,
        request: T_NetstackRequest,
        mem_usage: Optional[RequestMemoryUsage] = None,
    ) -> None:
        self._name = name
        self._role = role
        self._request = request
        if mem_usage is None:
            mem_usage = RequestMemoryUsage.all_in(request.num_pairs, 0)
        self._mem_usage = mem_usage

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> EprCreateRole:
        return self._role

    @property
    def request(self) -> T_NetstackRequest:
        return self._request

    def serialize(self) -> str:
        s = f"REQUEST {self.name}"
        s += f"role: {self.role}"
        s += f"remote_id: {self.request.remote_id}"
        s += f"epr_socket_id: {self.request.epr_socket_id}"
        s += f"typ: {self.request.typ.name}"
        s += f"num_pairs: {self.request.num_pairs}"
        s += f"fidelity: {self.request.fidelity}"
        s += f"virt_qubit_ids: {','.join(self.request.virt_qubit_ids)}"
        s += f"result_array_addr: {self.request.result_array_addr}"
        return s

    def __str__(self) -> str:
        return self.serialize()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IqoalaRequest):
            return NotImplemented
        return (
            self.name == other.name
            and self.role == other.role
            and self.request == other.request
        )
