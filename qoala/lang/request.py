from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from netqasm.lang.operand import Template


class CallbackType(Enum):
    SEQUENTIAL = 0
    WAIT_ALL = auto()


class EprType(Enum):
    CREATE_KEEP = 0
    MEASURE_DIRECTLY = auto()
    REMOTE_STATE_PREP = auto()


class EprRole(Enum):
    CREATE = 0
    RECEIVE = auto()


class VirtIdMappingType(Enum):
    # All virt IDs have the same single value.
    # E.g. if virt_ids_equal = 0, and num_pairs = 4,
    # virt IDs are [0, 0, 0, 0]
    EQUAL = 0

    # Virt IDs are increasing sequence starting at given value.
    # E.g. if virt_ids_increment = 0, and num_pairs = 4,
    # virt IDs are [0, 1, 2, 3]
    INCREMENT = auto()

    # Explicit list of virt IDs used. Length needs to be equal to num_pairs.
    CUSTOM = auto()


@dataclass(eq=True, frozen=True)
class RequestVirtIdMapping:
    typ: VirtIdMappingType
    # Only allow templates with "all" or "increment", not with "custom"
    single_value: Optional[Union[Template, int]]
    custom_values: Optional[List[int]]

    def __str__(self) -> str:
        if self.typ == VirtIdMappingType.EQUAL:
            return f"all {self.single_value}"
        elif self.typ == VirtIdMappingType.INCREMENT:
            return f"increment {self.single_value}"
        elif self.typ == VirtIdMappingType.CUSTOM:
            return f"custom {', '.join(str(v) for v in self.custom_values)}"

    @classmethod
    def from_str(cls, text: str) -> RequestVirtIdMapping:
        if text.startswith("all "):
            value = text[4:]
            if value.startswith("{") and value.endswith("}"):
                value = value.strip("{}").strip()
                value = Template(value)
            else:
                value = int(value)
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.EQUAL, single_value=value, custom_values=None
            )
        elif text.startswith("increment "):
            value = text[10:]
            if value.startswith("{") and value.endswith("}"):
                value = value.strip("{}").strip()
                value = Template(value)
            else:
                value = int(value)
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.INCREMENT, single_value=value, custom_values=None
            )
        elif text.startswith("custom "):
            int_list = text[7:]
            ints = [int(i) for i in int_list.split(", ")]
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.CUSTOM, single_value=None, custom_values=ints
            )


@dataclass(eq=True)
class IqoalaRequest:
    name: str
    remote_id: Union[int, Template]
    epr_socket_id: Union[int, Template]
    num_pairs: Union[int, Template]
    virt_ids: RequestVirtIdMapping
    timeout: float
    fidelity: float
    typ: EprType
    role: EprRole
    result_array_addr: int  # TODO remove when implementing proper shared memory

    def instantiate(self, values: Dict[str, Any]) -> None:
        if isinstance(self.remote_id, Template):
            self.remote_id = values[self.remote_id.name]
        if isinstance(self.epr_socket_id, Template):
            self.epr_socket_id = values[self.epr_socket_id.name]
        if isinstance(self.num_pairs, Template):
            self.num_pairs = values[self.num_pairs.name]
        if isinstance(self.virt_ids.single_value, Template):
            # Only need to check single_value. If "custom", singe_value is None,
            # and custom values themselves are never Templates.
            self.virt_ids.single_value = values[self.virt_ids.single_value.name]
        if isinstance(self.timeout, Template):
            self.timeout = values[self.timeout.name]
        if isinstance(self.fidelity, Template):
            self.fidelity = values[self.fidelity.name]

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


@dataclass
class RequestRoutine:
    request: IqoalaRequest
    callback_type: CallbackType
    callback: Optional[str]  # Local Routine name
