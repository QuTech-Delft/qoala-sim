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
    # E.g. if single_value = 0 and num_pairs = 4,
    # virt IDs are [0, 0, 0, 0]
    EQUAL = 0

    # Virt IDs are increasing sequence starting at given value.
    # E.g. if single_value = 0 and num_pairs = 4,
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

    def get_id(self, index: int) -> int:
        if self.typ == VirtIdMappingType.EQUAL:
            assert isinstance(self.single_value, int)
            return self.single_value
        elif self.typ == VirtIdMappingType.INCREMENT:
            assert isinstance(self.single_value, int)
            return self.single_value + index
        elif self.typ == VirtIdMappingType.CUSTOM:
            assert self.custom_values is not None
            return self.custom_values[index]
        raise ValueError

    def __str__(self) -> str:
        if self.typ == VirtIdMappingType.EQUAL:
            return f"all {self.single_value}"
        elif self.typ == VirtIdMappingType.INCREMENT:
            return f"increment {self.single_value}"
        elif self.typ == VirtIdMappingType.CUSTOM:
            assert self.custom_values is not None
            return f"custom {', '.join(str(v) for v in self.custom_values)}"
        raise ValueError

    @classmethod
    def from_str(cls, text: str) -> RequestVirtIdMapping:
        if text.startswith("all "):
            value_str = text[4:]
            if value_str.startswith("{") and value_str.endswith("}"):
                value_str = value_str.strip("{}").strip()
                value = Template(value_str)
            else:
                value = int(value_str)
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.EQUAL, single_value=value, custom_values=None
            )
        elif text.startswith("increment "):
            value_str = text[10:]
            if value_str.startswith("{") and value_str.endswith("}"):
                value_str = value_str.strip("{}").strip()
                value = Template(value_str)
            else:
                value = int(value_str)
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.INCREMENT, single_value=value, custom_values=None
            )
        elif text.startswith("custom "):
            int_list = text[7:]
            ints = [int(i) for i in int_list.split(", ")]
            return RequestVirtIdMapping(
                typ=VirtIdMappingType.CUSTOM, single_value=None, custom_values=ints
            )
        raise ValueError


@dataclass(eq=True)
class QoalaRequest:
    name: str  # TODO: remove?
    remote_id: Union[int, Template]
    epr_socket_id: Union[int, Template]
    num_pairs: Union[int, Template]
    virt_ids: RequestVirtIdMapping
    timeout: float
    fidelity: float
    typ: EprType
    role: EprRole

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
            self.virt_ids.single_value = values[self.virt_ids.single_value.name]  # type: ignore
        if isinstance(self.timeout, Template):
            self.timeout = values[self.timeout.name]
        if isinstance(self.fidelity, Template):
            self.fidelity = values[self.fidelity.name]

    def serialize(self) -> str:
        s = f"REQUEST {self.name}"
        s += f"remote_id: {self.remote_id}"
        s += f"epr_socket_id: {self.epr_socket_id}"
        s += f"num_pairs: {self.num_pairs}"
        s += f"virt_ids: {self.virt_ids}"
        s += f"timeout: {self.timeout}"
        s += f"fidelity: {self.fidelity}"
        s += f"typ: {self.typ.name}"
        s += f"role: {self.role}"
        return s

    def __str__(self) -> str:
        return self.serialize()


@dataclass(eq=True, frozen=True)
class RrReturnVector:
    name: str
    size: Union[int, Template]

    def __str__(self) -> str:
        return f"{self.name}<{self.size}>"


@dataclass
class RequestRoutine:
    name: str
    request: QoalaRequest

    return_vars: List[Union[str, RrReturnVector]]

    callback_type: CallbackType
    callback: Optional[str]  # Local Routine name

    def get_return_size(self) -> int:
        size = 0
        for v in self.return_vars:
            if isinstance(v, RrReturnVector):
                assert isinstance(v.size, int)
                size += v.size
            else:
                size += 1
        return size
