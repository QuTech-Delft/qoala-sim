from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from netqasm.lang.operand import Template

from qoala.lang.hostlang import IqoalaVector


class CallbackType(Enum):
    """
    Callbacks can be called in two ways:
    - SEQUENTIAL = The callback is called after each pair is generated.
    - WAIT_ALL = The callback is called after all pairs are generated.
    """

    SEQUENTIAL = 0
    WAIT_ALL = auto()


class EprType(Enum):
    """
    EPR Pairs can be handled in different ways after creations:
    - CREATE_KEEP = Creates the pairs and keeps them.
    - MEASURE_DIRECTLY = Measure pairs directly after the creation.
    - REMOTE_STATE_PREP = ??? TODO
    """

    CREATE_KEEP = 0
    MEASURE_DIRECTLY = auto()
    REMOTE_STATE_PREP = auto()


class EprRole(Enum):
    """
    TODO
    """

    CREATE = 0
    RECEIVE = auto()


class VirtIdMappingType(Enum):
    """
    Virtual IDs of EPR Pairs can be mapped to different values:
    - EQUAL = All virt IDs have the same single value. E.g. if single_value = 0 and num_pairs = 4,
        virt IDs are [0, 0, 0, 0]
    - INCREMENT = Virt IDs are increasing sequence starting at given value. E.g. if single_value = 0
        and num_pairs = 4, virt IDs are [0, 1, 2, 3]
    - CUSTOM = Explicit list of virt IDs used. Length needs to be equal to num_pairs.
    """

    EQUAL = 0
    INCREMENT = auto()
    CUSTOM = auto()


@dataclass
class RequestVirtIdMapping:
    """
    Virtual Id mapping of the Request. Template values are not allowed for "custom_values".

    :param typ: Type of the mapping.
    :param single_value: Single value used for mapping. When typ == EQUAL every qubit has this value.
    When typ == INCREMENT, every qubit has a different value starting at this value.
    :param custom_values: List of custom values used for mapping. Only used if typ == CUSTOM.
    """

    typ: VirtIdMappingType
    # Only allow templates with "all" or "increment", not with "custom"
    single_value: Optional[Union[Template, int]]
    custom_values: Optional[List[int]]

    def get_id(self, index: int) -> int:
        """
        Returns the virtual ID for the given index.
        :param index: Index of the qubit.

        :return: Virtual ID of the qubit.
        """
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
        """
        Creates a RequestVirtIdMapping from a string. Text must start with a keyword "all", "increment" or "custom".
        Then the mapping is parsed accordingly.

        Examples:
        all 0 -> Virtual IDs will be 0, 0, 0, ...
        increment 0 -> Virtual IDs will be 0, 1, 2, ...
        custom 0, 1, 2, 3 -> Virtual IDs will be 0, 1, 2, 3

        Also Template values can be used for "all" and "increment":
        all {x} -> Virtual IDs will be x, x, x, ...
        increment {x} -> Virtual IDs will be x, x+1, x+2, ...

        :param text: String representation of the mapping.

        :return: Specified RequestVirtIdMapping.
        """
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


@dataclass
class QoalaRequest:
    """
    Request for EPR Pairs. Contains the information about the EPR Pair generation.

    :param name: Name of the request.
    :param remote_id: ID of the remote node.
    :param epr_socket_id: ID of the EPR Socket.
    :param num_pairs: Number of EPR Pairs to create.
    :param virt_ids: Virtual ID mapping of the EPR Pairs.
    :param timeout: Timeout for the request.
    :param fidelity: Fidelity of the EPR Pairs. It takes values between 0 and 1. If it is 1, the EPR Pairs are
    maximally entangled.
    :param typ: Type of the EPR pair generation.
    :param role: Role of the node in the EPR pair generation.
    """

    name: str  # TODO: remove?
    remote_id: Union[int, Template]
    epr_socket_id: Union[int, Template]
    num_pairs: Union[int, Template]
    virt_ids: RequestVirtIdMapping
    timeout: Union[float, Template]
    fidelity: Union[float, Template]
    typ: EprType
    role: EprRole

    def instantiate(self, values: Dict[str, Any]) -> None:
        """
        Instantiates all templates in the request with the given values.

        :param values: Values to instantiate the templates with. Every template in the request must be in the values.
        """
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


@dataclass
class RequestRoutine:
    """
    Routine for EPR Pair generation. Contains the all information about the EPR Pair generation. Addition to the
    QoalaRequest it contains the return variables and the callback information.

    :param name: Name of the routine.
    :param request: Request for the EPR Pair generation.
    :param return_vars: Return variables of the routine.
    :param callback_type: Type of the callback.
    :param callback: Name of the callback. It is the name of the local routine.
    """

    name: str
    request: QoalaRequest

    return_vars: List[Union[str, IqoalaVector]]

    callback_type: CallbackType
    callback: Optional[str]  # Local Routine name

    def instantiate(self, values: Dict[str, Any]) -> None:
        """
        Instantiates all templates in the routine with the given values.

        :param values: Values to instantiate the templates with. Every template in the routine must be in the values.
        """
        for i in range(len(self.return_vars)):
            ret_var = self.return_vars[i]
            if isinstance(ret_var, IqoalaVector):
                if isinstance(ret_var.size, Template):
                    size = values[ret_var.size.name]
                    # print(f"instantiating ret_var {ret_var.name} with size {size}")
                    self.return_vars[i] = IqoalaVector(ret_var.name, size)
        self.request.instantiate(values)

    def get_return_size(self, prog_input: Optional[Dict[str, int]] = None) -> int:
        """
        Returns the size of the return vector of this routine. This is the sum of the sizes of all return variables.
        Strings have size 1, vectors have size equal to their size.

        :return: Combined size of the return variables.
        """
        size = 0
        for v in self.return_vars:
            if isinstance(v, IqoalaVector):
                if isinstance(v.size, int):
                    size += v.size
                elif isinstance(v.size, Template):
                    # Size is a template. It should be in the Program Input.
                    assert prog_input is not None
                    size += prog_input[v.size.name]
            else:
                size += 1
        return size

    def serialize(self) -> str:
        s = f"REQUEST {self.name}"
        s += f"\ncallback_type: {self.callback_type.name}"
        s += f"\ncallback: {str(self.callback or '')}"
        s += f"\nreturn_vars: {', '.join(str(v) for v in self.return_vars)}"
        s += f"\nremote_id: {self.request.remote_id}"
        s += f"\nepr_socket_id: {self.request.epr_socket_id}"
        s += f"\nnum_pairs: {self.request.num_pairs}"
        s += f"\nvirt_ids: {self.request.virt_ids}"
        s += f"\ntimeout: {self.request.timeout}"
        s += f"\nfidelity: {self.request.fidelity}"
        s += f"\ntyp: {self.request.typ.name}"
        s += f"\nrole: {self.request.role.name}"
        return s

    def __str__(self) -> str:
        return self.serialize()
