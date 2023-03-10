from dataclasses import dataclass
from typing import List, Optional, Union

from qoala.lang.request import EprType


@dataclass
class NetstackCreateRequest:
    # Request parameters.
    remote_id: int
    epr_socket_id: int
    typ: EprType
    num_pairs: int
    fidelity: float
    virt_qubit_ids: List[int]

    # Info for writing results.
    result_array_addr: int


@dataclass
class NetstackReceiveRequest:
    # Request parameters.
    remote_id: int
    epr_socket_id: int
    typ: Optional[EprType]  # not knowable from recv_epr instruction! TODO
    num_pairs: Optional[int]  # not knowable from recv_epr instruction! TODO
    fidelity: float
    virt_qubit_ids: List[int]

    # Info for writing results.
    result_array_addr: int


@dataclass
class NetstackBreakpointCreateRequest:
    pid: int


@dataclass
class NetstackBreakpointReceiveRequest:
    pid: int


T_NetstackRequest = Union[
    NetstackCreateRequest,
    NetstackReceiveRequest,
    NetstackBreakpointCreateRequest,
    NetstackBreakpointReceiveRequest,
]
