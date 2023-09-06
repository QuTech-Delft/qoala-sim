from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from netqasm.lang import operand
from netqasm.lang.encoding import RegisterName
from netqasm.sdk.shared_memory import RegisterGroup, setup_registers

from qoala.lang.request import RequestRoutine
from qoala.lang.routine import LocalRoutine
from qoala.runtime.sharedmem import MemAddr, SharedMemory


class RegisterMeta:
    @classmethod
    def prefixes(cls) -> List[str]:
        return ["R", "C", "Q", "M"]

    @classmethod
    def parse(cls, name: str) -> Tuple[RegisterName, int]:
        assert len(name) >= 2
        assert name[0] in cls.prefixes()
        group = RegisterName[name[0]]
        index = int(name[1:])
        assert index < 16
        return group, index


class HostMemory:
    """Classical program memory only available to the Host.
    Simple mapping from variable names to values."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

        # Host memory is represented as a mapping from variables to values.
        # Variables have a name (str) and values (int).
        self._mem: Dict[str, int] = {}

        # Vectors are stored separately.
        self._vec_mem: Dict[str, List[int]] = {}

    def write(self, loc: str, value: int) -> None:
        self._mem[loc] = value

    def read(self, loc: str) -> int:
        return self._mem[loc]

    def write_vec(self, loc: str, values: List[int]) -> None:
        self._vec_mem[loc] = values

    def read_vec(self, loc: str) -> List[int]:
        return self._vec_mem[loc]


@dataclass
class RunningLocalRoutine:
    routine: LocalRoutine
    params_addr: MemAddr
    result_addr: MemAddr


@dataclass
class RunningRequestRoutine:
    routine: RequestRoutine
    params_addr: MemAddr
    result_addr: MemAddr
    cb_input_addrs: List[MemAddr]
    cb_output_addrs: List[MemAddr]


class QnosMemory:
    """Classical program memory only available to Qnos."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

        # TODO: allow multiple instances of same routine (name)?
        # Currently not possible
        self._running_local_routines: Dict[str, RunningLocalRoutine] = {}
        self._running_request_routines: Dict[str, RunningRequestRoutine] = {}

        # NetQASM registers.
        register_names: Dict[RegisterName, RegisterGroup] = setup_registers()
        self._registers: Dict[RegisterName, Dict[int, int]] = {}
        # TODO fix this abomination of handling registers
        for name in register_names.keys():
            self._registers[name] = {}  # type: ignore
            for i in range(16):
                self._registers[name][i] = 0  # type: ignore

    def add_running_local_routine(self, routine: RunningLocalRoutine) -> None:
        self._running_local_routines[routine.routine.name] = routine

    def get_running_local_routine(self, name: str) -> RunningLocalRoutine:
        return self._running_local_routines[name]

    def get_all_running_local_routines(self) -> Dict[str, RunningLocalRoutine]:
        return self._running_local_routines

    def add_running_request_routine(self, routine: RunningRequestRoutine) -> None:
        self._running_request_routines[routine.routine.name] = routine

    def get_running_request_routine(self, name: str) -> RunningRequestRoutine:
        return self._running_request_routines[name]

    def get_all_running_request_routines(self) -> Dict[str, RunningRequestRoutine]:
        return self._running_request_routines

    def set_reg_value(self, register: Union[str, operand.Register], value: int) -> None:
        if isinstance(register, str):
            name, index = RegisterMeta.parse(register)
        else:
            name, index = register.name, register.index
        self._registers[name][index] = value  # type: ignore

    def get_reg_value(self, register: Union[str, operand.Register]) -> int:
        if isinstance(register, str):
            name, index = RegisterMeta.parse(register)
        else:
            name, index = register.name, register.index
        return self._registers[name][index]  # type: ignore

    # for compatibility with netqasm Futures
    def get_register(self, register: Union[str, operand.Register]) -> Optional[int]:
        return self.get_reg_value(register)


class ProgramMemory:
    """Dynamic runtime memory, divided into
    - Host Memory: local to the Host
    - Qnos Memory: local to Qnos
    - Shared Memory: shared between Host, Qnos and Netstack"""

    def __init__(self, pid: int) -> None:
        self._pid: int = pid

        # TODO: remove pids?
        self._host_memory = HostMemory(pid)
        self._qnos_memory = QnosMemory(pid)
        self._shared_memory = SharedMemory()

    @property
    def host_mem(self) -> HostMemory:
        return self._host_memory

    @property
    def shared_mem(self) -> SharedMemory:
        return self._shared_memory

    @property
    def qnos_mem(self) -> QnosMemory:
        return self._qnos_memory
