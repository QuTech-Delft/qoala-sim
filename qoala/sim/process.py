from dataclasses import dataclass
from typing import Dict

from qoala.lang.program import QoalaProgram
from qoala.lang.request import RequestRoutine
from qoala.lang.routine import LocalRoutine
from qoala.runtime.memory import HostMemory, ProgramMemory, QnosMemory
from qoala.runtime.program import ProgramInput, ProgramInstance, ProgramResult
from qoala.runtime.sharedmem import SharedMemory
from qoala.sim.eprsocket import EprSocket
from qoala.sim.host.csocket import ClassicalSocket


@dataclass
class QoalaProcess:
    prog_instance: ProgramInstance

    # Mutable
    prog_memory: ProgramMemory
    result: ProgramResult

    # Immutable
    csockets: Dict[int, ClassicalSocket]
    epr_sockets: Dict[int, EprSocket]

    def get_local_routine(self, name: str) -> LocalRoutine:
        return self.program.local_routines[name]

    def get_all_local_routines(self) -> Dict[str, LocalRoutine]:
        return self.program.local_routines

    def get_request_routine(self, name: str) -> RequestRoutine:
        return self.program.request_routines[name]

    def get_all_request_routines(self) -> Dict[str, RequestRoutine]:
        return self.program.request_routines

    @property
    def pid(self) -> int:
        return self.prog_instance.pid  # type: ignore

    @property
    def program(self) -> QoalaProgram:
        return self.prog_instance.program

    @property
    def inputs(self) -> ProgramInput:
        return self.prog_instance.inputs

    @property
    def host_mem(self) -> HostMemory:
        return self.prog_memory.host_mem

    @property
    def qnos_mem(self) -> QnosMemory:
        return self.prog_memory.qnos_mem

    @property
    def shared_mem(self) -> SharedMemory:
        return self.prog_memory.shared_mem
