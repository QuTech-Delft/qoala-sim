from dataclasses import dataclass
from typing import Dict

from qoala.lang.iqoala import IqoalaRequest, IqoalaSubroutine
from qoala.runtime.program import ProgramInstance, ProgramResult
from qoala.sim.csocket import ClassicalSocket
from qoala.sim.eprsocket import EprSocket
from qoala.sim.memory import HostMemory, ProgramMemory, SharedMemory


@dataclass
class IqoalaProcess:
    prog_instance: ProgramInstance

    # Mutable
    prog_memory: ProgramMemory
    result: ProgramResult

    # Immutable
    csockets: Dict[int, ClassicalSocket]
    epr_sockets: Dict[int, EprSocket]
    subroutines: Dict[str, IqoalaSubroutine]
    requests: Dict[str, IqoalaRequest]

    @property
    def pid(self) -> int:
        return self.prog_instance.pid  # type: ignore

    @property
    def host_mem(self) -> HostMemory:
        return self.prog_memory.host_mem

    @property
    def shared_mem(self) -> SharedMemory:
        return self.prog_memory.shared_mem
