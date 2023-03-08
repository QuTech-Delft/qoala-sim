from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

from qoala.lang.program import IqoalaRequest, LocalRoutine
from qoala.runtime.program import ProgramInstance, ProgramResult
from qoala.sim.csocket import ClassicalSocket
from qoala.sim.eprsocket import EprSocket
from qoala.sim.memory import HostMemory, ProgramMemory, SharedMemory


@dataclass
class RoutineInstance:
    routine: LocalRoutine

    # TODO: currently not used, since the HostProcessor creates a deepcopy
    # Refactor such that copies are not needed and instead these arguments here
    # are used.
    arguments: Dict[str, int]


@dataclass
class IqoalaProcess:
    prog_instance: ProgramInstance

    # Mutable
    prog_memory: ProgramMemory
    result: ProgramResult

    active_routines: Dict[str, RoutineInstance]

    # Immutable
    csockets: Dict[int, ClassicalSocket]
    epr_sockets: Dict[int, EprSocket]

    def get_local_routine(self, name: str) -> LocalRoutine:
        return self.prog_instance.program.local_routines[name]

    def get_all_local_routines(self) -> Dict[str, LocalRoutine]:
        return self.prog_instance.program.local_routines

    def get_request(self, name: str) -> IqoalaRequest:
        return self.prog_instance.program.requests[name]

    def get_all_requests(self) -> Dict[str, IqoalaRequest]:
        return self.prog_instance.program.requests

    def instantiate_routine(
        self, name: str, pid: int, arguments: Dict[str, int]
    ) -> None:
        routine = self.get_local_routine(name)

        # Create a copy of the routine in which we can resolve templates.
        instance = RoutineInstance(deepcopy(routine), arguments)
        instance.routine.subroutine.instantiate(pid, arguments)
        self.active_routines[name] = instance

    def get_active_routine(self, name: str) -> RoutineInstance:
        return self.active_routines[name]

    @property
    def pid(self) -> int:
        return self.prog_instance.pid  # type: ignore

    @property
    def host_mem(self) -> HostMemory:
        return self.prog_memory.host_mem

    @property
    def shared_mem(self) -> SharedMemory:
        return self.prog_memory.shared_mem
