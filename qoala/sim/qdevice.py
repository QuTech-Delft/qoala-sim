from dataclasses import dataclass
from typing import Generator, List, Optional, Set

from netsquid.components.instructions import INSTR_INIT, Instruction
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.nodes import Node
from netsquid.qubits.qubit import Qubit

from pydynaa import EventExpression
from qoala.runtime.lhi import LhiTopology


class UnsupportedQDeviceCommandError(Exception):
    pass


class NonInitializedQubitError(Exception):
    pass


@dataclass(frozen=True)
class QDeviceCommand:
    instr: Instruction
    indices: List[int]
    angle: Optional[float] = None


class QDevice:
    def __init__(self, node: Node, topology: LhiTopology) -> None:
        self._node = node
        self._topology = topology

    @property
    def qprocessor(self) -> QuantumProcessor:
        """Get the NetSquid `QuantumProcessor` object of this node."""
        return self._node.qmemory

    @property
    def node(self) -> Node:
        """Get the NetSquid `Node` object of this QDevice."""
        return self._node

    @property
    def topology(self) -> LhiTopology:
        return self._topology

    def get_qubit_count(self) -> int:
        return len(self.get_all_qubit_ids())

    def get_comm_qubit_count(self) -> int:
        return len(self.get_comm_qubit_ids())

    def get_non_comm_qubit_count(self) -> int:
        return len(self.get_non_comm_qubit_ids())

    def get_all_qubit_ids(self) -> Set[int]:
        return {q for q, _ in self.topology.qubit_infos.items()}

    def get_comm_qubit_ids(self) -> Set[int]:
        return {
            q for q, info in self.topology.qubit_infos.items() if info.is_communication
        }

    def get_non_comm_qubit_ids(self) -> Set[int]:
        return {
            q
            for q, info in self.topology.qubit_infos.items()
            if not info.is_communication
        }

    def is_allowed(self, cmd: QDeviceCommand) -> bool:
        all_phys_instructions = self.qprocessor.get_physical_instructions()

        # Get the physical instruction with the same type ('gate').
        matches = [
            i for i in all_phys_instructions if i.instruction.name == cmd.instr.name
        ]
        # Should be at least one matching.
        if len(matches) == 0:
            return False

        for phys_instr in matches:
            # If there is no topology, this instruction is allowed on any qubit.
            if phys_instr.topology is None:
                return True

            # Else, check if it is allowed for our current qubit(s).
            if len(cmd.indices) == 1:
                if cmd.indices[0] in phys_instr.topology:
                    return True
            else:
                if (cmd.indices[0], cmd.indices[1]) in phys_instr.topology:
                    return True

        # We didn't find any matching instruction.
        return False

    def set_mem_pos_in_use(self, id: int, in_use: bool) -> None:
        self.qprocessor.mem_positions[id].in_use = in_use

    def execute_commands(
        self, commands: List[QDeviceCommand]
    ) -> Generator[EventExpression, None, Optional[int]]:
        """Can only return at most 1 measurement result."""
        prog = QuantumProgram()

        # TODO: rewrite this abomination

        for cmd in commands:
            # Check if this instruction is allowed on this processor.
            # If not, NetSquid will just silently skip this instruction which is confusing.
            if not self.is_allowed(cmd):
                raise UnsupportedQDeviceCommandError(cmd)

        for cmd in commands:
            # Check if the qubit has been initialized, since instructions won't work
            # if this is not the case.
            # Anything after an INSTR_INIT instruction is fine.
            # TODO: better logic for detecting INITs of individual qubits.
            if cmd.instr == INSTR_INIT:
                break
            for index in cmd.indices:
                if self.get_local_qubit(index) is None:
                    raise NonInitializedQubitError

        for cmd in commands:
            if cmd.angle is not None:
                prog.apply(cmd.instr, qubit_indices=cmd.indices, angle=cmd.angle)
            else:
                prog.apply(cmd.instr, qubit_indices=cmd.indices)
        yield self.qprocessor.execute_program(prog)

        last_result = prog.output["last"]
        if last_result is not None:
            meas_outcome: int = last_result[0]
            return meas_outcome
        return None

    def execute_program(
        self, prog: QuantumProgram
    ) -> Generator[EventExpression, None, None]:
        raise DeprecationWarning

    def get_local_qubit(self, index: int) -> Qubit:
        return self.qprocessor.peek([index])[0]

    def get_local_qubits(self, indices: List[int]) -> List[Qubit]:
        return self.qprocessor.peek(indices)  # type: ignore
