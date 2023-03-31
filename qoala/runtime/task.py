from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from netqasm.lang.instr import core

from qoala.lang.ehi import ExposedHardwareInfo, NetworkEhi
from qoala.lang.hostlang import (
    BasicBlockType,
    ReceiveCMsgOp,
    RunRequestOp,
    RunSubroutineOp,
)
from qoala.lang.program import IqoalaProgram
from qoala.lang.routine import LocalRoutine


class TaskExecutionMode(Enum):
    ROUTINE_ATOMIC = 0
    ROUTINE_SPLIT = auto()


@dataclass(eq=True, frozen=True)
class BlockTask:
    pid: int
    block_name: str
    typ: BasicBlockType
    duration: Optional[float] = None
    max_time: Optional[float] = None
    remote_id: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.block_name} ({self.typ.name}), dur={self.duration}"


class TaskCreator:
    def __init__(self, mode: TaskExecutionMode) -> None:
        self._mode = mode

    def from_program(
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
        remote_id: Optional[int] = None,
    ) -> List[BlockTask]:
        if self._mode == TaskExecutionMode.ROUTINE_ATOMIC:
            return self._from_program_routine_atomic(
                program, pid, ehi, network_ehi, remote_id
            )
        else:
            raise NotImplementedError

    def _from_program_routine_atomic(
        self,
        program: IqoalaProgram,
        pid: int,
        ehi: Optional[ExposedHardwareInfo] = None,
        network_ehi: Optional[NetworkEhi] = None,
        remote_id: Optional[int] = None,
    ) -> List[BlockTask]:
        tasks: List[BlockTask] = []

        for block in program.blocks:
            if block.typ == BasicBlockType.CL:
                if ehi is not None:
                    duration = ehi.latencies.host_instr_time * len(block.instructions)
                else:
                    duration = None
                cputask = BlockTask(pid, block.name, block.typ, duration)
                tasks.append(cputask)
            elif block.typ == BasicBlockType.CC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, ReceiveCMsgOp)
                if ehi is not None:
                    duration = ehi.latencies.host_peer_latency
                else:
                    duration = None
                cputask = BlockTask(pid, block.name, block.typ, duration)
                tasks.append(cputask)
            elif block.typ == BasicBlockType.QL:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunSubroutineOp)
                if ehi is not None:
                    local_routine = program.local_routines[instr.subroutine]
                    duration = self._compute_lr_duration(ehi, local_routine)
                else:
                    duration = None
                qputask = BlockTask(pid, block.name, block.typ, duration)
                tasks.append(qputask)
            elif block.typ == BasicBlockType.QC:
                assert len(block.instructions) == 1
                instr = block.instructions[0]
                assert isinstance(instr, RunRequestOp)
                if network_ehi is not None:
                    # TODO: refactor!!
                    epr_time = list(network_ehi.links.values())[0].duration
                    req_routine = program.request_routines[instr.req_routine]
                    duration = epr_time * req_routine.request.num_pairs
                else:
                    duration = None

                qputask = BlockTask(
                    pid, block.name, block.typ, duration, remote_id=remote_id
                )
                tasks.append(qputask)

        return tasks

    def _compute_lr_duration(
        self, ehi: ExposedHardwareInfo, routine: LocalRoutine
    ) -> float:
        duration = 0.0
        # TODO: refactor this
        for instr in routine.subroutine.instructions:
            if (
                type(instr)
                in [
                    core.SetInstruction,
                    core.StoreInstruction,
                    core.LoadInstruction,
                    core.LeaInstruction,
                ]
                or isinstance(instr, core.BranchBinaryInstruction)
                or isinstance(instr, core.BranchUnaryInstruction)
                or isinstance(instr, core.JmpInstruction)
                or isinstance(instr, core.ClassicalOpInstruction)
                or isinstance(instr, core.ClassicalOpModInstruction)
            ):
                duration += ehi.latencies.qnos_instr_time
            else:
                # TODO: can we always use index 0 ??
                if info := ehi.find_single_gate(0, type(instr)):
                    duration += info.duration
                # TODO: can we always use indices 0, 1 ??
                elif info := ehi.find_multi_gate([0, 1], type(instr)):
                    duration += info.duration
                else:
                    raise RuntimeError
        return duration
