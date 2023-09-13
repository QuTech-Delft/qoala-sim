from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from qoala.lang.hostlang import (
    AssignCValueOp,
    BasicBlock,
    BasicBlockType,
    ClassicalIqoalaOp,
    IqoalaSingleton,
    IqoalaTuple,
    IqoalaVector,
    ReturnResultOp,
    RunSubroutineOp,
)
from qoala.lang.request import RequestRoutine
from qoala.lang.routine import LocalRoutine


@dataclass
class ProgramMeta:
    name: str
    parameters: List[str]  # list of parameter names (all have type int)
    csockets: Dict[int, str]  # socket ID -> remote node name
    epr_sockets: Dict[int, str]  # socket ID -> remote node name

    @classmethod
    def empty(cls, name: str) -> ProgramMeta:
        return ProgramMeta(name=name, parameters=[], csockets={}, epr_sockets={})

    def serialize(self) -> str:
        s = "META_START"
        s += f"\nname: {self.name}"
        s += f"\nparameters: {', '.join(self.parameters)}"
        s += f"\ncsockets: {', '.join(f'{k} -> {v}' for k,v in self.csockets.items())}"
        s += f"\nepr_sockets: {', '.join(f'{k} -> {v}' for k,v in self.epr_sockets.items())}"
        s += "\nMETA_END"
        return s


class QoalaProgram:
    def __init__(
        self,
        meta: ProgramMeta,
        blocks: List[BasicBlock],
        local_routines: Optional[Dict[str, LocalRoutine]] = None,
        request_routines: Optional[Dict[str, RequestRoutine]] = None,
    ) -> None:
        self._meta: ProgramMeta = meta

        # List to keep order of blocks.
        self._blocks: List[BasicBlock] = blocks
        # Dict to easily find blocks.
        self._block_mapping: Dict[str, BasicBlock] = {blk.name: blk for blk in blocks}
        self._local_routines: Dict[str, LocalRoutine]
        self._request_routines: Dict[str, RequestRoutine]

        if local_routines is None:
            self._local_routines = {}
        else:
            self._local_routines = local_routines

        if request_routines is None:
            self._request_routines = {}
        else:
            self._request_routines = request_routines

    @property
    def meta(self) -> ProgramMeta:
        return self._meta

    @property
    def blocks(self) -> List[BasicBlock]:
        return self._blocks

    @blocks.setter
    def blocks(self, new_blocks: List[BasicBlock]) -> None:
        self._blocks = new_blocks

    def get_block(self, name: str) -> BasicBlock:
        return self._block_mapping[name]

    def get_block_id(self, name: str) -> int:
        return self._blocks.index(self._block_mapping[name])

    @property
    def instructions(self) -> List[ClassicalIqoalaOp]:
        instrs = []
        for b in self.blocks:
            instrs.extend(b.instructions)
        return instrs

    @property
    def local_routines(self) -> Dict[str, LocalRoutine]:
        return self._local_routines

    @local_routines.setter
    def local_routines(self, new_local_routines: Dict[str, LocalRoutine]) -> None:
        self._local_routines = new_local_routines

    @property
    def request_routines(self) -> Dict[str, RequestRoutine]:
        return self._request_routines

    @request_routines.setter
    def request_routines(self, new_routines: Dict[str, RequestRoutine]) -> None:
        self._request_routines = new_routines

    def __str__(self) -> str:
        return "\n".join("  " + str(i) for i in self.instructions)

    def serialize_meta(self) -> str:
        return self.meta.serialize()

    def serialize_block(self, block: BasicBlock) -> str:
        return str(block)

    def serialize_host_code(self) -> str:
        return "\n\n".join(self.serialize_block(b) for b in self.blocks)

    def serialize_subroutines(self) -> str:
        return "\n".join(s.serialize() for s in self.local_routines.values())

    def serialize(self) -> str:
        return (
            self.meta.serialize()
            + "\n"
            + self.serialize_host_code()
            + "\n"
            + self.serialize_subroutines()
        )


class QoalaProgramBuilder:
    @classmethod
    def single_routine(cls, routine: LocalRoutine, args: List[int]) -> QoalaProgram:
        meta = ProgramMeta.empty("name")
        prepare_args: List[ClassicalIqoalaOp] = []
        arg_singletons = [IqoalaSingleton(f"arg_{i}") for i in range(len(args))]

        for i in range(len(args)):
            prepare_args.append(AssignCValueOp(arg_singletons[i], args[i]))

        result_vec = IqoalaVector("result", routine.get_return_size())
        arg_names = [arg.name for arg in arg_singletons]
        args_tup = IqoalaTuple(arg_names)
        call_instr = RunSubroutineOp(
            result=result_vec, values=args_tup, subrt=routine.name
        )
        return_instr = ReturnResultOp(IqoalaSingleton("result"))

        instructions = prepare_args + [call_instr, return_instr]

        block = BasicBlock("b0", typ=BasicBlockType.QL, instructions=instructions)

        return QoalaProgram(
            meta=meta, blocks=[block], local_routines={routine.name: routine}
        )
