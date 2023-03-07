from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from netqasm.lang.instr import NetQASMInstruction

from qoala.lang.hostlang import (
    ClassicalIqoalaOp,
    IqoalaInstructionSignature,
    IqoalaInstructionType,
    RunSubroutineOp,
)
from qoala.lang.request import IqoalaRequest
from qoala.lang.routine import IqoalaSubroutine


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


class StaticIqoalaProgramInfo:
    pass


class DynamicIqoalaProgramInfo:
    pass


def netqasm_instr_to_type(instr: NetQASMInstruction) -> IqoalaInstructionType:
    if instr.mnemonic in ["create_epr", "recv_epr"]:
        return IqoalaInstructionType.QC
    else:
        return IqoalaInstructionType.QL


class IqoalaProgram:
    def __init__(
        self,
        instructions: List[ClassicalIqoalaOp],
        local_routines: Dict[str, IqoalaSubroutine],
        meta: ProgramMeta,
        requests: Optional[Dict[str, IqoalaRequest]] = None,
    ) -> None:
        self._instructions: List[ClassicalIqoalaOp] = instructions
        self._local_routines: Dict[str, IqoalaSubroutine] = local_routines
        self._meta: ProgramMeta = meta

        if requests is None:
            self._requests: Dict[str, IqoalaRequest] = {}
        else:
            self._requests: Dict[str, IqoalaRequest] = requests

    @property
    def meta(self) -> ProgramMeta:
        return self._meta

    @property
    def instructions(self) -> List[ClassicalIqoalaOp]:
        return self._instructions

    @instructions.setter
    def instructions(self, new_instrs) -> None:
        self._instructions = new_instrs

    def get_instr_signatures(self) -> List[IqoalaInstructionSignature]:
        sigs: List[IqoalaInstructionSignature] = []
        for instr in self.instructions:
            if isinstance(instr, RunSubroutineOp):
                subrt = instr.subroutine
                for nq_instr in subrt.subroutine.instructions:
                    typ = netqasm_instr_to_type(nq_instr)
                    # TODO: add duration
                    sigs.append(IqoalaInstructionSignature(typ))
            else:
                sigs.append(IqoalaInstructionSignature(instr.TYP))
        return sigs

    @property
    def local_routines(self) -> Dict[str, IqoalaSubroutine]:
        return self._local_routines

    @local_routines.setter
    def local_routines(self, new_local_routines: Dict[str, IqoalaSubroutine]) -> None:
        self._local_routines = new_local_routines

    @property
    def requests(self) -> Dict[str, IqoalaRequest]:
        return self._requests

    @requests.setter
    def requests(self, new_requests: Dict[str, IqoalaRequest]) -> None:
        self._requests = new_requests

    def __str__(self) -> str:
        # self.me
        # instrs = [
        #     f"{str(i)}\n{self.subroutines[i.arguments[0]]}"  # inline subroutine contents
        #     if isinstance(i, RunSubroutineOp)
        #     else str(i)
        #     for i in self.instructions
        # ]

        # return "\n".join("  " + i for i in instrs)
        return "\n".join("  " + str(i) for i in self.instructions)

    def serialize_meta(self) -> str:
        return self.meta.serialize()

    def serialize_instructions(self) -> str:
        return "\n".join("  " + str(i) for i in self.instructions)

    def serialize_subroutines(self) -> str:
        return "\n".join(s.serialize() for s in self.local_routines.values())

    def serialize(self) -> str:
        return (
            self.meta.serialize()
            + "\n"
            + self.serialize_instructions()
            + "\n"
            + self.serialize_subroutines()
        )
