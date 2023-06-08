import abc
from typing import Dict, List, Type

from netqasm.lang.instr import core, nv, vanilla
from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour, NVFlavour, VanillaFlavour
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_CZ,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.instructions import Instruction as NetSquidInstruction

from qoala.runtime.lhi import INSTR_MEASURE_INSTANT


class NtfInterface(abc.ABC):
    @abc.abstractmethod
    def flavour(self) -> Type[Flavour]:
        raise NotImplementedError

    @abc.abstractmethod
    def native_to_netqasm(
        self, ns_instr: Type[NetSquidInstruction]
    ) -> List[Type[NetQASMInstruction]]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        raise NotImplementedError

    @abc.abstractmethod
    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
        raise NotImplementedError


class DefaultGenericNtf(NtfInterface):
    _NS_NQ_MAP: Dict[Type[NetSquidInstruction], List[Type[NetQASMInstruction]]] = {
        INSTR_INIT: [core.InitInstruction],
        INSTR_X: [vanilla.GateXInstruction],
        INSTR_Y: [vanilla.GateYInstruction],
        INSTR_Z: [vanilla.GateZInstruction],
        INSTR_H: [vanilla.GateHInstruction],
        INSTR_ROT_X: [vanilla.RotXInstruction],
        INSTR_ROT_Y: [vanilla.RotYInstruction],
        INSTR_ROT_Z: [vanilla.RotZInstruction],
        INSTR_CNOT: [vanilla.CnotInstruction],
        INSTR_CZ: [vanilla.CphaseInstruction],
        INSTR_MEASURE: [core.MeasInstruction],
        INSTR_MEASURE_INSTANT: [core.MeasInstruction],
    }

    _NQ_NS_MAP: Dict[Type[NetSquidInstruction], List[Type[NetQASMInstruction]]] = {
        core.InitInstruction: [INSTR_INIT],
        vanilla.GateXInstruction: [INSTR_X],
        vanilla.GateYInstruction: [INSTR_Y],
        vanilla.GateZInstruction: [INSTR_Z],
        vanilla.GateHInstruction: [INSTR_H],
        vanilla.RotXInstruction: [INSTR_ROT_X],
        vanilla.RotYInstruction: [INSTR_ROT_Y],
        vanilla.RotZInstruction: [INSTR_ROT_Z],
        vanilla.CnotInstruction: [INSTR_CNOT],
        vanilla.CphaseInstruction: [INSTR_CZ],
        core.MeasInstruction: [INSTR_MEASURE],
        core.MeasInstruction: [INSTR_MEASURE_INSTANT],
    }

    def flavour(self) -> Type[Flavour]:
        return VanillaFlavour  # type: ignore

    def native_to_netqasm(
        self, ns_instr: Type[NetSquidInstruction]
    ) -> List[Type[NetQASMInstruction]]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        return self._NS_NQ_MAP[ns_instr]

    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
    # return se


class NvToNvInterface(NtfInterface):
    _MAP: Dict[Type[NetSquidInstruction], Type[NetQASMInstruction]] = {
        INSTR_INIT: core.InitInstruction,
        INSTR_ROT_X: nv.RotXInstruction,
        INSTR_ROT_Y: nv.RotYInstruction,
        INSTR_ROT_Z: nv.RotZInstruction,
        INSTR_CXDIR: nv.ControlledRotXInstruction,
        INSTR_CYDIR: nv.ControlledRotYInstruction,
        INSTR_MEASURE: core.MeasInstruction,
    }

    def flavour(self) -> Type[Flavour]:
        return NVFlavour  # type: ignore

    def map(self, ns_instr: Type[NetSquidInstruction]) -> Type[NetQASMInstruction]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        return self._MAP[ns_instr]
