from abc import ABC, abstractmethod
from typing import Dict, List, Type

from netqasm.lang.instr import core, nv, trapped_ion, vanilla
from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import (
    Flavour,
    NVFlavour,
    TrappedIonFlavour,
    VanillaFlavour,
)
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

from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)
from qoala.runtime.lhi import INSTR_MEASURE_INSTANT


class NtfInterface(ABC):
    @abstractmethod
    def flavour(self) -> Type[Flavour]:
        raise NotImplementedError

    @abstractmethod
    def native_to_netqasm(
        self, ns_instr: Type[NetSquidInstruction]
    ) -> List[Type[NetQASMInstruction]]:
        """Responsiblity of implementor that return instructions are of the
        flavour returned by flavour()."""
        raise NotImplementedError

    @abstractmethod
    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
        raise NotImplementedError


class GenericNtf(NtfInterface):
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
        return self._NS_NQ_MAP[ns_instr]

    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
        return self._NQ_NS_MAP[nq_instr]


class NvNtf(NtfInterface):
    _NS_NQ_MAP: Dict[Type[NetSquidInstruction], List[Type[NetQASMInstruction]]] = {
        INSTR_INIT: [core.InitInstruction],
        INSTR_ROT_X: [nv.RotXInstruction],
        INSTR_ROT_Y: [nv.RotYInstruction],
        INSTR_ROT_Z: [nv.RotZInstruction],
        INSTR_CXDIR: [nv.ControlledRotXInstruction],
        INSTR_CYDIR: [nv.ControlledRotYInstruction],
        INSTR_MEASURE: [core.MeasInstruction],
        INSTR_MEASURE_INSTANT: [core.MeasInstruction],
    }

    _NQ_NS_MAP: Dict[Type[NetQASMInstruction], List[Type[NetSquidInstruction]]] = {
        core.InitInstruction: [INSTR_INIT],
        nv.RotXInstruction: [INSTR_ROT_X],
        nv.RotYInstruction: [INSTR_ROT_Y],
        nv.RotZInstruction: [INSTR_ROT_Z],
        nv.ControlledRotXInstruction: [INSTR_CXDIR],
        nv.ControlledRotYInstruction: [INSTR_CYDIR],
        core.MeasInstruction: [INSTR_MEASURE],
        core.MeasInstruction: [INSTR_MEASURE_INSTANT],
    }

    def flavour(self) -> Type[Flavour]:
        return NVFlavour  # type: ignore

    def native_to_netqasm(
        self, ns_instr: Type[NetSquidInstruction]
    ) -> List[Type[NetQASMInstruction]]:
        return self._NS_NQ_MAP[ns_instr]

    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
        return self._NQ_NS_MAP[nq_instr]


class TrappedIonNtf(NtfInterface):
    #  TODO Check if this is correct
    _NS_NQ_MAP: Dict[Type[NetSquidInstruction], List[Type[NetQASMInstruction]]] = {
        INSTR_INIT: [core.InitInstruction, trapped_ion.AllQubitsInitInstruction],
        INSTR_ROT_Z: [trapped_ion.RotZInstruction],
        INSTR_BICHROMATIC: [trapped_ion.BichromaticInstruction],
        INSTR_MEASURE_ALL: [trapped_ion.AllQubitsMeasInstruction],
        INSTR_ROT_X_ALL: [trapped_ion.AllQubitsRotXInstruction],
        INSTR_ROT_Y_ALL: [trapped_ion.AllQubitsRotYInstruction],
        INSTR_ROT_Z_ALL: [trapped_ion.AllQubitsRotZInstruction],
        INSTR_MEASURE: [core.MeasInstruction],
        INSTR_MEASURE_INSTANT: [core.MeasInstruction],
    }

    _NQ_NS_MAP: Dict[Type[NetSquidInstruction], List[Type[NetQASMInstruction]]] = {
        core.InitInstruction: [INSTR_INIT],
        trapped_ion.AllQubitsInitInstruction: [INSTR_INIT],
        trapped_ion.RotZInstruction: [INSTR_ROT_Z],
        core.MeasInstruction: [INSTR_MEASURE],
        trapped_ion.AllQubitsMeasInstruction: [INSTR_MEASURE_ALL],
        trapped_ion.AllQubitsRotXInstruction: [INSTR_ROT_X_ALL],
        trapped_ion.AllQubitsRotYInstruction: [INSTR_ROT_Y_ALL],
        trapped_ion.AllQubitsRotZInstruction: [INSTR_ROT_Z_ALL],
        trapped_ion.BichromaticInstruction: [INSTR_BICHROMATIC],
    }

    def flavour(self) -> Type[Flavour]:
        return TrappedIonFlavour  # type: ignore

    def native_to_netqasm(
        self, ns_instr: Type[NetSquidInstruction]
    ) -> List[Type[NetQASMInstruction]]:
        return self._NS_NQ_MAP[ns_instr]

    def netqasm_to_native(
        self, nq_instr: Type[NetQASMInstruction]
    ) -> List[Type[NetSquidInstruction]]:
        return self._NQ_NS_MAP[nq_instr]


class NtfInterfaceConfigInterface(ABC):
    @abstractmethod
    def to_ntf_interface(self) -> Type[NtfInterface]:
        raise NotImplementedError
