from dataclasses import dataclass
from typing import Dict, List

from netqasm.lang.instr.base import NetQASMInstruction
from netqasm.lang.instr.flavour import Flavour


@dataclass(eq=True, frozen=True)
class ExposedQubitInfo:
    is_communication: bool
    decoherence_rate: float  # rate per second


@dataclass(eq=True, frozen=True)
class ExposedGateInfo:
    ids: List[int]  # ordered list of qubit IDs
    duration: int  # ns
    decoherence: List[int]  # rate per second, per qubit ID (same order as `ids`)


@dataclass(eq=True, frozen=True)
class ExposedHardwareInfo:
    """Hardware made available to offline compiler."""

    qubit_infos: Dict[int, ExposedQubitInfo]  # qubit ID -> info

    flavour: Flavour  # set of NetQASM instrs, no info about which qubits can do what instr
    gate_infos: Dict[NetQASMInstruction, ExposedGateInfo]  # instr -> info
