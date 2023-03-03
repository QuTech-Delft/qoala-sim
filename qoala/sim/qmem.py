from __future__ import annotations

from qoala.lang.ehi import UnitModule


class QuantumMemory:
    """Quantum memory only available to Qnos. Represented as unit modules."""

    # Only describes the virtual memory space (i.e. unit module).
    # Does not contain 'values' (quantum states) of the virtual memory locations.
    # Does not contain the mapping from virtual to physical space. (Managed by memmgr)

    def __init__(self, pid: int, unit_module: UnitModule) -> None:
        self._pid = pid
        self._unit_module = unit_module

    @property
    def unit_module(self) -> UnitModule:
        return self._unit_module
