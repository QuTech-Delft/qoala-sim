from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type


@dataclass(eq=True, frozen=True)
class MultiQubit:
    qubit_ids: List[int]

    def __hash__(self) -> int:
        return hash(tuple(self.qubit_ids))
