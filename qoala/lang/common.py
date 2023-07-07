from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MultiQubit:
    qubit_ids: List[int]

    def __hash__(self) -> int:
        return hash(tuple(self.qubit_ids))
