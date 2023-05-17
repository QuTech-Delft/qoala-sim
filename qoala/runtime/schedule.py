from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass
class NetworkSchedule:
    bin_length: int
    first_bin: int
    bin_period: int

    def next_bin(self, time: int) -> int:
        offset = time - self.first_bin
        next_bin_index = ceil(offset / self.bin_period)
        next_bin_start = next_bin_index * self.bin_period
        return next_bin_start + self.first_bin
