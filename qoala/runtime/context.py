from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qoala.runtime.environment import NetworkInfo
from qoala.sim.globals import GlobalSimData


@dataclass
class SimulationContext:
    network_info: Optional[NetworkInfo] = None
    global_sim_data: Optional[GlobalSimData] = None
