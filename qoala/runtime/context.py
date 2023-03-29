from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qoala.runtime.environment import NetworkEhi
from qoala.sim.globals import GlobalSimData


@dataclass
class SimulationContext:
    network_ehi: Optional[NetworkEhi] = None
    global_sim_data: Optional[GlobalSimData] = None
