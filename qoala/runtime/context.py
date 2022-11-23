from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qoala.runtime.environment import GlobalEnvironment
from qoala.sim.globals import GlobalSimData


@dataclass
class SimulationContext:
    global_env: Optional[GlobalEnvironment] = None
    global_sim_data: Optional[GlobalSimData] = None
