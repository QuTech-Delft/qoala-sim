from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qoala.runtime.config import LinkConfig
from qoala.runtime.program import ProgramInstance


@dataclass
class GlobalNodeInfo:
    """Node information available at runtime."""

    name: str
    id: int


@dataclass
class GlobalLinkInfo:
    node_name1: str
    node_name2: str

    fidelity: float

    @classmethod
    def from_config(
        cls, node_name1: str, node_name2: str, config: LinkConfig
    ) -> GlobalLinkInfo:
        if config.typ == "perfect":
            return GlobalLinkInfo(
                node_name1=node_name1, node_name2=node_name2, fidelity=1.0
            )
        elif config.typ == "depolarise":
            return GlobalLinkInfo(
                node_name1=node_name1,
                node_name2=node_name2,
                fidelity=config.cfg.fidelity,  # type: ignore
            )
        else:
            raise NotImplementedError


class GlobalEnvironment:
    def __init__(self) -> None:
        # node ID -> node info
        self._nodes: Dict[int, GlobalNodeInfo] = {}

        # (node A ID, node B ID) -> link info
        # for a pair (a, b) there exists no separate (b, a) info (it is the same)
        self._links: Dict[Tuple[int, int], GlobalLinkInfo] = {}

        self._global_schedule: Optional[List[int]] = None
        self._timeslot_len: Optional[int] = None

    def get_nodes(self) -> Dict[int, GlobalNodeInfo]:
        return self._nodes

    def get_node_id(self, name: str) -> int:
        for id, node in self._nodes.items():
            if node.name == name:
                return id
        raise ValueError

    def set_nodes(self, nodes: Dict[int, GlobalNodeInfo]) -> None:
        self._nodes = nodes

    def add_node(self, id: int, node: GlobalNodeInfo) -> None:
        self._nodes[id] = node

    def get_links(self) -> Dict[Tuple[int, int], GlobalLinkInfo]:
        return self._links

    def set_links(self, links: Dict[Tuple[int, int], GlobalLinkInfo]) -> None:
        self._links = links

    def add_link(self, id1: int, id2: int, link: GlobalLinkInfo) -> None:
        self._links[(id1, id2)] = link

    def set_global_schedule(self, schedule: List[int]) -> None:
        self._global_schedule = schedule

    def get_global_schedule(self) -> List[int]:
        assert self._global_schedule is not None
        return self._global_schedule

    def set_timeslot_len(self, len: int) -> None:
        self._timeslot_len = len

    def get_timeslot_len(self) -> int:
        assert self._timeslot_len is not None
        return self._timeslot_len


class LocalEnvironment:
    def __init__(
        self,
        global_env: GlobalEnvironment,
        node_id: int,
    ) -> None:
        self._global_env: GlobalEnvironment = global_env

        # node ID of self
        self._node_id: int = node_id

        self._programs: List[ProgramInstance] = []
        self._csockets: List[str] = []
        self._epr_sockets: List[str] = []

    def get_global_env(self) -> GlobalEnvironment:
        return self._global_env

    def get_node_id(self) -> int:
        return self._node_id

    def register_program(self, program: ProgramInstance) -> None:
        self._programs.append(program)

    def open_epr_socket(self) -> None:
        pass

    def get_all_node_names(self) -> List[str]:
        return [info.name for info in self.get_global_env().get_nodes().values()]

    def get_all_other_node_names(self) -> List[str]:
        return [
            info.name
            for info in self.get_global_env().get_nodes().values()
            if info.id != self._node_id
        ]


class ProgramEnvironment:
    """Environment interface given to a program"""

    pass
