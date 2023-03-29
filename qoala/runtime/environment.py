from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qoala.lang.ehi import ExposedLinkInfo
from qoala.runtime.program import ProgramInstance


class NetworkEhi:
    def __init__(
        self, nodes: Dict[int, str], links: Dict[Tuple[int, int], ExposedLinkInfo]
    ) -> None:
        # node ID -> node name
        self._nodes = nodes

        # (node A ID, node B ID) -> link info
        # for a pair (a, b) there exists no separate (b, a) info (it is the same)
        self._links = links

        self._global_schedule: Optional[List[int]] = None
        self._timeslot_len: Optional[int] = None

    @classmethod
    def with_nodes_no_links(cls, nodes: Dict[int, str]) -> NetworkEhi:
        return NetworkEhi(nodes, links={})

    def get_nodes(self) -> Dict[int, str]:
        return self._nodes

    def get_node_id(self, name: str) -> int:
        for id, node_name in self._nodes.items():
            if node_name == name:
                return id
        raise ValueError

    def get_all_node_names(self) -> List[str]:
        return list(self._nodes.values())

    def set_nodes(self, nodes: Dict[int, str]) -> None:
        self._nodes = nodes

    def add_node(self, id: int, name: str) -> None:
        self._nodes[id] = name

    def get_links(self) -> Dict[Tuple[int, int], ExposedLinkInfo]:
        return self._links

    def set_links(self, links: Dict[Tuple[int, int], ExposedLinkInfo]) -> None:
        self._links = links

    def add_link(self, id1: int, id2: int, link: ExposedLinkInfo) -> None:
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
        network_ehi: NetworkEhi,
        node_id: int,
    ) -> None:
        self._network_ehi: NetworkEhi = network_ehi

        # node ID of self
        self._node_id: int = node_id

        self._programs: List[ProgramInstance] = []
        self._csockets: List[str] = []
        self._epr_sockets: List[str] = []

    def get_network_ehi(self) -> NetworkEhi:
        return self._network_ehi

    def get_node_id(self) -> int:
        return self._node_id

    def register_program(self, program: ProgramInstance) -> None:
        self._programs.append(program)

    def open_epr_socket(self) -> None:
        pass

    def get_all_node_names(self) -> List[str]:
        return self.get_network_ehi().get_all_node_names()

    def get_all_other_node_names(self) -> List[str]:
        return [
            name
            for id, name in self.get_network_ehi().get_nodes().items()
            if id != self._node_id
        ]


class ProgramEnvironment:
    """Environment interface given to a program"""

    pass
