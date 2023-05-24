from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qoala.lang.ehi import EhiLinkInfo


class StaticNetworkInfo:
    """Static network info: node IDs. EPR links are managed by EhiNetworkInfo."""

    def __init__(self, nodes: Dict[int, str]) -> None:
        # node ID -> node name
        self._nodes = nodes

        self._global_schedule: Optional[List[int]] = None
        self._timeslot_len: Optional[int] = None

    @classmethod
    def with_nodes(cls, nodes: Dict[int, str]) -> StaticNetworkInfo:
        return StaticNetworkInfo(nodes)

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

    def get_links(self) -> Dict[Tuple[int, int], EhiLinkInfo]:
        return self._links

    def set_links(self, links: Dict[Tuple[int, int], EhiLinkInfo]) -> None:
        self._links = links

    def add_link(self, id1: int, id2: int, link: EhiLinkInfo) -> None:
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
