from __future__ import annotations

from typing import Dict, List


class StaticNetworkInfo:
    """Static network info: node IDs. EPR links are managed by EhiNetworkInfo."""

    def __init__(self, nodes: Dict[int, str]) -> None:
        # node ID -> node name
        self._nodes = nodes

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
