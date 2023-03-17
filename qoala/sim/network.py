from __future__ import annotations

from typing import Dict

from netsquid.components import QuantumProcessor
from netsquid.nodes.network import Network

from qoala.sim.entdist.entdist import EntDist
from qoala.sim.procnode import ProcNode


class ProcNodeNetwork(Network):
    """A network of `ProcNode`s connected by links, which are
    `MagicLinkLayerProtocol`s."""

    def __init__(self, nodes: Dict[str, ProcNode], entdist: EntDist) -> None:
        """ProcNodeNetwork constructor.

        :param nodes: dictionary of node name to `ProcNode` object representing
        that node
        :param links: list of link layer protocol objects. Each object internally
        contains the IDs of the two nodes that this link connects
        """
        self._nodes = nodes
        self._entdist = entdist

    @property
    def nodes(self) -> Dict[str, ProcNode]:
        return self._nodes

    @property
    def entdist(self) -> EntDist:
        return self._entdist

    @property
    def qdevices(self) -> Dict[str, QuantumProcessor]:
        return {name: node.qdevice for name, node in self._nodes.items()}

    def start_all_nodes(self) -> None:
        for node in self.nodes.values():
            node.start()

    def start_entdist(self) -> None:
        self.entdist.start()

    def start(self) -> None:
        self.start_entdist()
        self.start_all_nodes()
