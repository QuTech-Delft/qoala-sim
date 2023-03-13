from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple

from netsquid.nodes import Node
from netsquid.protocols import Protocol
from netsquid.qubits import qubitapi
from netsquid.qubits.qrepr import QRepr
from netsquid.qubits.qubit import Qubit
from netsquid_magic.state_delivery_sampler import (
    DeliverySample,
    IStateDeliverySamplerFactory,
    StateDeliverySampler,
)

from pydynaa import EventExpression
from qoala.sim.events import EPR_DELIVERY


@dataclass
class EprDeliverySample:
    state: QRepr
    duration: float

    @classmethod
    def from_ns_magic_delivery_sample(cls, sample: DeliverySample) -> EprDeliverySample:
        assert sample.state.num_qubits == 2
        return EprDeliverySample(
            state=sample.state.dm, duration=sample.delivery_duration
        )


class GlobalEntanglementDistributor(Protocol):
    def __init__(self, nodes: List[Node]) -> None:
        # Node ID -> Node
        self._nodes: Dict[int, Node] = {node.ID: node for node in nodes}

        # (Node ID 1, Node ID 2) -> Sampler
        self._samplers: Dict[Tuple[int, int], StateDeliverySampler] = {}

    def add_sampler(
        self,
        node1_id: int,
        node2_id: int,
        factory: IStateDeliverySamplerFactory,
        kwargs: Dict[str, Any],
    ) -> None:
        sampler = factory.create_state_delivery_sampler(**kwargs)
        self._samplers[(node1_id, node2_id)] = sampler

    def sample_state(self, sampler: StateDeliverySampler) -> EprDeliverySample:
        raw_sample: DeliverySample = sampler.sample()
        return EprDeliverySample.from_ns_magic_delivery_sample(raw_sample)

    def create_epr_pair_with_state(self, state: QRepr) -> Tuple[Qubit, Qubit]:
        q0, q1 = qubitapi.create_qubits(2)
        qubitapi.assign_qstate([q0, q1], state)
        return q0, q1

    def deliver(
        self,
        node1_id: int,
        node1_phys_id: int,
        node2_id: int,
        node2_phys_id: int,
        state_delay: float,
    ) -> Generator[EventExpression, None, None]:
        sampler = self._samplers[(node1_id, node2_id)]
        sample = self.sample_state(sampler)
        epr = self.create_epr_pair_with_state(sample.state)

        total_delay = sample.duration + state_delay

        node1_mem = self._nodes[node1_id].qmemory
        node2_mem = self._nodes[node2_id].qmemory

        node1_mem.mem_positions[node1_phys_id].in_use = True
        node2_mem.mem_positions[node2_phys_id].in_use = True

        self._schedule_after(total_delay, EPR_DELIVERY)
        event_expr = EventExpression(source=self, event_type=EPR_DELIVERY)
        yield event_expr

        node1_mem.put(
            qubits=epr[0], positions=node1_phys_id, replace=True, check_positions=True
        )
        node2_mem.put(
            qubits=epr[1], positions=node2_phys_id, replace=True, check_positions=True
        )
