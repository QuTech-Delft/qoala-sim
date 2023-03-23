from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

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
from qoala.runtime.environment import GlobalEnvironment
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.events import EPR_DELIVERY
from qoala.sim.signals import MSG_REQUEST_DELIVERED
from qoala.util.logging import LogManager


@dataclass(eq=True, frozen=True)
class EntDistRequest:
    local_node_id: int
    remote_node_id: int
    local_qubit_id: int


@dataclass(eq=True, frozen=True)
class JointRequest:
    node1_id: int
    node2_id: int
    node1_qubit_id: int
    node2_qubit_id: int


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


class EntDist(Protocol):
    def __init__(
        self, nodes: List[Node], global_env: GlobalEnvironment, comp: EntDistComponent
    ) -> None:
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._global_env = global_env
        self._comp = comp

        # Owned objects.
        self._interface = EntDistInterface(comp, global_env)

        # Node ID -> Node
        self._nodes: Dict[int, Node] = {node.ID: node for node in nodes}

        # (Node ID 1, Node ID 2) -> Sampler
        self._samplers: Dict[Tuple[int, int], StateDeliverySampler] = {}

        # Node ID -> list of requests
        self._requests: Dict[int, List[EntDistRequest]] = {
            node.ID: [] for node in nodes
        }

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}(EntDist)"
        )

    @property
    def comp(self) -> EntDistComponent:
        return self._comp

    def add_sampler(
        self,
        node1_id: int,
        node2_id: int,
        factory: IStateDeliverySamplerFactory,
        kwargs: Dict[str, Any],
    ) -> None:
        if (node1_id, node2_id) in self._samplers:
            raise ValueError(f"Sampler for ({node1_id}, {node2_id}) already registered")
        if (node2_id, node1_id) in self._samplers:
            raise ValueError(
                f"Sampler for ({node1_id}, {node2_id}) already registered. "
                "NOTE: only one sampler per node pair is allowed; order does not matter."
            )
        sampler = factory.create_state_delivery_sampler(**kwargs)
        self._samplers[(node1_id, node2_id)] = sampler

    def get_sampler(self, node1_id: int, node2_id: int) -> StateDeliverySampler:
        try:
            return self._samplers[(node1_id, node2_id)]
        except KeyError:
            pass
        try:
            return self._samplers[(node2_id, node1_id)]
        except KeyError:
            raise ValueError(f"No sampler registered for pair ({node1_id}, {node2_id})")

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
        sampler = self.get_sampler(node1_id, node2_id)
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

        node1_mem.put(qubits=epr[0], positions=node1_phys_id)
        node2_mem.put(qubits=epr[1], positions=node2_phys_id)

        # Send messages to the nodes indictating a request has been delivered.
        node1 = self._interface.remote_id_to_peer_name(node1_id)
        node2 = self._interface.remote_id_to_peer_name(node2_id)
        self._interface.send_node_msg(node1, Message(MSG_REQUEST_DELIVERED))
        self._interface.send_node_msg(node2, Message(MSG_REQUEST_DELIVERED))

    def put_request(self, request: EntDistRequest) -> None:
        if request.local_node_id not in self._nodes:
            raise ValueError(
                f"Invalid request: node ID {request.local_node_id} not registered in EntDist."
            )
        if request.remote_node_id not in self._nodes:
            raise ValueError(
                f"Invalid request: node ID {request.remote_node_id} not registered in EntDist."
            )
        self._requests[request.local_node_id].append(request)

    def get_requests(self, node_id: int) -> List[EntDistRequest]:
        return self._requests[node_id]

    def pop_request(self, node_id: int, index: int) -> EntDistRequest:
        return self._requests[node_id].pop(index)

    def get_remote_request_for(self, local_request: EntDistRequest) -> Optional[int]:
        """Return index in request list of remote node."""
        remote_request_index = None

        try:
            remote_requests = self._requests[local_request.remote_node_id]
        except KeyError:
            # Invalid remote node ID.
            raise ValueError(
                f"Request {local_request} refers to remote node "
                f"{local_request.remote_node_id} "
                "but this node is not registed in the EntDist."
            )

        for i, req in enumerate(remote_requests):
            # Find the remote request that corresponds to the local request.
            if req.remote_node_id == local_request.local_node_id:
                remote_request_index = i
                break  # break out of "for i, remote_request" loop

        return remote_request_index

    def get_next_joint_request(self) -> Optional[JointRequest]:
        for _, local_requests in self._requests.items():
            if len(local_requests) == 0:
                continue
            local_request = local_requests.pop(0)
            remote_request_id = self.get_remote_request_for(local_request)
            if remote_request_id is not None:
                remote_id = local_request.remote_node_id
                remote_request = self._requests[remote_id].pop(remote_request_id)
                return JointRequest(
                    local_request.local_node_id,
                    remote_request.local_node_id,
                    local_request.local_qubit_id,
                    remote_request.local_qubit_id,
                )
            else:
                # Put local request back
                local_requests.insert(0, local_request)

        # No joint requests found
        return None

    def serve_request(
        self, request: JointRequest
    ) -> Generator[EventExpression, None, None]:
        yield from self.deliver(
            node1_id=request.node1_id,
            node1_phys_id=request.node1_qubit_id,
            node2_id=request.node2_id,
            node2_phys_id=request.node2_qubit_id,
            state_delay=1000,
        )

    def serve_all_requests(self) -> Generator[EventExpression, None, None]:
        while request := self.get_next_joint_request():
            yield from self.deliver(
                node1_id=request.node1_id,
                node1_phys_id=request.node1_qubit_id,
                node2_id=request.node2_id,
                node2_phys_id=request.node2_qubit_id,
                state_delay=1000,
            )

    def start(self) -> None:
        assert self._interface is not None
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()

    def run(self) -> Generator[EventExpression, None, None]:
        # Loop forever acting on messages from the nodes.
        while True:
            # Wait for a new message.
            msg = yield from self._interface.receive_msg()
            self._logger.debug(f"received new msg from node: {msg}")
            request: EntDistRequest = msg.content
            self.put_request(request)
            yield from self.serve_all_requests()
