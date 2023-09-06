from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Generator, List, Optional, Tuple

import netsquid as ns
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
from qoala.lang.ehi import EhiNetworkInfo, EhiNetworkTimebin
from qoala.runtime.lhi import LhiLinkInfo
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.events import EPR_DELIVERY
from qoala.util.logging import LogManager


@dataclass(frozen=True)
class EntDistRequest:
    local_node_id: int
    remote_node_id: int
    local_qubit_id: int
    local_pid: int
    remote_pid: int

    def is_opposite(self, req: EntDistRequest) -> bool:
        return (
            self.local_node_id == req.remote_node_id
            and self.remote_node_id == req.local_node_id
            and self.local_pid == req.remote_pid
            and self.remote_pid == req.local_pid
        )

    def matches_timebin(self, bin: EhiNetworkTimebin) -> bool:
        if frozenset({self.local_node_id, self.remote_node_id}) != bin.nodes:
            return False
        return (
            bin.pids[self.local_node_id] == self.local_pid
            and bin.pids[self.remote_node_id] == self.remote_pid
        )


@dataclass(frozen=True)
class JointRequest:
    node1_id: int
    node2_id: int
    node1_qubit_id: int
    node2_qubit_id: int
    node1_pid: int
    node2_pid: int


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


@dataclass
class DelayedSampler:
    sampler: StateDeliverySampler
    delay: float


class EntDist(Protocol):
    def __init__(
        self,
        nodes: List[Node],
        ehi_network: EhiNetworkInfo,
        comp: EntDistComponent,
    ) -> None:
        super().__init__(name=f"{comp.name}_protocol")

        # References to objects.
        self._ehi_network = ehi_network
        self._comp = comp
        self._netschedule = ehi_network.network_schedule

        # Owned objects.
        self._interface = EntDistInterface(comp, ehi_network)

        # Node ID -> Node
        self._nodes: Dict[int, Node] = {node.ID: node for node in nodes}

        # (Node ID 1, Node ID 2) -> Sampler
        self._samplers: Dict[FrozenSet[int], DelayedSampler] = {}

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

    def clear_requests(self) -> None:
        self._requests = {id: [] for id in self._nodes.keys()}

    def _add_sampler(
        self,
        node1_id: int,
        node2_id: int,
        factory: IStateDeliverySamplerFactory,
        kwargs: Dict[str, Any],
        delay: float,
    ) -> None:
        if node1_id == node2_id:
            raise ValueError("Cannot add sampler for same node.")
        if node1_id not in self._nodes:
            raise ValueError(f"Node {node1_id} not in network.")
        if node2_id not in self._nodes:
            raise ValueError(f"Node {node2_id} not in network.")

        link = frozenset([node1_id, node2_id])
        if link in self._samplers:
            raise ValueError(
                f"Sampler for ({node1_id}, {node2_id}) already registered \
                NOTE: only one sampler per node pair is allowed; order does not matter."
            )
        sampler = factory.create_state_delivery_sampler(**kwargs)
        self._samplers[link] = DelayedSampler(sampler, delay)

    def add_sampler(self, node1_id: int, node2_id: int, info: LhiLinkInfo) -> None:
        self._add_sampler(
            node1_id=node1_id,
            node2_id=node2_id,
            factory=info.sampler_factory(),
            kwargs=info.sampler_kwargs,
            delay=info.state_delay,
        )

    def get_sampler(self, node1_id: int, node2_id: int) -> DelayedSampler:
        link = frozenset([node1_id, node2_id])
        try:
            return self._samplers[link]
        except KeyError:
            raise ValueError(
                f"No sampler registered for pair ({node1_id}, {node2_id}) \
                NOTE: only one sampler per node pair is allowed; order does not matter."
            )

    def sample_state(cls, sampler: StateDeliverySampler) -> EprDeliverySample:
        raw_sample: DeliverySample = sampler.sample()
        return EprDeliverySample.from_ns_magic_delivery_sample(raw_sample)

    def create_epr_pair_with_state(cls, state: QRepr) -> Tuple[Qubit, Qubit]:
        q0, q1 = qubitapi.create_qubits(2)
        qubitapi.assign_qstate([q0, q1], state)
        return q0, q1

    def deliver(
        self,
        node1_id: int,
        node1_phys_id: int,
        node2_id: int,
        node2_phys_id: int,
        node1_pid: int,
        node2_pid: int,
    ) -> Generator[EventExpression, None, None]:
        timed_sampler = self.get_sampler(node1_id, node2_id)
        sample = self.sample_state(timed_sampler.sampler)
        epr = self.create_epr_pair_with_state(sample.state)

        self._logger.info(f"sample duration: {sample.duration}")
        self._logger.info(f"total duration: {timed_sampler.delay}")
        total_delay = sample.duration + timed_sampler.delay

        node1_mem = self._nodes[node1_id].qmemory
        node2_mem = self._nodes[node2_id].qmemory

        if not (0 <= node1_phys_id < node1_mem.num_positions):
            raise ValueError(
                f"qubit location id of {node1_phys_id} is not present in \
                    quantum memory of node ID {node1_id}."
            )
        if not (0 <= node2_phys_id < node2_mem.num_positions):
            raise ValueError(
                f"qubit location id of {node2_phys_id} is not present in \
                    quantum memory of node ID {node2_id}."
            )
        node1_mem.mem_positions[node1_phys_id].in_use = True
        node2_mem.mem_positions[node2_phys_id].in_use = True

        self._schedule_after(total_delay, EPR_DELIVERY)
        event_expr = EventExpression(source=self, event_type=EPR_DELIVERY)
        yield event_expr

        self._logger.info("pair delivered")

        node1_mem.put(qubits=epr[0], positions=node1_phys_id)
        node2_mem.put(qubits=epr[1], positions=node2_phys_id)

        # Send messages to the nodes indictating a request has been delivered.
        node1 = self._interface.remote_id_to_peer_name(node1_id)
        node2 = self._interface.remote_id_to_peer_name(node2_id)
        # TODO: use PIDs??
        self._interface.send_node_msg(node1, Message(-1, -1, node1_pid))
        self._interface.send_node_msg(node2, Message(-1, -1, node2_pid))

    def put_request(self, request: EntDistRequest) -> None:
        if request.local_node_id not in self._nodes:
            raise ValueError(
                f"Invalid request: node ID {request.local_node_id} not registered in EntDist."
            )
        if request.remote_node_id not in self._nodes:
            raise ValueError(
                f"Invalid request: node ID {request.remote_node_id} not registered in EntDist."
            )
        if request.remote_node_id == request.local_node_id:
            raise ValueError(
                f"Invalid request: local node ID {request.local_node_id} and remote node ID "
                "{request.remote_node_id} are the same."
            )

        self._requests[request.local_node_id].append(request)

    def get_requests(self, node_id: int) -> List[EntDistRequest]:
        return self._requests[node_id]

    def pop_request(self, node_id: int, index: int) -> EntDistRequest:
        return self._requests[node_id].pop(index)

    def get_remote_request_for(self, local_request: EntDistRequest) -> Optional[int]:
        """Return index in the request list of the remote node."""

        try:
            remote_requests = self._requests[local_request.remote_node_id]
        except KeyError:
            # Invalid remote node ID.
            raise ValueError(
                f"Request {local_request} refers to remote node \
                {local_request.remote_node_id} \
                but this node is not registed in the EntDist."
            )

        for i, req in enumerate(remote_requests):
            # Find the remote request that corresponds to the local request.
            if local_request.is_opposite(req):
                return i

        return None

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
                    local_request.local_pid,
                    local_request.remote_pid,
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
            node1_pid=request.node1_pid,
            node2_pid=request.node2_pid,
        )

    def serve_all_requests(self) -> Generator[EventExpression, None, None]:
        while (request := self.get_next_joint_request()) is not None:
            yield from self.serve_request(request)

    def start(self) -> None:
        assert self._interface is not None
        super().start()
        self._interface.start()

    def stop(self) -> None:
        self._interface.stop()
        super().stop()

    def _run_with_netschedule(self) -> Generator[EventExpression, None, None]:
        assert self._netschedule is not None

        while True:
            # Wait until a message arrives.
            yield from self._interface.wait_for_any_msg()

            # Wait until the next time bin.
            now = ns.sim_time()
            next_slot_time, next_slot = self._netschedule.next_bin(now)

            if next_slot_time - now > 0:
                yield from self._interface.wait(next_slot_time - now + 1)
            elif next_slot_time - now < 0:
                raise RuntimeError()

            messages = self._interface.pop_all_messages()

            requesting_nodes: List[int] = []
            for msg in messages:
                self._logger.info(f"received new msg from node: {msg}")
                request: EntDistRequest = msg.content
                requesting_nodes.append(request.local_node_id)

                if request.matches_timebin(next_slot):
                    self._logger.warning(f"putting request: {request}")
                    self.put_request(request)
                else:
                    self._logger.warning(f"not handling msg {msg} (wrong timebin)")
            joint_request = self.get_next_joint_request()
            if joint_request is not None:
                self._logger.warning("serving request")
                yield from self.serve_request(joint_request)
                self._logger.warning("served request")
            else:
                for node_id in requesting_nodes:
                    node = self._interface.remote_id_to_peer_name(node_id)
                    self._interface.send_node_msg(node, Message(-1, -1, None))
            self.clear_requests()
            yield from self._interface.wait(1)

    def _run_without_netschedule(self) -> Generator[EventExpression, None, None]:
        while True:
            # Wait for a new message.
            msg = yield from self._interface.receive_msg()
            self._logger.info(f"received new msg from node: {msg}")
            request: EntDistRequest = msg.content
            self.put_request(request)
            yield from self.serve_all_requests()

    def run(self) -> Generator[EventExpression, None, None]:
        if self._netschedule is None:
            yield from self._run_without_netschedule()
        else:
            yield from self._run_with_netschedule()
