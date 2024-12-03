from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
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
from qoala.sim.events import BIN_END, EPR_DELIVERY
from qoala.util.logging import LogManager


class EntDistEventType(Enum):
    MSG_ARRIVED = auto()
    EPR_DELIVERY = auto()
    BIN_END = auto()


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


@dataclass
class EprDeliveryEvent:
    request: JointRequest
    qubits: Tuple[Qubit, Qubit]
    abs_time: float


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

        # Requests from individual nodes that may get handled in the current time bin.
        # Node ID -> list of requests.
        self._requests: Dict[int, List[EntDistRequest]] = {
            node.ID: [] for node in nodes
        }

        # Successful deliveries of EPR pairs that have been scheduled but not actually
        # realized yet. Sorted by increasing delivery time.
        self._deliveries: List[EprDeliveryEvent] = []

        # Joint requests that will fail at the end of the current time bin.
        self._failed_requests: List[JointRequest] = []

        # Whether an event has been scheduled that will be triggered at the end of the
        # current time bin.
        self._bin_end_event_scheduled: bool = False

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}(EntDist)"
        )

    @property
    def comp(self) -> EntDistComponent:
        return self._comp

    def clear_requests(self) -> None:
        self._requests = {id: [] for id in self._nodes.keys()}

    def node_name_for(self, id: int) -> str:
        """Convenience method for converting a node ID to node name."""
        return self._interface.remote_id_to_peer_name(id)

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

    def schedule_deliveries(self, requests: List[JointRequest]) -> None:
        now = ns.sim_time()
        assert self._netschedule is not None
        curr_bin = self._netschedule.current_bin(now)
        assert curr_bin is not None

        for req in requests:
            # Set quantum memory used for EPR pairs to "in use".
            node1_mem = self._nodes[req.node1_id].qmemory
            node2_mem = self._nodes[req.node2_id].qmemory
            node1_mem.mem_positions[req.node1_qubit_id].in_use = True
            node2_mem.mem_positions[req.node2_qubit_id].in_use = True

            # Sample the EPR generation, resulting in a (noisy) EPR pair and a duration.
            # It is not yet put into actual memory.
            epr, duration = self._sample_epr_gen(req)

            if now + duration < curr_bin.end:
                # EPR generation succeeded before end of time bin.
                self._deliveries.append(EprDeliveryEvent(req, epr, now + duration))
            else:
                # EPR generation did not succeed before end of time bin.
                self._failed_requests.append(req)
                self._logger.info(f"failed request: {req} (duration {duration})")

        # Sort deliveries by increasing finish time.
        self._deliveries.sort(key=lambda d: d.abs_time)

        # Schedule an event for each successful delivery.
        # Failed requests are handled at the end of the time bin.
        for delivery in self._deliveries:
            self._logger.info(f"scheduling delivery {delivery}")
            self._schedule_at(delivery.abs_time, EPR_DELIVERY)

        # If there is at least one failed request, there should be an event at the end
        # of the current time bin, simulating that EPR creation for these requests
        # stops at that time.
        if len(self._failed_requests) > 0:
            if not self._bin_end_event_scheduled:
                self._schedule_next_bin_end_event()

    def _sample_epr_gen(
        self, request: JointRequest
    ) -> Tuple[Tuple[Qubit, Qubit], float]:
        timed_sampler = self.get_sampler(request.node1_id, request.node2_id)
        sample = self.sample_state(timed_sampler.sampler)
        epr = self.create_epr_pair_with_state(sample.state)

        self._logger.info(f"sample duration: {sample.duration}")
        self._logger.info(f"total duration: {timed_sampler.delay}")
        total_delay = sample.duration + timed_sampler.delay

        return (epr, total_delay)

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

    def get_all_joint_requests(
        self, pop_node_requests: bool = True
    ) -> List[JointRequest]:
        # pop_node_requests=False can be used to peek without changing state
        # However it may reorder the requests internally (but I don't think this matters)

        joint_requests = []

        nextJointRequest = self.get_next_joint_request()
        while nextJointRequest is not None:
            self._logger.debug(
                f"Handling Joint Request for Nodes: ({nextJointRequest.node1_id}, {nextJointRequest.node2_id})"
            )
            joint_requests.append(nextJointRequest)
            nextJointRequest = self.get_next_joint_request()

        # Need to reconstruct the requests
        if not pop_node_requests:
            for joint_request in joint_requests:
                node1_id = joint_request.node1_id
                node2_id = joint_request.node2_id
                node1_qubit_id = joint_request.node1_qubit_id
                node2_qubit_id = joint_request.node2_qubit_id
                node1_pid = joint_request.node1_pid
                node2_pid = joint_request.node2_pid
                self.put_request(
                    EntDistRequest(
                        node1_id, node2_id, node1_qubit_id, node1_pid, node2_pid
                    )
                )
                self.put_request(
                    EntDistRequest(
                        node2_id, node1_id, node2_qubit_id, node2_pid, node1_pid
                    )
                )

        return joint_requests

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

    def _handle_messages(self) -> None:
        now = ns.sim_time()
        assert self._netschedule is not None
        curr_bin = self._netschedule.current_bin(now)

        messages = self._interface.pop_all_messages()

        # Gather requests from the messages and add them to the request list
        # (self._requests). Only keep those that are allowed to be served in the
        # current time bin. For the others, immediately send back a failure response.
        for msg in messages:
            self._logger.info(f"received new msg from node: {msg}")
            request: EntDistRequest = msg.content

            self._logger.info(f"netschedule: {self._netschedule}")
            self._logger.info(f"current bin: {curr_bin}")
            node = self._interface.remote_id_to_peer_name(request.local_node_id)
            if curr_bin and request.matches_timebin(curr_bin.bin):
                if request in self._requests:
                    # This exact request was already received earlier and is still pending.
                    self._logger.info(f"not handling msg {msg} (already pending)")
                    self._interface.send_node_msg(node, Message(-1, -1, None))
                else:
                    self.put_request(request)
            else:
                self._logger.info(f"not handling msg {msg} (wrong timebin)")
                self._interface.send_node_msg(node, Message(-1, -1, None))

        # Find pairs of requests which match.
        joint_requests = self.get_all_joint_requests()

        # Schedule EPR delivery events for them.
        self.schedule_deliveries(joint_requests)

        # Schedule an event for the end of the time bin (if not already scheduled),
        # such that non-fulfilled node requests can be removed at that time.
        if not self._bin_end_event_scheduled:
            self._schedule_next_bin_end_event()

    def _handle_delivery(self) -> None:
        now = ns.sim_time()

        # Pop the first scheduled delivery.
        # Since self._deliveries is sorted by time, the one we just popped (call it D)
        # had the earliest scheduled completion, and hence the delivery event that just
        # happened and triggered this method must match D.
        delivery = self._deliveries.pop(0)
        assert delivery.abs_time == now

        node1 = self.node_name_for(delivery.request.node1_id)
        node2 = self.node_name_for(delivery.request.node2_id)
        node1_mem = self._nodes[delivery.request.node1_id].qmemory
        node2_mem = self._nodes[delivery.request.node2_id].qmemory

        # EPR creation happened. Put the qubits into actual memory.
        q0, q1 = delivery.qubits
        node1_mem.put(qubits=q0, positions=delivery.request.node1_qubit_id)
        node2_mem.put(qubits=q1, positions=delivery.request.node2_qubit_id)

        # Send messages to the nodes indicating a request has been delivered.
        node1_pid, node2_pid = delivery.request.node1_pid, delivery.request.node2_pid
        self._interface.send_node_msg(node1, Message(-1, -1, node1_pid))
        self._interface.send_node_msg(node2, Message(-1, -1, node2_pid))

    def _handle_bin_end(self) -> None:
        # Reset flag.
        self._bin_end_event_scheduled = False

        # Release memory for failed requests and send back a failure message.
        for freq in self._failed_requests:
            node1 = self.node_name_for(freq.node1_id)
            node2 = self.node_name_for(freq.node2_id)
            node1_mem = self._nodes[freq.node1_id].qmemory
            node2_mem = self._nodes[freq.node2_id].qmemory
            node1_mem.mem_positions[freq.node1_qubit_id].in_use = False
            node2_mem.mem_positions[freq.node2_qubit_id].in_use = False

            # TODO determine content of failure message
            self._interface.send_node_msg(node1, Message(-1, -1, None))
            self._interface.send_node_msg(node2, Message(-1, -1, None))

        # Send back a failure message for each individual request (request that didn't
        # have a matching request from the other node during this time bin).
        for req_list in self._requests.values():
            for req in req_list:
                node = self.node_name_for(req.local_node_id)
                self._interface.send_node_msg(node, Message(-1, -1, None))

        # Clear all requests.
        self._failed_requests.clear()
        self.clear_requests()

    def _get_event_type(
        self, ev_expr: EventExpression
    ) -> Generator[EventExpression, None, EntDistEventType]:
        # TODO find out if this can be done in a better way.

        # ev_expr = (ev_msg_arrived | ev_epr_delivery) | ev_bin_end
        if len(ev_expr.first_term.triggered_events) > 0:  # type: ignore
            # First term triggered, i.e. (ev_msg_arrived | ev_epr_delivery)
            if len(ev_expr.first_term.first_term.triggered_events) > 0:  # type: ignore
                # First term triggered, i.e. ev_msg_arrived
                # Need to process this event (flushing potential other messages)
                self._logger.debug("message evnet")
                yield from self._interface.handle_msg_evexpr(ev_expr)
                return EntDistEventType.MSG_ARRIVED
            else:
                # Second term triggered, i.e. ev_epr_delivery
                return EntDistEventType.EPR_DELIVERY
        else:
            # Second term triggered, i.e. ev_bin_end
            return EntDistEventType.BIN_END

    def _schedule_next_bin_end_event(self) -> None:
        now = ns.sim_time()
        assert self._netschedule is not None
        curr_bin = self._netschedule.current_bin(now)
        self._logger.debug("scheduling bin end event")
        if curr_bin is None:
            # Currently not inside a bin; get the next one.
            next_bin = self._netschedule.next_bin(now)
            # end - 1 since time bin is *excluding* end time
            self._schedule_at(next_bin.end - 1, BIN_END)
        else:
            self._schedule_at(curr_bin.end - 1, BIN_END)
        self._bin_end_event_scheduled = True

    def _run_with_netschedule(self) -> Generator[EventExpression, None, None]:
        while True:
            # Possible events that should make the EntDist do something:
            # 1. A message arrived.
            ev_msg_arrived = self._interface.get_evexpr_for_any_msg()
            # 2. An EPR pair was created.
            ev_epr_delivery = EventExpression(source=self, event_type=EPR_DELIVERY)
            # 3. A time bin ends.
            ev_bin_end = EventExpression(source=self, event_type=BIN_END)

            # Wait for any of the above events to happen.
            ev_union = (ev_msg_arrived | ev_epr_delivery) | ev_bin_end  # type: ignore
            yield ev_union

            # Check which type of event happened.
            ev_type = yield from self._get_event_type(ev_union)

            if ev_type == EntDistEventType.MSG_ARRIVED:
                self._logger.debug("message arrived")
                self._handle_messages()
            elif ev_type == EntDistEventType.EPR_DELIVERY:
                self._logger.debug("epr delivery")
                self._handle_delivery()
            elif ev_type == EntDistEventType.BIN_END:
                self._logger.debug("bin end")
                self._handle_bin_end()

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
