from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Generator, List, Optional, Tuple, Union

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

from pydynaa import EventExpression, EventType
from qoala.lang.ehi import EhiNetworkInfo, EhiNetworkTimebin
from qoala.runtime.lhi import LhiLinkInfo
from qoala.runtime.message import Message
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.entdist.entdistinterface import EntDistInterface
from qoala.sim.events import EPR_DELIVERY, CUTOFF_REACHED
from qoala.util.logging import LogManager


@dataclass(frozen=True)
class EntDistRequest:
    local_node_id: int
    remote_node_id: int
    local_qubit_id: int
    local_pid: int
    remote_pid: int
    local_batch_id: int
    remote_batch_id: int

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
                bin.batch_ids[self.local_node_id] == self.local_batch_id
                and bin.batch_ids[self.remote_node_id] == self.remote_batch_id
        )


@dataclass(frozen=True)
class WindowedEntDistRequest(EntDistRequest):
    local_qubit_id: List[int]
    window: int
    num_pairs: int


@dataclass(frozen=True)
class JointRequest:
    node1_id: int
    node2_id: int
    node1_qubit_id: int
    node2_qubit_id: int
    node1_pid: int
    node2_pid: int
    node1_batch_id: int
    node2_batch_id: int


@dataclass(frozen=True)
class WindowedJointRequest(JointRequest):
    node1_qubit_id: List[int]
    node2_qubit_id: List[int]
    window: int
    num_pairs: int


@dataclass
class OutstandingRequest:
    request: JointRequest
    end_of_qc: float
    link_generation_times: List[float]
    next_qubit_index: Optional[int] = None


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
class TimedEntDistRequest:
    request: EntDistRequest


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

        self._outstanding_requests: List[OutstandingRequest] = []  # OutstandingRequest(req., end of QC,link_gen_times)
        self._outstanding_generated_pairs: Dict[float, List[Tuple[
            EprDeliverySample, OutstandingRequest]]] = {}  # delay:(sample, req.); a request in here means there is a pair to deliver in this time slot.

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

    def test_single_time_slot(
            self,
            node1_id: int,
            node2_id: int,
    ) -> Tuple[Optional[EventType], Optional[float], Optional[EprDeliverySample]]:
        timed_sampler = self.get_sampler(node1_id, node2_id)
        sample = self.sample_state(timed_sampler.sampler)

        if sample.duration != 0.:  # Sample time is zero if there were no failures, i.e. the state was immediately produced.

            return None, None, None

        else:

            return EPR_DELIVERY, timed_sampler.delay, sample

    def deliver_outstanding_pairs(self) -> Generator[EventExpression, None, None]:

        delivery_times = list(self._outstanding_generated_pairs.keys())
        delivery_times.sort()

        delivery_times_offset = [delivery_times[0]] + (
            [delivery_times[i] - delivery_times[i - 1] for i in range(1, len(delivery_times))] if len(
                delivery_times) > 1 else [])

        assert len(delivery_times) == len(delivery_times_offset)

        for delay_to_next_pair, real_delay in zip(delivery_times_offset, delivery_times):
            self._schedule_after(delay_to_next_pair, EPR_DELIVERY)
            event_expr = EventExpression(source=self, event_type=EPR_DELIVERY)

            yield event_expr

            now = ns.sim_time()

            for sample, outstanding_request in self._outstanding_generated_pairs[real_delay]:

                epr = self.create_epr_pair_with_state(sample.state)

                node1_id = outstanding_request.request.node1_id
                node2_id = outstanding_request.request.node2_id

                node1_pid = outstanding_request.request.node1_pid
                node2_pid = outstanding_request.request.node2_pid

                if isinstance(outstanding_request.request, WindowedJointRequest):

                    node1_phys_id = outstanding_request.request.node1_qubit_id[outstanding_request.next_qubit_index]
                    node2_phys_id = outstanding_request.request.node2_qubit_id[outstanding_request.next_qubit_index]

                    outstanding_request.next_qubit_index = (outstanding_request.next_qubit_index + 1) % len(
                        outstanding_request.request.node2_qubit_id)

                else:

                    node1_phys_id = outstanding_request.request.node1_qubit_id
                    node2_phys_id = outstanding_request.request.node2_qubit_id

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

                node1_mem.put(qubits=epr[0], positions=node1_phys_id)
                node2_mem.put(qubits=epr[1], positions=node2_phys_id)

                self._logger.warning(f"Created pair for demand {(node1_id, node1_pid, node2_id, node2_pid)}")

                outstanding_request.link_generation_times.append(now)

                satisfied_packet = False

                if isinstance(outstanding_request.request, WindowedJointRequest):
                    for t in outstanding_request.link_generation_times:
                        if now - t > outstanding_request.request.window:
                            outstanding_request.link_generation_times.remove(t)
                            self._logger.warning(
                                f"Dumped pair created at time {t} for demand {(node1_id, node1_pid, node2_id, node2_pid)} due to exceeding window")

                    if len(outstanding_request.link_generation_times) == outstanding_request.request.num_pairs:
                        satisfied_packet = True

                else:
                    satisfied_packet = True  # Non-windowed packet should be single pair only...

                if satisfied_packet:
                    node1 = self._interface.remote_id_to_peer_name(node1_id)
                    node2 = self._interface.remote_id_to_peer_name(node2_id)
                    # TODO: use PIDs??
                    self._interface.send_node_msg(node1, Message(-1, -1, node1_pid))
                    self._interface.send_node_msg(node2, Message(-1, -1, outstanding_request.request.node2_pid))

                    self._logger.warning(
                        f"Packet of entanglement successfully created for demand {(node1_id, node1_pid, node2_id, node2_pid)}")

                    self._outstanding_requests.remove(outstanding_request)

            self._outstanding_generated_pairs.pop(real_delay)

        assert len(self._outstanding_generated_pairs.keys()) == 0

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

    def deliver_with_failure(
            self,
            node1_id: int,
            node1_phys_id: int,
            node2_id: int,
            node2_phys_id: int,
            node1_pid: int,
            node2_pid: int,
            cutoff: float | None = None,
    ) -> Generator[EventExpression, None, None]:
        """
        Will return a failure if the length of time it would take to generate an entangled link is longer than the time allowed by cutoff. If cutoff is None, then will attempt to get the cutoff from the network schedule.
        """
        timed_sampler = self.get_sampler(node1_id, node2_id)
        sample = self.sample_state(timed_sampler.sampler)
        epr = self.create_epr_pair_with_state(sample.state)

        self._logger.info(f"sample duration: {sample.duration}")
        self._logger.info(f"total duration: {timed_sampler.delay}")
        total_delay = sample.duration + timed_sampler.delay

        if cutoff is None and (
                self._netschedule.length_of_qc_blocks is not None if self._netschedule is not None else False):
            try:
                cutoff = self._netschedule.length_of_qc_blocks[(node1_id, node1_pid, node2_id, node2_pid)]
                self._logger.warning(
                    f"Set max QC length to {cutoff}")
            except KeyError:
                cutoff = None
                self._logger.warning(
                    f"No known QC length for session {(node1_id, node1_pid, node2_id, node2_pid)}, defaulting to None")

        if cutoff is not None:
            if total_delay > cutoff:
                self._schedule_after(cutoff - 1, CUTOFF_REACHED)
                #  Cutoff-1 to stop off-by-one errors once it tries to request the next entanglement.
                event_expr = EventExpression(source=self, event_type=CUTOFF_REACHED)
                yield event_expr

                node1 = self._interface.remote_id_to_peer_name(node1_id)
                node2 = self._interface.remote_id_to_peer_name(node2_id)
                # TODO: use PIDs??
                self._interface.send_node_msg(node1, Message(-1, -1, None))
                self._interface.send_node_msg(node2, Message(-1, -1, None))

                self._logger.warning("Entanglement Generation Failed")

                return

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

        self._logger.warning("pair delivered")

        node1_mem.put(qubits=epr[0], positions=node1_phys_id)
        node2_mem.put(qubits=epr[1], positions=node2_phys_id)

        # Send messages to the nodes indictating a request has been delivered.
        node1 = self._interface.remote_id_to_peer_name(node1_id)
        node2 = self._interface.remote_id_to_peer_name(node2_id)
        # TODO: use PIDs??
        self._interface.send_node_msg(node1, Message(-1, -1, node1_pid))
        self._interface.send_node_msg(node2, Message(-1, -1, node2_pid))

    def deliver_with_failure_windowed(
            self,
            node1_id: int,
            node1_phys_id: List[int],
            node2_id: int,
            node2_phys_id: List[int],
            node1_pid: int,
            node2_pid: int,
            window: int,
            num_pairs: int,
            cutoff: float | None = None,
    ) -> Generator[EventExpression, None, None]:
        """
        Will return a failure if the length of time it would take to generate an entangled link is longer than the time allowed by cutoff. If cutoff is None, then will attempt to get the cutoff from the network schedule.
        """

        if cutoff is None and (
                self._netschedule.length_of_qc_blocks is not None if self._netschedule is not None else False):
            try:
                cutoff = self._netschedule.length_of_qc_blocks[(node1_id, node1_pid, node2_id, node2_pid)]
                self._logger.warning(
                    f"Set max QC length to {cutoff}")
            except KeyError:
                cutoff = None
                self._logger.warning(
                    f"No known QC length for session {(node1_id, node1_pid, node2_id, node2_pid)}, defaulting to None")

        assert num_pairs == len(node1_phys_id) == len(node2_phys_id)

        total_elapsed_time_in_request = 0
        links_generated = -1
        link_generation_time_dictionary: Dict[int, Union[int, None]] = {x: None for x in range(
            num_pairs)}  # Just need to track the ages on one side.
        number_of_links_alive = 0

        node1_mem = self._nodes[node1_id].qmemory
        node2_mem = self._nodes[node2_id].qmemory

        while True:
            timed_sampler = self.get_sampler(node1_id, node2_id)
            sample = self.sample_state(timed_sampler.sampler)
            links_generated += 1
            number_of_links_alive += 1

            self._logger.info(f"sample duration: {sample.duration}")
            self._logger.info(f"total duration: {timed_sampler.delay}")

            time_to_get_pair = sample.duration + timed_sampler.delay
            total_elapsed_time_in_request += sample.duration + timed_sampler.delay

            self._logger.info(
                f"Next pair ({links_generated}) would be generated at time {total_elapsed_time_in_request} after start of PGA")

            if not (total_elapsed_time_in_request < cutoff if cutoff is not None else True):
                for memory_slot in range(num_pairs):
                    # Clear holds on memory slots.
                    node1_mem.mem_positions[node1_phys_id[memory_slot]].in_use = False
                    node2_mem.mem_positions[node2_phys_id[memory_slot]].in_use = False

                time_to_cutoff = cutoff - (total_elapsed_time_in_request - time_to_get_pair)

                self._schedule_after(time_to_cutoff - 1, CUTOFF_REACHED)
                #  Cutoff-1 to stop off-by-one errors once it tries to request the next entanglement.
                event_expr = EventExpression(source=self, event_type=CUTOFF_REACHED)
                yield event_expr

                node1 = self._interface.remote_id_to_peer_name(node1_id)
                node2 = self._interface.remote_id_to_peer_name(node2_id)
                # TODO: use PIDs??
                self._interface.send_node_msg(node1, Message(-1, -1, None))
                self._interface.send_node_msg(node2, Message(-1, -1, None))

                self._logger.warning("Entanglement Packet Generation Failed")

                return

            memory_slot = links_generated % num_pairs
            epr = self.create_epr_pair_with_state(sample.state)

            self._schedule_after(time_to_get_pair, EPR_DELIVERY)
            #  Cutoff-1 to stop off-by-one errors once it tries to request the next entanglement.
            event_expr = EventExpression(source=self, event_type=EPR_DELIVERY)
            yield event_expr

            self._logger.warning(
                f"Generated pair number {links_generated} for a windowed packet in slot index {memory_slot}")
            link_generation_time_dictionary[memory_slot] = total_elapsed_time_in_request

            if not (0 <= node1_phys_id[memory_slot] < node1_mem.num_positions):
                raise ValueError(
                    f"qubit location id of {node1_phys_id} is not present in \
                                    quantum memory of node ID {node1_id}."
                )
            if not (0 <= node2_phys_id[memory_slot] < node2_mem.num_positions):
                raise ValueError(
                    f"qubit location id of {node2_phys_id} is not present in \
                                    quantum memory of node ID {node2_id}."
                )

            node1_mem.put(qubits=epr[0], positions=node1_phys_id[memory_slot])
            node2_mem.put(qubits=epr[1], positions=node2_phys_id[memory_slot])

            for slot in range(num_pairs):
                if total_elapsed_time_in_request - link_generation_time_dictionary[slot] > window if \
                        link_generation_time_dictionary[slot] is not None else False:
                    # # If links are too old, then release use of memory slot
                    # node1_mem.mem_positions[node1_phys_id[slot]].in_use = False  # TODO: separate bool for "alive"
                    # node2_mem.mem_positions[node2_phys_id[slot]].in_use = False
                    number_of_links_alive -= 1
                    link_generation_time_dictionary[slot] = None
                    self._logger.warning(f"Dumped pair in slot {slot}")

            if number_of_links_alive == num_pairs:
                # Then all slots in use and holding recent enough pairs to form a valid packet:

                self._logger.warning("Packet of entanglement successfully created")
                # Send messages to the nodes indicating a request has been delivered.
                node1 = self._interface.remote_id_to_peer_name(node1_id)
                node2 = self._interface.remote_id_to_peer_name(node2_id)

                node1_mem.mem_positions[node1_phys_id[memory_slot]].in_use = True
                node2_mem.mem_positions[node2_phys_id[memory_slot]].in_use = True

                # TODO: use PIDs??
                self._interface.send_node_msg(node1, Message(-1, -1, node1_pid))
                self._interface.send_node_msg(node2, Message(-1, -1, node2_pid))

                return

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
                if isinstance(local_request, WindowedEntDistRequest):

                    assert isinstance(remote_request, WindowedEntDistRequest)
                    assert remote_request.window == local_request.window

                    return WindowedJointRequest(
                        node1_id=local_request.local_node_id,
                        node2_id=remote_request.local_node_id,
                        node1_qubit_id=local_request.local_qubit_id,
                        node2_qubit_id=remote_request.local_qubit_id,
                        node1_pid=local_request.local_pid,
                        node2_pid=remote_request.local_pid,
                        window=local_request.window,
                        num_pairs=local_request.num_pairs,
                        node1_batch_id=local_request.local_batch_id,
                        node2_batch_id=remote_request.local_batch_id,
                    )
                else:
                    return JointRequest(
                        local_request.local_node_id,
                        remote_request.local_node_id,
                        local_request.local_qubit_id,
                        remote_request.local_qubit_id,
                        local_request.local_pid,
                        local_request.remote_pid,
                        node1_batch_id=local_request.local_batch_id,
                        node2_batch_id=remote_request.local_batch_id,
                    )
            else:
                # Put local request back
                local_requests.insert(0, local_request)

        # No joint requests found
        return None

    def get_all_joint_requests(self) -> Tuple[Optional[List[JointRequest]], List[int]]:
        _all_joint_requests: List[JointRequest] = []
        while True:
            next_joint_request = self.get_next_joint_request()
            if next_joint_request is not None:
                _all_joint_requests.append(next_joint_request)
                self._logger.warning(
                    f"Accepted joint request from demand {(next_joint_request.node1_id, next_joint_request.node1_pid, next_joint_request.node2_id, next_joint_request.node2_pid)}")
            else:
                break

        outstanding_nodes = [x for x in self._requests.keys() if self._requests[x] != []]

        return (_all_joint_requests if _all_joint_requests != [] else None), outstanding_nodes

    def serve_request(
            self, request: JointRequest, fixed_length_qc_blocks: bool = False, cutoff: Union[int, None] = None
    ) -> Generator[EventExpression, None, None]:
        if fixed_length_qc_blocks:
            if isinstance(request, WindowedJointRequest):
                yield from self.deliver_with_failure_windowed(
                    node1_id=request.node1_id,
                    node1_phys_id=request.node1_qubit_id,
                    node2_id=request.node2_id,
                    node2_phys_id=request.node2_qubit_id,
                    node1_pid=request.node1_pid,
                    node2_pid=request.node2_pid,
                    window=request.window,
                    num_pairs=request.num_pairs,
                    cutoff=cutoff
                )
            else:
                yield from self.deliver_with_failure(
                    node1_id=request.node1_id,
                    node1_phys_id=request.node1_qubit_id,
                    node2_id=request.node2_id,
                    node2_phys_id=request.node2_qubit_id,
                    node1_pid=request.node1_pid,
                    node2_pid=request.node2_pid,
                    cutoff=cutoff,
                )
        else:
            if isinstance(request, WindowedJointRequest):
                yield from self.deliver_with_failure_windowed(
                    node1_id=request.node1_id,
                    node1_phys_id=request.node1_qubit_id,
                    node2_id=request.node2_id,
                    node2_phys_id=request.node2_qubit_id,
                    node1_pid=request.node1_pid,
                    node2_pid=request.node2_pid,
                    window=request.window,
                    num_pairs=request.num_pairs,
                    cutoff=cutoff
                )
            else:
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

    def _run_with_netschedule_parallel_one_epr_per_timeslot(self) -> Generator[EventExpression, None, None]:
        """
        This method is designed to allow the distribution of entanglement to many QC blocks in parallel. It does this
        in the following manner:

        We add an object OutstandingRequest, which holds the request and information about when it expires and what
        entanglement has already been generated to help satisfy it (see above for more information).

        We add the list self._oustanding_requests. This list contains the QC blocks which are currently being executed
        by the nodes and network.

        We add the dictionary self._outstanding_generated_pairs. This is a dictionary from the state_delay to a list of
        (sample, Outstanding request pairs), which are to be delivered in the current time slot.

        We allow the network schedule bin pattern to be of the form [[b1,b2],[b3,b4],...,], and check admissibility by
        whether the demand in question matches at least one of b1,b2

        At the start of each time slot we do the following:
        1. Check for new requests and whether there exists a matching timebin in the network schedule. If so put the
            request onto the node.

        2. We get all joint requests which have been submitted. If there are requests which cannot be formed into joint
            requests, then we inform the nodes their request has been rejected. Otherwise, for each new joint request
            create a new outstanding request, gathering end of QC info from the network schedule if possible, else no
            end of QC block.

        3. For each outstanding request, we test if it has expired, and if so inform the nodes and remove it from the
            outstanding requests. Otherwise, we conduct a (biased) coin flip to determine if an epr pair was generated
            for that demand in this timeslot.

            If an EPR pair was created, we add it to the dictionary of outstanding generated pairs under the correct delay "header", along with the corresponding OutstandingRequest for ease of access.

        4. We run the generator self.deliver_outstanding_pairs. This does the following:
            For each delay header, we schedule a simulator event at the delay time after the start of the time bin, and
            wait until the sim time advances to this time. Then for each request which gets a new pair at this time:
                a) we allocate the EPR pair generated at this time
                b) we check for pairs which are too old to satisfy the window, if applicable
                c) we check for packet satisfaction. If so, inform the nodes and remove the request from the list of
                   outstanding requests.

            If there are no outstanding requests, we move the clock on one timestep to avoid getting stuck until
            something happens.  [This may be removed if it causes issues once nodes are running properly, but required
            for testing the EntDist.]



        """

        assert self._netschedule is not None

        while True:

            if not self._outstanding_requests:
                # Wait until a message arrives.
                yield from self._interface.wait_for_any_msg()

            # Wait until the next time bin.
            now = ns.sim_time()
            # self._logger.warning(now)
            next_slot_time, next_slot = self._netschedule.next_bin(now)

            if next_slot_time - now > 0:
                yield from self._interface.wait(next_slot_time - now)
            elif next_slot_time - now < 0:
                raise RuntimeError()

            # print(ns.sim_time())

            messages = self._interface.pop_all_messages()

            requesting_nodes: List[int] = []
            wrong_timebin_nodes: List[int] = []
            served_batches: List[Tuple[int,int]] = []
            for msg in messages:
                self._logger.warning(f"received new msg from node: {msg}")
                request: EntDistRequest = msg.content
                requesting_nodes.append(request.local_node_id)

                if isinstance(next_slot, list):

                    if any(request.matches_timebin(b) for b in next_slot):
                        self._logger.warning(f"putting request: {request}")
                        self.put_request(request)

                    else:
                        self._logger.warning(f"not handling msg {msg} (wrong timebin - {next_slot})")
                        wrong_timebin_nodes.append(request.local_node_id)
                else:
                    if request.matches_timebin(next_slot):
                        self._logger.warning(f"putting request: {request}")
                        self.put_request(request)
                        served_batches.append((request.local_batch_id,request.remote_batch_id))
                    else:
                        self._logger.warning(f"not handling msg {msg} (wrong timebin - {next_slot})")
                        wrong_timebin_nodes.append(request.local_node_id)

            all_new_joint_requests, outstanding_nodes = self.get_all_joint_requests()

            for node_id in outstanding_nodes:
                node = self._interface.remote_id_to_peer_name(node_id)
                self._interface.send_node_msg(node, Message(-1, -1, None))
                self._logger.info(f"sending message to reject demand from node {node_id} as no joint request")
            self.clear_requests()

            for node_id in wrong_timebin_nodes:
                node = self._interface.remote_id_to_peer_name(node_id)
                self._interface.send_node_msg(node, Message(-1, -1, None))
                self._logger.info(f"Sending message rejecting demand from node {node_id} as in wrong timebin")
            self.clear_requests()

            if all_new_joint_requests:

                for request in all_new_joint_requests:
                    request: JointRequest
                    try:
                        self._outstanding_requests.append(OutstandingRequest(request, ns.sim_time() +
                                                                             self._netschedule.length_of_qc_blocks[(
                                                                                 request.node1_id, request.node1_batch_id,
                                                                                 request.node2_id,
                                                                                 request.node2_batch_id)] - 1,
                                                                             [], 0))
                    except KeyError as F:
                        self._logger.warning(
                            f"No specified QC block length for {(request.node1_id, request.node1_pid, request.node2_id, request.node2_pid)}, defaulting to None")
                        self._outstanding_requests.append(OutstandingRequest(request, math.inf, []))

            # Determine if a request has expired, or if a pair is delivered for that request.

            for outstanding_request in self._outstanding_requests:
                outcome, delay, sample = self.test_single_time_slot(
                    outstanding_request.request.node1_id,
                    outstanding_request.request.node2_id
                )
                if outcome is not None:
                    if delay in self._outstanding_generated_pairs.keys():
                        self._outstanding_generated_pairs[delay].append((sample, outstanding_request))
                    else:
                        self._outstanding_generated_pairs[delay] = [(sample, outstanding_request)]

            if self._outstanding_generated_pairs:
                yield from self.deliver_outstanding_pairs()

            now = ns.sim_time()

            next_slot_time, _ = self._netschedule.next_bin(now, future=True)
            if next_slot_time - now - 1 > 0:
                yield from self._interface.wait(
                    next_slot_time - now - 1)  # Should advance to 1ns before the end of current time slot.

            expired_requests = []
            for outstanding_request in self._outstanding_requests:

                if outstanding_request.end_of_qc <= ns.sim_time():
                    node1 = self._interface.remote_id_to_peer_name(outstanding_request.request.node1_id)
                    node2 = self._interface.remote_id_to_peer_name(outstanding_request.request.node2_id)



                    self._interface.send_node_msg(node1, Message(-1, -1, None))
                    self._interface.send_node_msg(node2, Message(-1, -1, None))

                    self._logger.warning(
                        f"Entanglement Packet Generation Failed for demand {(outstanding_request.request.node1_id, outstanding_request.request.node1_pid, outstanding_request.request.node2_id, outstanding_request.request.node2_pid)}")
                    expired_requests.append(outstanding_request)

            for e in expired_requests:
                self._outstanding_requests.remove(
                    e)  # Need to do this with a separate list rather than removing in iterator above as otherwise does not iterate over the entire list,
                # causing some requests to survive for an extra  timestep.

    def _run_with_netschedule(self) -> Generator[EventExpression, None, None]:
        assert self._netschedule is not None

        while True:
            # Wait until a message arrives.
            yield from self._interface.wait_for_any_msg()

            # Wait until the next time bin.
            now = ns.sim_time()
            # self._logger.warning(now)
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

                if isinstance(next_slot, list):
                    raise NotImplementedError  # This should use
                    # self._run_with_netschedule_parallel_one_epr_per_timeslot() instead.

                if request.matches_timebin(next_slot):
                    self._logger.warning(f"putting request: {request}")
                    self.put_request(request)
                else:
                    self._logger.warning(f"not handling msg {msg} (wrong timebin)")
            joint_request = self.get_next_joint_request()
            if joint_request is not None:
                self._logger.warning("serving request")
                yield from self.serve_request(joint_request, fixed_length_qc_blocks=True)
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
            yield from self._run_with_netschedule_parallel_one_epr_per_timeslot()
