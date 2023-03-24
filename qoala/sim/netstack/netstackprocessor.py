import logging
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Union

import netsquid as ns
from netqasm.sdk.build_epr import (
    SER_RESPONSE_KEEP_IDX_BELL_STATE,
    SER_RESPONSE_KEEP_IDX_GOODNESS,
    SER_RESPONSE_KEEP_LEN,
)
from netsquid.components.instructions import INSTR_MEASURE, INSTR_ROT_X, INSTR_ROT_Z
from netsquid.qubits.ketstates import BellIndex
from qlink_interface import (
    ReqCreateAndKeep,
    ReqCreateBase,
    ReqMeasureDirectly,
    ReqReceive,
    ReqRemoteStatePrep,
    ResCreateAndKeep,
)

from pydynaa import EventExpression
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    IqoalaRequest,
    RequestRoutine,
)
from qoala.runtime.memory import ProgramMemory, RunningRequestRoutine, SharedMemory
from qoala.runtime.message import Message
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.entdist.entdist import EntDistRequest
from qoala.sim.memmgr import AllocError
from qoala.sim.netstack.netstackinterface import NetstackInterface, NetstackLatencies
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice, QDeviceCommand
from qoala.sim.requests import (
    NetstackBreakpointCreateRequest,
    NetstackBreakpointReceiveRequest,
    T_NetstackRequest,
)
from qoala.sim.signals import MSG_REQUEST_DELIVERED
from qoala.util.constants import PI
from qoala.util.logging import LogManager


class NetstackProcessor:
    def __init__(
        self, interface: NetstackInterface, latencies: NetstackLatencies
    ) -> None:
        self._interface = interface
        self._latencies = latencies

        self._name = f"{interface.name}_NetstackProcessor"

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({self._name})"
        )

        # memory of current program, only not-None when processor is active
        self._current_prog_mem: Optional[ProgramMemory] = None

        self._current_routine: Optional[RunningRequestRoutine] = None

    def _prog_mem(self) -> ProgramMemory:
        # May only be called when processor is active
        assert self._current_prog_mem is not None
        return self._current_prog_mem

    def _routine(self) -> RunningRequestRoutine:
        # May only be called when processor is active
        assert self._current_routine is not None
        return self._current_routine

    @property
    def qdevice(self) -> QDevice:
        return self._interface.qdevice

    def get_shared_mem(self, pid: int) -> SharedMemory:
        prog_mem = self._interface.memmgr.get_process(pid).prog_memory
        return prog_mem.shared_mem

    def _create_link_layer_create_request(
        self, request: IqoalaRequest
    ) -> ReqCreateBase:
        """Construct a link layer request from application request info.

        :param remote_id: ID of remote node
        :param args: NetQASM array elements from the arguments array specified by the
            application
        :return: link layer request object
        """

        if request.typ == EprType.CREATE_KEEP:
            ll_request = ReqCreateAndKeep(
                remote_node_id=request.remote_id,
                number=request.num_pairs,
                minimum_fidelity=request.fidelity,
            )
        elif request.typ == EprType.MEASURE_DIRECTLY:
            ll_request = ReqMeasureDirectly(
                remote_node_id=request.remote_id,
                number=request.num_pairs,
                minimum_fidelity=request.fidelity,
            )
        elif request.typ == EprType.REMOTE_STATE_PREP:
            ll_request = ReqRemoteStatePrep(
                remote_node_id=request.remote_id,
                number=request.num_pairs,
                minimum_fidelity=request.fidelity,
            )
        else:
            raise ValueError(f"Unsupported create type {request.typ}")
        return ll_request

    def assign(
        self, process: IqoalaProcess, request: Union[T_NetstackRequest, IqoalaRequest]
    ) -> Generator[EventExpression, None, None]:
        if isinstance(request, IqoalaRequest):
            if request.role == EprRole.CREATE:
                yield from self.handle_create_request(process, request)
                self._logger.debug("create request done")
            elif request.role == EprRole.RECEIVE:
                yield from self.handle_receive_request(process, request)
                self._logger.debug("receive request done")
            else:
                raise RuntimeError
        elif isinstance(request, NetstackBreakpointCreateRequest):
            yield from self.handle_breakpoint_create_request(request)
            self._logger.debug("breakpoint create request done")
        elif isinstance(request, NetstackBreakpointReceiveRequest):
            yield from self.handle_breakpoint_receive_request(request)
            self._logger.debug("breakpoint receive request done")
        else:
            raise RuntimeError

    def handle_create_request(
        self, process: IqoalaProcess, req: IqoalaRequest
    ) -> Generator[EventExpression, None, None]:
        """Issue a request to create entanglement with a remote node.

        :param req: request info
        """

        # Synchronize with the remote node.

        # Send the request to the receiver node and wait for an acknowledgement.
        peer = self._interface.remote_id_to_peer_name(req.remote_id)
        self._interface.send_peer_msg(peer, Message(content=req))
        peer_msg = yield from self._interface.receive_peer_msg(peer)
        yield from self._interface.wait(self._latencies.netstack_peer_latency)
        self._logger.debug(f"received peer msg: {peer_msg}")

        # Handle the request.
        if req.typ == EprType.CREATE_KEEP:
            yield from self.handle_create_ck_request(process, req)
        elif req.typ == EprType.MEASURE_DIRECTLY:
            raise NotImplementedError
            # TODO: Fix MD handling
            # yield from self.handle_create_md_request(process, req)
        else:
            raise RuntimeError

    def handle_create_ck_request(
        self, process: IqoalaProcess, req: IqoalaRequest
    ) -> Generator[EventExpression, None, None]:
        """Handle a Create and Keep request as the initiator/creator, until all
        pairs have been created.

        This method uses the EGP protocol to create and measure EPR pairs with
        the remote node. It will fully complete the request before returning. If
        the pair created by the EGP protocol is another Bell state than Phi+,
        local gates are applied to do a correction, such that the final
        delivered pair is always Phi+.

        The method can however yield (i.e. give control back to the simulator
        scheduler) in the following cases: - no communication qubit is
        available; this method will resume when a
          SIGNAL_MEMORY_FREED is given (currently only the processor can do
          this)
        - when waiting for the EGP protocol to produce the next pair; this
          method resumes when the pair is delivered
        - a Bell correction gate is applied

        This method does not return anything. This method has the side effect
        that NetQASM array value are written to.

        :param req: application request info (app ID and NetQASM array IDs)
        :param request: link layer request object
        """
        num_pairs = req.num_pairs

        self._logger.info(f"putting CK request to EGP for {num_pairs} pairs")
        self._logger.info(f"qubit IDs specified by application: {req.virt_ids}")
        self._logger.info(f"splitting request into {num_pairs} 1-pair requests")

        start_time = ns.sim_time()

        for pair_index in range(num_pairs):
            virt_id = req.virt_ids.get_id(pair_index)
            ll_result = yield from self.create_single_pair(
                process, req, virt_id, wait_for_free=True
            )

            gen_duration = ns.sim_time() - start_time

            self.write_pair_result(
                process,
                ll_result,
                pair_index,
                req.result_array_addr,
                gen_duration,
            )

            self._interface.send_qnos_msg(Message(content="wrote to array"))

    def create_single_pair(
        self,
        process: IqoalaProcess,
        request: IqoalaRequest,
        virt_id: int,
        wait_for_free: bool = False,
    ) -> Generator[EventExpression, None, ResCreateAndKeep]:
        """
        :param wait_for_free: whether to wait (block) on a "memory free" signal in case
            of an AllocError
        """
        ll_request = self._create_link_layer_create_request(request)
        ll_request.number = 1

        self._logger.info("trying to allocate comm qubit")
        while True:
            try:
                self._interface.memmgr.allocate_comm(process.pid, virt_id)
                break
            except AllocError:
                if not wait_for_free:
                    raise AllocError

                # else:
                self._logger.info("no comm qubit available, waiting...")

                # Wait for a signal indicating the communication qubit might be free
                # again.
                yield from self._interface.await_memory_freed_signal(
                    process.pid, virt_id
                )
                self._logger.info(
                    "a 'free' happened, trying again to allocate comm qubit..."
                )

        # Put the request to the EGP.
        self._logger.info("putting CK request")
        self._interface.put_request(request.remote_id, ll_request)

        # Wait for a signal from the EGP.
        self._logger.info("waiting for result")
        result = yield from self._interface.await_result_create_keep(request.remote_id)
        self._logger.info(f"got result: {result}")

        # Bell state corrections. Resulting state is always Phi+ (i.e. B00).
        if result.bell_state == BellIndex.B00:
            pass
        elif result.bell_state == BellIndex.B01:
            commands = [QDeviceCommand(INSTR_ROT_X, [0], angle=PI)]
            yield from self._interface.qdevice.execute_commands(commands)
        elif result.bell_state == BellIndex.B10:
            commands = [QDeviceCommand(INSTR_ROT_Z, [0], angle=PI)]
            yield from self._interface.qdevice.execute_commands(commands)
        elif result.bell_state == BellIndex.B11:
            commands = [
                QDeviceCommand(INSTR_ROT_X, [0], angle=PI),
                QDeviceCommand(INSTR_ROT_Z, [0], angle=PI),
            ]
            yield from self._interface.qdevice.execute_commands(commands)

        return result

    def write_pair_result(
        self,
        process: IqoalaProcess,
        ll_result: ResCreateAndKeep,
        pair_index: int,
        array_addr: int,
        duration: float,  # in ns
    ) -> None:
        shared_mem = process.prog_memory.shared_mem

        gen_duration_us_int = int(duration / 1000)
        self._logger.info(f"gen duration (us): {gen_duration_us_int}")

        # Length of response array slice for a single pair.
        slice_len = SER_RESPONSE_KEEP_LEN

        # Populate results array.
        for i in range(slice_len):
            # Write -1 to unused array elements.
            value = -1

            # Write corresponding result value to the other array elements.
            if i == SER_RESPONSE_KEEP_IDX_GOODNESS:
                value = gen_duration_us_int
            if i == SER_RESPONSE_KEEP_IDX_BELL_STATE:
                value = ll_result.bell_state

            # Calculate array element location.
            arr_index = slice_len * pair_index + i

            shared_mem.set_array_value(array_addr, arr_index, value)
        self._logger.debug(
            f"wrote to @{array_addr}[{slice_len * pair_index}:"
            f"{slice_len * pair_index + slice_len}] for app ID {process.pid}"
        )

    # def handle_create_md_request(
    #     self, req: NetstackCreateRequest, request: ReqMeasureDirectly
    # ) -> Generator[EventExpression, None, None]:
    #     """Handle a Create and Measure request as the initiator/creator, until all
    #     pairs have been created and measured.

    #     This method uses the EGP protocol to create EPR pairs with the remote node.
    #     It will fully complete the request before returning.

    #     No Bell state corrections are done. This means that application code should
    #     use the result information to check, for each pair, the generated Bell state
    #     and possibly post-process the measurement outcomes.

    #     The method can yield (i.e. give control back to the simulator scheduler) in
    #     the following cases:
    #     - no communication qubit is available; this method will resume when a
    #       SIGNAL_MEMORY_FREED is given (currently only the processor can do this)
    #     - when waiting for the EGP protocol to produce the next pair; this method
    #       resumes when the pair is delivered

    #     This method does not return anything.
    #     This method has the side effect that NetQASM array value are written to.

    #     :param req: application request info (app ID and NetQASM array IDs)
    #     :param request: link layer request object
    #     """

    #     # Put the reqeust to the EGP.
    #     self._interface.put_request(req.remote_node_id, request)

    #     results: List[ResMeasureDirectly] = []

    #     # Wait for all pairs to be created. For each pair, the EGP sends a separate
    #     # signal that is awaited here. Only after the last pair, we write the results
    #     # to the array. This is done since the whole request (i.e. all pairs) is
    #     # expected to finish in a short time anyway. However, writing results for a
    #     # pair as soon as they are done may be implemented in the future.
    #     for _ in range(request.number):
    #         self._interface.memmgr.allocate(req.pid, 0)

    #         result = yield from self._interface.await_result_measure_directly(
    #             req.remote_node_id
    #         )
    #         self._logger.debug(f"bell index: {result.bell_state}")
    #         results.append(result)
    #         self._interface.memmgr.free(req.pid, 0)

    #     shared_mem = self.get_shared_mem(req.pid)

    #     # Length of response array slice for a single pair.
    #     slice_len = SER_RESPONSE_MEASURE_LEN

    #     # Populate results array.
    #     for pair_index in range(request.number):
    #         result = results[pair_index]

    #         for i in range(slice_len):
    #             # Write -1 to unused array elements.
    #             value = -1

    #             # Write corresponding result value to the other array elements.
    #             if i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_OUTCOME:
    #                 value = result.measurement_outcome
    #             elif i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_BASIS:
    #                 value = result.measurement_basis.value
    #             elif i == SER_RESPONSE_KEEP_IDX_BELL_STATE:
    #                 value = result.bell_state.value

    #             # Calculate array element location.
    #             arr_index = slice_len * pair_index + i

    #             shared_mem.set_array_value(req.result_array_addr, arr_index, value)

    #     self._interface.send_qnos_msg(Message(content="wrote to array"))

    def handle_receive_ck_request(
        self, process: IqoalaProcess, req: IqoalaRequest
    ) -> Generator[EventExpression, None, None]:
        """Handle a Create and Keep request as the receiver, until all pairs have
        been created.

        This method uses the EGP protocol to create EPR pairs with the remote
        node. It will fully complete the request before returning.

        If the pair created by the EGP protocol is another Bell state than Phi+,
        it is assumed that the *other* node applies local gates such that the
        final delivered pair is always Phi+.

        The method can yield (i.e. give control back to the simulator scheduler)
        in the following cases: - no communication qubit is available; this
        method will resume when a
          SIGNAL_MEMORY_FREED is given (currently only the processor can do
          this)
        - when waiting for the EGP protocol to produce the next pair; this
          method resumes when the pair is delivered

        This method does not return anything. This method has the side effect
        that NetQASM array value are written to.

        :param req: application request info (app ID and NetQASM array IDs)
        :param request: link layer request object
        """

        num_pairs = req.num_pairs

        self._logger.info(f"putting CK request to EGP for {num_pairs} pairs")
        self._logger.info(f"splitting request into {num_pairs} 1-pair requests")

        start_time = ns.sim_time()

        for pair_index in range(num_pairs):
            virt_id = req.virt_ids.get_id(pair_index)
            ll_result = yield from self.receive_single_pair(
                process, req, virt_id, wait_for_free=True
            )

            gen_duration = ns.sim_time() - start_time
            self.write_pair_result(
                process, ll_result, pair_index, req.result_array_addr, gen_duration
            )

            self._interface.send_qnos_msg(Message(content="wrote to array"))

    def receive_single_pair(
        self,
        process: IqoalaProcess,
        request: IqoalaRequest,
        virt_id: int,
        wait_for_free: bool = False,
    ) -> Generator[EventExpression, None, ResCreateAndKeep]:
        self._logger.info("trying to allocate comm qubit")
        while True:
            try:
                self._interface.memmgr.allocate_comm(process.pid, virt_id)
                break
            except AllocError:
                if not wait_for_free:
                    raise AllocError

                self._logger.info("no comm qubit available, waiting...")

                # Wait for a signal indicating the communication qubit might be free
                # again.
                yield from self._interface.await_memory_freed_signal(
                    process.pid, virt_id
                )
                self._logger.info(
                    "a 'free' happened, trying again to allocate comm qubit..."
                )

        # Put the request to the EGP.
        self._logger.info("putting CK request")
        self._interface.put_request(request.remote_id, ReqReceive(request.remote_id))
        self._logger.info("waiting for result")

        # Wait for a signal from the EGP.
        result = yield from self._interface.await_result_create_keep(request.remote_id)
        self._logger.info(f"got result: {result}")

        return result

    # def handle_receive_md_request(
    #     self, req: NetstackReceiveRequest, request: ReqMeasureDirectly
    # ) -> Generator[EventExpression, None, None]:
    #     """Handle a Create and Measure request as the receiver, until all
    #     pairs have been created and measured.

    #     This method uses the EGP protocol to create EPR pairs with the remote node.
    #     It will fully complete the request before returning.

    #     No Bell state corrections are done. This means that application code should
    #     use the result information to check, for each pair, the generated Bell state
    #     and possibly post-process the measurement outcomes.

    #     The method can yield (i.e. give control back to the simulator scheduler)
    #     in the following cases: - no communication qubit is available; this
    #     method will resume when a
    #       SIGNAL_MEMORY_FREED is given (currently only the processor can do
    #       this)
    #     - when waiting for the EGP protocol to produce the next pair; this
    #       method resumes when the pair is delivered

    #     This method does not return anything. This method has the side effect
    #     that NetQASM array value are written to.

    #     :param req: application request info (app ID and NetQASM array IDs)
    #     :param request: link layer request object
    #     """
    #     assert isinstance(request, ReqMeasureDirectly)

    #     self._interface.put_request(req.remote_node_id, ReqReceive(req.remote_node_id))

    #     results: List[ResMeasureDirectly] = []

    #     for _ in range(request.number):
    #         self._interface.memmgr.allocate(req.pid, 0)

    #         result = yield from self._interface.await_result_measure_directly(
    #             req.remote_node_id
    #         )
    #         results.append(result)

    #         self._interface.memmgr.free(req.pid, 0)

    #     shared_mem = self.get_shared_mem(req.pid)

    #     # Length of response array slice for a single pair.
    #     slice_len = SER_RESPONSE_MEASURE_LEN

    #     # Populate results array.
    #     for pair_index in range(request.number):
    #         result = results[pair_index]

    #         for i in range(slice_len):
    #             # Write -1 to unused array elements.
    #             value = -1

    #             # Write corresponding result value to the other array elements.
    #             if i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_OUTCOME:
    #                 value = result.measurement_outcome
    #             elif i == SER_RESPONSE_MEASURE_IDX_MEASUREMENT_BASIS:
    #                 value = result.measurement_basis.value
    #             elif i == SER_RESPONSE_KEEP_IDX_BELL_STATE:
    #                 value = result.bell_state.value

    #             # Calculate array element location.
    #             arr_index = slice_len * pair_index + i

    #             shared_mem.set_array_value(req.result_array_addr, arr_index, value)

    #         self._interface.send_qnos_msg(Message(content="wrote to array"))

    def handle_receive_request(
        self, process: IqoalaProcess, req: IqoalaRequest
    ) -> Generator[EventExpression, None, None]:
        """Issue a request to receive entanglement from a remote node.

        :param req: request info
        """

        # Synchronize with the remote node.

        # Wait for the network stack in the remote node to get the corresponding
        # 'create' request from its local application and send it to us.
        # NOTE: we do not check if the request from the other node matches our own
        # request. Also, we simply block until synchronizing with the other node,
        # and then fully handle the request. There is no support for queueing
        # and/or interleaving multiple different requests.
        peer = self._interface.remote_id_to_peer_name(req.remote_id)
        msg = yield from self._interface.receive_peer_msg(peer)
        yield from self._interface.wait(self._latencies.netstack_peer_latency)
        create_request = msg.content
        self._logger.debug(f"received {create_request} from peer")

        # Acknowledge to the remote node that we received the request and we will
        # start handling it.
        self._logger.debug("sending 'ready' to peer")
        self._interface.send_peer_msg(peer, Message(content="ready"))

        if req.typ == EprType.CREATE_KEEP:
            yield from self.handle_receive_ck_request(process, req)
        elif req.typ == EprType.MEASURE_DIRECTLY:
            raise NotImplementedError
            # TODO: Fix MD handling
            # yield from self.handle_receive_md_request(process, req)
        else:
            raise RuntimeError

    def handle_breakpoint_create_request(
        self, request: NetstackBreakpointCreateRequest
    ) -> Generator[EventExpression, None, None]:
        # Use epr sockets for this process to get all relevant remote nodes.
        epr_sockets = self._interface.memmgr.get_process(request.pid).epr_sockets
        remote_ids = [esck.remote_id for esck in epr_sockets.values()]
        remote_names = [self._interface.remote_id_to_peer_name(id) for id in remote_ids]

        # Synchronize with the remote nodes.
        for peer in remote_names:
            self._interface.send_peer_msg(peer, Message(content="breakpoint start"))

        for peer in remote_names:
            response = yield from self._interface.receive_peer_msg(peer)
            # DON'T apply network latency since breakpoints are only meant to check
            # quantum states in simulation and therefore in reality no messages would
            # be exchanged (inducing latency).
            # yield from self._interface.wait(self._latencies.netstack_peer_latency)
            assert response.content == "breakpoint start"

        # Remote nodes are now ready. Notify the processor.
        self._interface.send_qnos_msg(Message(content="breakpoint ready"))

        # Wait for the processor to finish handling the breakpoint.
        processor_msg = yield from self._interface.receive_qnos_msg()
        assert processor_msg.content == "breakpoint end"

        # Tell the remote nodes that the breakpoint has finished.
        for peer in remote_names:
            self._interface.send_peer_msg(peer, Message(content="breakpoint end"))

        # Wait for the remote node to have finsihed as well.
        for peer in remote_names:
            response = yield from self._interface.receive_peer_msg(peer)
            # DON'T apply network latency since breakpoints are only meant to check
            # quantum states in simulation and therefore in reality no messages would
            # be exchanged (inducing latency).
            # yield from self._interface.wait(self._latencies.netstack_peer_latency)
            assert response.content == "breakpoint end"

        # Notify the processor that we are done.
        self._interface.send_qnos_msg(Message(content="breakpoint finished"))

    def handle_breakpoint_receive_request(
        self, request: NetstackBreakpointReceiveRequest
    ) -> Generator[EventExpression, None, None]:
        # Use epr sockets for this process to get all relevant remote nodes.
        epr_sockets = self._interface.memmgr.get_process(request.pid).epr_sockets
        remote_ids = [esck.remote_id for esck in epr_sockets.values()]
        remote_names = [self._interface.remote_id_to_peer_name(id) for id in remote_ids]

        # Synchronize with the remote nodes.
        for peer in remote_names:
            msg = yield from self._interface.receive_peer_msg(peer)
            # DON'T apply network latency since breakpoints are only meant to check
            # quantum states in simulation and therefore in reality no messages would
            # be exchanged (inducing latency).
            # yield from self._interface.wait(self._latencies.netstack_peer_latency)
            assert msg.content == "breakpoint start"

        for peer in remote_names:
            self._interface.send_peer_msg(peer, Message(content="breakpoint start"))

        # Notify the processor we are ready to handle the breakpoint.
        self._interface.send_qnos_msg(Message(content="breakpoint ready"))

        # Wait for the processor to finish handling the breakpoint.
        processor_msg = yield from self._interface.receive_qnos_msg()
        assert processor_msg.content == "breakpoint end"

        # Wait for the remote nodes to finish and tell it we are finished as well.
        for peer in remote_names:
            peer_msg = yield from self._interface.receive_peer_msg(peer)
            # DON'T apply network latency since breakpoints are only meant to check
            # quantum states in simulation and therefore in reality no messages would
            # be exchanged (inducing latency).
            # yield from self._interface.wait(self._latencies.netstack_peer_latency)
            assert peer_msg.content == "breakpoint end"

        for peer in remote_names:
            self._interface.send_peer_msg(peer, Message(content="breakpoint end"))

        # Notify the processor that we are done.
        self._interface.send_qnos_msg(Message(content="breakpoint finished"))

    def execute_entdist_request(
        self, request: EntDistRequest
    ) -> Generator[EventExpression, None, None]:
        self._interface.send_entdist_msg(Message(request))
        result = yield from self._interface.receive_entdist_msg()
        if result.content != MSG_REQUEST_DELIVERED:
            raise RuntimeError("Request was not served")

    def allocate_for_pair(
        self, process: IqoalaProcess, request: IqoalaRequest, index: int
    ) -> int:
        memmgr = self._interface.memmgr
        pid = process.pid

        virt_id = request.virt_ids.get_id(index)
        memmgr.allocate(pid, virt_id)

        return virt_id

    def create_entdist_request(
        self, process: IqoalaProcess, request: IqoalaRequest, virt_id: int
    ) -> EntDistRequest:
        memmgr = self._interface.memmgr
        pid = process.pid
        phys_id = memmgr.phys_id_for(pid, virt_id)

        return EntDistRequest(
            local_node_id=self._interface.node_id,
            remote_node_id=request.remote_id,
            local_qubit_id=phys_id,
        )

    def measure_epr_qubit(
        self, process: IqoalaProcess, virt_id: int
    ) -> Generator[EventExpression, None, int]:
        phys_id = self._interface.memmgr.phys_id_for(process.pid, virt_id)
        # Should have been allocated by `handle_req_routine_md`
        assert phys_id is not None
        commands = [QDeviceCommand(INSTR_MEASURE, [phys_id])]
        m = yield from self.qdevice.execute_commands(commands=commands)
        assert m is not None
        return m

    def handle_req_routine_md(
        self, process: IqoalaProcess, routine_name: str
    ) -> Generator[EventExpression, None, None]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request
        assert request.typ == EprType.MEASURE_DIRECTLY
        num_pairs = request.num_pairs

        outcomes: List[int] = []

        if routine.callback_type == CallbackType.SEQUENTIAL:
            raise NotImplementedError
        else:
            for i in range(num_pairs):
                virt_id = self.allocate_for_pair(process, request, i)
                entdist_req = self.create_entdist_request(process, request, virt_id)
                # Create EPR pair
                yield from self.execute_entdist_request(entdist_req)
                # Measure local qubit
                m = yield from self.measure_epr_qubit(process, virt_id)
                # Free virt qubit
                self._interface.memmgr.free(process.pid, virt_id)
                outcomes.append(m)

        shared_mem = process.prog_memory.shared_memmgr
        results_addr = running_routine.result_addr
        shared_mem.write_rr_out(results_addr, outcomes)

    def handle_req_routine_ck(
        self, process: IqoalaProcess, routine_name: str
    ) -> Generator[EventExpression, None, None]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request
        num_pairs = request.num_pairs

        if routine.callback_type == CallbackType.SEQUENTIAL:
            raise NotImplementedError
        else:
            for i in range(num_pairs):
                virt_id = self.allocate_for_pair(process, request, i)
                entdist_req = self.create_entdist_request(process, request, virt_id)
                yield from self.execute_entdist_request(entdist_req)

    def instantiate_routine(
        self,
        process: IqoalaProcess,
        routine: RequestRoutine,
        args: Dict[str, Any],
        input_addr: MemAddr,
        result_addr: MemAddr,
    ) -> None:
        """Instantiates and activates routine."""
        instance = deepcopy(routine)
        instance.request.instantiate(args)

        running_routine = RunningRequestRoutine(instance, input_addr, result_addr)
        process.qnos_mem.add_running_request_routine(running_routine)

    def assign_request_routine(
        self,
        process: IqoalaProcess,
        routine_name: str,
        # TODO: remove default values?
        input_addr: MemAddr = MemAddr(0),
        result_addr: MemAddr = MemAddr(0),
    ) -> Generator[EventExpression, None, None]:
        routine = process.get_request_routine(routine_name)
        global_args = process.prog_instance.inputs.values
        self.instantiate_routine(process, routine, global_args, input_addr, result_addr)

        if routine.request.typ == EprType.CREATE_KEEP:
            yield from self.handle_req_routine_ck(process, routine_name)
        elif routine.request.typ == EprType.MEASURE_DIRECTLY:
            yield from self.handle_req_routine_md(process, routine_name)
        else:
            raise NotImplementedError
