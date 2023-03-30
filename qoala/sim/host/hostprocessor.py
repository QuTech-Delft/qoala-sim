from __future__ import annotations

import logging
from typing import Generator, List

from netqasm.lang.operand import Template

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.lang.request import CallbackType
from qoala.runtime.message import LrCallTuple, Message, RrCallTuple
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.host.hostinterface import HostInterface, HostLatencies
from qoala.sim.process import IqoalaProcess
from qoala.util.logging import LogManager


class HostProcessor:
    """Does not have state itself. Acts on and changes process objects."""

    def __init__(
        self,
        interface: HostInterface,
        latencies: HostLatencies,
        asynchronous: bool = False,
    ) -> None:
        self._interface = interface
        self._latencies = latencies
        self._asynchronous = asynchronous

        # TODO: name
        self._name = f"{interface.name}_HostProcessor"
        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({self._name})"
        )

    def initialize(self, process: IqoalaProcess) -> None:
        host_mem = process.prog_memory.host_mem
        inputs = process.prog_instance.inputs
        for name, value in inputs.values.items():
            host_mem.write(name, value)

    def assign_instr_index(
        self, process: IqoalaProcess, instr_idx: int
    ) -> Generator[EventExpression, None, None]:
        program = process.prog_instance.program
        instr = program.instructions[instr_idx]
        yield from self.assign_instr(process, instr)

    def assign_block(
        self, process: IqoalaProcess, block_name: str
    ) -> Generator[EventExpression, None, None]:
        block = process.program.get_block(block_name)
        for instr in block.instructions:
            yield from self.assign_instr(process, instr)

    def assign_instr(
        self, process: IqoalaProcess, instr: hostlang.ClassicalIqoalaOp
    ) -> Generator[EventExpression, None, None]:
        csockets = process.csockets
        host_mem = process.prog_memory.host_mem

        # Instruction duration is simulated for each instruction by adding a "wait".
        # Duration of wait is "host_instr_time".
        # Half of it is applied *before* any operations.
        # The other half is applied *after* all 'reads' from shared memory,
        # and *before* any 'writes' to shared memory.
        # See #29 for rationale.

        # Apply half of the instruction duration.
        instr_time = self._latencies.host_instr_time
        first_half = instr_time / 2
        second_half = instr_time - first_half  # just to make it adds up

        self._logger.info(f"Interpreting LHR instruction {instr}")
        if isinstance(instr, hostlang.AssignCValueOp):
            yield from self._interface.wait(first_half)
            value = instr.attributes[0]
            assert isinstance(value, int)
            loc = instr.results[0]  # type: ignore
            self._logger.info(f"writing {value} to {loc}")
            yield from self._interface.wait(second_half)
            host_mem.write(loc, value)
        elif isinstance(instr, hostlang.SendCMsgOp):
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.arguments[1], str)

            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            value = host_mem.read(instr.arguments[1])
            self._logger.info(f"sending msg {value}")
            csck.send_int(value)
            # Simulate instruction duration.
            yield from self._interface.wait(first_half)
            yield from self._interface.wait(second_half)
        elif isinstance(instr, hostlang.ReceiveCMsgOp):
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.results, list)
            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            msg = yield from csck.recv_int()

            # Just like local instructions, first apply half of the latency (before
            # doing anything), and the other half after writing to shared memory.
            # The actual receiving is done as soon as possible, however (see above).
            # peer_latency = self._latencies.host_peer_latency
            # peer_first_half = peer_latency / 2
            # peer_second_half = peer_latency - peer_first_half  # just to make it adds up
            # yield from self._interface.wait(peer_first_half)
            # yield from self._interface.wait(peer_second_half)
            yield from self._interface.wait(self._latencies.host_peer_latency)
            host_mem.write(instr.results[0], msg)
            self._logger.info(f"received msg {msg}")
        elif isinstance(instr, hostlang.AddCValueOp):
            yield from self._interface.wait(first_half)
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.arguments[1], str)
            arg0 = host_mem.read(instr.arguments[0])
            arg1 = host_mem.read(instr.arguments[1])
            loc = instr.results[0]  # type: ignore
            result = arg0 + arg1
            self._logger.info(f"computing {loc} = {arg0} + {arg1} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.MultiplyConstantCValueOp):
            yield from self._interface.wait(first_half)
            assert isinstance(instr.arguments[0], str)
            arg0 = host_mem.read(instr.arguments[0])
            const = instr.attributes[0]
            assert isinstance(const, int)
            loc = instr.results[0]  # type: ignore
            result = arg0 * const
            self._logger.info(f"computing {loc} = {arg0} * {const} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.BitConditionalMultiplyConstantCValueOp):
            yield from self._interface.wait(first_half)
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.arguments[1], str)
            arg0 = host_mem.read(instr.arguments[0])
            cond = host_mem.read(instr.arguments[1])
            const = instr.attributes[0]
            assert isinstance(const, int)
            loc = instr.results[0]  # type: ignore
            if cond == 1:
                result = arg0 * const
            else:
                result = arg0
            self._logger.info(f"computing {loc} = {arg0} * {const}^{cond} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.RunSubroutineOp):
            lrcall = self.prepare_lr_call(process, instr)

            # Send a message to Qnos asking to run the routine.
            self._interface.send_qnos_msg(Message(lrcall))

            # Wait until Qnos says it has finished.
            yield from self._interface.receive_qnos_msg()

            self.post_lr_call(process, instr, lrcall)
        elif isinstance(instr, hostlang.RunRequestOp):
            yield from self._interface.wait(first_half)
            rrcall = self.prepare_rr_call(process, instr)

            # Send a message to the Netstack asking to run the routine.
            self._interface.send_netstack_msg(Message(rrcall))

            # Wait until the Netstack says it has finished.
            yield from self._interface.receive_netstack_msg()

            # TODO: read results

        elif isinstance(instr, hostlang.ReturnResultOp):
            assert isinstance(instr.arguments[0], str)
            loc = instr.arguments[0]
            value = host_mem.read(loc)
            self._logger.info(f"returning {loc} = {value}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            process.result.values[loc] = value

    def prepare_lr_call(
        self, process: IqoalaProcess, instr: hostlang.RunSubroutineOp
    ) -> LrCallTuple:
        host_mem = process.prog_memory.host_mem

        assert isinstance(instr.arguments[0], hostlang.IqoalaVector)
        arg_vec: hostlang.IqoalaVector = instr.arguments[0]
        args = arg_vec.values
        subrt_name = instr.attributes[0]
        assert isinstance(subrt_name, str)

        routine = process.get_local_routine(subrt_name)
        self._logger.info(f"executing subroutine {routine}")

        arg_values = {arg: host_mem.read(arg) for arg in args}

        # self._logger.info(f"instantiating subroutine with values {arg_values}")
        # process.instantiate_routine(subrt_name, arg_values)

        shared_mem = process.prog_memory.shared_memmgr

        # Allocate input memory and write args to it.
        input_addr = shared_mem.allocate_lr_in(len(arg_values))
        shared_mem.write_lr_in(input_addr, list(arg_values.values()))

        # Allocate result memory.
        result_addr = shared_mem.allocate_lr_out(len(routine.return_vars))

        return LrCallTuple(subrt_name, input_addr, result_addr)

    def post_lr_call(
        self,
        process: IqoalaProcess,
        instr: hostlang.RunSubroutineOp,
        lrcall: LrCallTuple,
    ) -> None:
        shared_mem = process.prog_memory.shared_memmgr

        # Collect the host variables that should obtain the LR results.
        result_vars: List[str]
        if isinstance(instr.results, list):
            result_vars = instr.results
        elif isinstance(instr.results, hostlang.IqoalaVector):
            result_vec: hostlang.IqoalaVector = instr.results
            result_vars = result_vec.values
        else:
            raise RuntimeError

        # Read the results from shared memory.
        result = shared_mem.read_lr_out(lrcall.result_addr, len(result_vars))

        # Copy results to local host variables.
        assert len(result) == len(result_vars)
        for value, var in zip(result, result_vars):
            process.host_mem.write(var, value)

    def prepare_rr_call(
        self, process: IqoalaProcess, instr: hostlang.RunRequestOp
    ) -> RrCallTuple:
        host_mem = process.prog_memory.host_mem

        assert isinstance(instr.arguments[0], hostlang.IqoalaVector)
        arg_vec: hostlang.IqoalaVector = instr.arguments[0]
        args = arg_vec.values
        routine_name = instr.attributes[0]
        assert isinstance(routine_name, str)

        routine = process.get_request_routine(routine_name)
        self._logger.info(f"executing request routine {routine}")

        arg_values = {arg: host_mem.read(arg) for arg in args}

        shared_mem = process.prog_memory.shared_memmgr

        # Allocate input memory for RR itself and write args to it.
        input_addr = shared_mem.allocate_rr_in(len(arg_values))
        shared_mem.write_rr_in(input_addr, list(arg_values.values()))

        cb_input_addrs: List[MemAddr] = []
        cb_output_addrs: List[MemAddr] = []

        # TODO: refactor!!
        # the `num_pairs` entry of an RR may be a template.
        # Its value should be provided in the ProgramInput of this ProgamInstance.
        # This value is filled in when `instantiating` the RR, but currently this only
        # happens by the NetstackProcessor when it's assigned to execute the RR.
        # The filled-in value is then part of a `RunningRequestRoutine`. However, it is
        # not accessible by this code here.
        # For now we use the following 'hack' where we peek in the ProgramInputs:
        if isinstance(routine.request.num_pairs, Template):
            template_name = routine.request.num_pairs.name
            num_pairs = process.prog_instance.inputs.values[template_name]
        else:
            num_pairs = routine.request.num_pairs

        # Allocate memory for callbacks.
        if routine.callback is not None:
            if routine.callback_type == CallbackType.SEQUENTIAL:
                cb_routine = process.get_local_routine(routine.callback)
                for _ in range(num_pairs):
                    # Allocate input memory.
                    cb_args = cb_routine.subroutine.arguments
                    # TODO: can it just be LR_in instead of CR_in?
                    cb_input_addrs.append(shared_mem.allocate_cr_in(len(cb_args)))

                    # Allocate result memory.
                    cb_results = cb_routine.return_vars
                    cb_output_addrs.append(shared_mem.allocate_lr_out(len(cb_results)))
            elif routine.callback_type == CallbackType.WAIT_ALL:
                cb_routine = process.get_local_routine(routine.callback)
                # Allocate input memory.
                cb_args = cb_routine.subroutine.arguments
                # TODO: can it just be LR_in instead of CR_in?
                cb_input_addrs.append(shared_mem.allocate_cr_in(len(cb_args)))

                # Allocate result memory.
                cb_results = cb_routine.return_vars
                cb_output_addrs.append(shared_mem.allocate_lr_out(len(cb_results)))

        # Allocate result memory for RR itself.
        result_addr = shared_mem.allocate_rr_out(len(routine.return_vars))

        return RrCallTuple(
            routine_name, input_addr, result_addr, cb_input_addrs, cb_output_addrs
        )

    def post_rr_call(
        self, process: IqoalaProcess, instr: hostlang.RunRequestOp, rrcall: RrCallTuple
    ) -> None:
        shared_mem = process.prog_memory.shared_memmgr
        routine = process.get_request_routine(rrcall.routine_name)

        # Read the RR results from shared memory.
        rr_result = shared_mem.read_rr_out(rrcall.result_addr, len(routine.return_vars))

        # Bit of a hack; see prepare_rr_call comments.
        if isinstance(routine.request.num_pairs, Template):
            template_name = routine.request.num_pairs.name
            num_pairs = process.prog_instance.inputs.values[template_name]
        else:
            num_pairs = routine.request.num_pairs

        # Read the callback results from shared memory.
        cb_results: List[int] = []
        if routine.callback is not None:
            if routine.callback_type == CallbackType.SEQUENTIAL:
                cb_routine = process.get_local_routine(routine.callback)
                for i in range(num_pairs):
                    # Read result memory.
                    cb_ret_vars = cb_routine.return_vars
                    cb_output_addr = rrcall.cb_output_addrs[i]
                    result = shared_mem.read_lr_out(cb_output_addr, len(cb_ret_vars))
                    cb_results.extend(result)
            elif routine.callback_type == CallbackType.WAIT_ALL:
                cb_routine = process.get_local_routine(routine.callback)
                # Read result memory.
                cb_ret_vars = cb_routine.return_vars
                cb_output_addr = rrcall.cb_output_addrs[0]
                result = shared_mem.read_lr_out(cb_output_addr, len(cb_ret_vars))
                cb_results.extend(result)

        all_results = rr_result + cb_results
        # At this point, `all_results` contains all the results of both the RR itself
        # as well as from all the callbacks, in order.

        # Collect the host variables to which to copy these results.
        result_vars: List[str]
        if isinstance(instr.results, list):
            result_vars = instr.results
        elif isinstance(instr.results, hostlang.IqoalaVector):
            result_vec: hostlang.IqoalaVector = instr.results
            result_vars = result_vec.values
        else:
            raise RuntimeError

        # Copy all results to local host variables.
        assert len(all_results) == len(result_vars)
        for value, var in zip(all_results, result_vars):
            process.host_mem.write(var, value)
