from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional, Union

from netqasm.lang.operand import Template

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.lang.hostlang import ClassicalIqoalaOp, IqoalaSingleton, IqoalaVectorElement
from qoala.lang.request import CallbackType
from qoala.runtime.memory import HostMemory
from qoala.runtime.message import LrCallTuple, RrCallTuple
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.host.hostinterface import HostInterface, HostLatencies
from qoala.sim.process import QoalaProcess
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

    def initialize(self, process: QoalaProcess) -> None:
        host_mem = process.prog_memory.host_mem
        inputs = process.prog_instance.inputs
        for name, value in inputs.values.items():
            host_mem.write(name, value)

    @staticmethod
    def _read_value_from_host_mem(
        arg: Union[IqoalaSingleton, IqoalaVectorElement], host_mem: HostMemory
    ) -> int:
        if isinstance(arg, hostlang.IqoalaSingleton):
            return host_mem.read(arg.name)
        else:
            loc = arg.name
            index = arg.index
            return host_mem.read_vec(loc)[index]

    def assign_instr_index(
        self, process: QoalaProcess, instr_idx: int
    ) -> Generator[EventExpression, None, None]:
        program = process.prog_instance.program
        instr = program.instructions[instr_idx]
        yield from self.assign_instr(process, instr)

    def assign_block(
        self, process: QoalaProcess, block_name: str
    ) -> Generator[EventExpression, None, None]:
        block = process.program.get_block(block_name)

        for instr in block.instructions:
            yield from self.assign_instr(process, instr)
            if self._interface.program_instance_jumps[process.pid] != -1:
                break

    def assign_instr(
        self, process: QoalaProcess, instr: hostlang.ClassicalIqoalaOp
    ) -> Generator[EventExpression, None, None]:
        csockets = process.csockets
        host_mem = process.prog_memory.host_mem
        pid = process.pid

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
        self._interface.program_instance_jumps[pid] = -1
        self._logger.debug(f"Interpreting LHR instruction {instr}")
        if isinstance(instr, hostlang.AssignCValueOp):
            yield from self._interface.wait(first_half)
            value = instr.attributes[0]
            assert isinstance(value, int)
            assert isinstance(instr.results, hostlang.IqoalaSingleton)
            loc = instr.results.name  # type: ignore
            self._logger.debug(f"writing {value} to {loc}")
            yield from self._interface.wait(second_half)
            host_mem.write(loc, value)
        elif isinstance(instr, hostlang.BusyOp):
            value = instr.attributes[0]
            if not isinstance(value, int):
                value = host_mem.read(value)
            self._logger.debug(f"busy for {value} ns")
            yield from self._interface.wait(value)
        elif isinstance(instr, hostlang.SendCMsgOp):
            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVectorElement)
            assert isinstance(
                instr.arguments[1], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[1], hostlang.IqoalaVectorElement)

            csck_id = self._read_value_from_host_mem(instr.arguments[0], host_mem)

            csck = csockets[csck_id]
            if isinstance(instr.arguments[1], hostlang.IqoalaSingleton):
                value = host_mem.read(instr.arguments[1].name)
            else:
                loc = instr.arguments[1].name
                index = instr.arguments[1].index
                value = host_mem.read_vec(loc)[index]
            self._logger.info(f"sending msg {value}")
            csck.send_int(value)
            # Simulate instruction duration.
            yield from self._interface.wait(self._latencies.host_instr_time)
        elif isinstance(instr, hostlang.ReceiveCMsgOp):
            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVectorElement)
            assert isinstance(instr.results, hostlang.IqoalaSingleton)

            csck_id = self._read_value_from_host_mem(instr.arguments[0], host_mem)

            csck = csockets[csck_id]

            # TODO: refactor
            tup = (csck.remote_pid, pid)
            if tup not in self._interface.get_available_messages(csck.remote_name):
                msg = yield from csck.recv_int()
            else:
                msg = csck.read_int()

            yield from self._interface.wait(self._latencies.host_peer_latency)
            host_mem.write(instr.results.name, msg)
            self._logger.info(f"received msg {msg}")
        elif isinstance(instr, hostlang.AddCValueOp):
            yield from self._interface.wait(first_half)
            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVectorElement)
            assert isinstance(
                instr.arguments[1], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[1], hostlang.IqoalaVectorElement)

            arg0 = self._read_value_from_host_mem(instr.arguments[0], host_mem)
            arg1 = self._read_value_from_host_mem(instr.arguments[1], host_mem)

            assert isinstance(instr.results, hostlang.IqoalaSingleton)
            loc = instr.results.name  # type: ignore
            result = arg0 + arg1
            self._logger.debug(f"computing {loc} = {arg0} + {arg1} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.MultiplyConstantCValueOp):
            yield from self._interface.wait(first_half)
            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVectorElement)

            arg0 = self._read_value_from_host_mem(instr.arguments[0], host_mem)

            const = instr.attributes[0]
            assert isinstance(const, int)
            assert isinstance(instr.results, hostlang.IqoalaSingleton)
            loc = instr.results.name  # type: ignore
            result = arg0 * const
            self._logger.debug(f"computing {loc} = {arg0} * {const} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.BitConditionalMultiplyConstantCValueOp):
            yield from self._interface.wait(first_half)

            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVectorElement)
            assert isinstance(
                instr.arguments[1], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[1], hostlang.IqoalaVectorElement)

            arg0 = self._read_value_from_host_mem(instr.arguments[0], host_mem)
            cond = self._read_value_from_host_mem(instr.arguments[1], host_mem)

            const = instr.attributes[0]
            assert isinstance(const, int)
            assert isinstance(instr.results, hostlang.IqoalaSingleton)
            loc = instr.results.name  # type: ignore
            if cond == 1:
                result = arg0 * const
            else:
                result = arg0
            self._logger.debug(f"computing {loc} = {arg0} * {const}^{cond} = {result}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.ReturnResultOp):
            yield from self._interface.wait(first_half)
            assert isinstance(
                instr.arguments[0], hostlang.IqoalaSingleton
            ) or isinstance(instr.arguments[0], hostlang.IqoalaVector)
            loc = instr.arguments[0].name
            if isinstance(instr.arguments[0], hostlang.IqoalaSingleton):
                value = host_mem.read(loc)
            else:
                value = host_mem.read_vec(loc)
            self._logger.debug(f"returning {loc} = {value}")
            # Simulate instruction duration.
            yield from self._interface.wait(second_half)
            process.result.values[loc] = value
        elif isinstance(instr, hostlang.JumpOp):
            yield from self._interface.wait(first_half)
            assert isinstance(instr.attributes[0], str)
            block_name = instr.attributes[0]
            self._logger.debug(f"jumping to block {block_name}")
            block_id = process.prog_instance.program.get_block_id(block_name)
            self._interface.program_instance_jumps[pid] = block_id
            yield from self._interface.wait(second_half)
        elif isinstance(instr, hostlang.BranchIfEqualOp):
            yield from self._branch(process, instr, "__eq__")
        elif isinstance(instr, hostlang.BranchIfNotEqualOp):
            yield from self._branch(process, instr, "__ne__")
        elif isinstance(instr, hostlang.BranchIfLessThanOp):
            yield from self._branch(process, instr, "__lt__")
        elif isinstance(instr, hostlang.BranchIfGreaterThanOp):
            yield from self._branch(process, instr, "__gt__")

    def _branch(
        self, process: QoalaProcess, instr: ClassicalIqoalaOp, comparison_op: str
    ) -> Generator[EventExpression, None, None]:
        instr_time = self._latencies.host_instr_time
        first_half = instr_time / 2
        second_half = instr_time - first_half  # just to make it adds up
        yield from self._interface.wait(first_half)

        assert isinstance(instr.arguments[0], hostlang.IqoalaSingleton) or isinstance(
            instr.arguments[0], hostlang.IqoalaVectorElement
        )
        assert isinstance(instr.arguments[1], hostlang.IqoalaSingleton) or isinstance(
            instr.arguments[1], hostlang.IqoalaVectorElement
        )

        host_mem = process.prog_memory.host_mem
        pid = process.pid

        value0 = self._read_value_from_host_mem(instr.arguments[0], host_mem)
        value1 = self._read_value_from_host_mem(instr.arguments[1], host_mem)

        assert isinstance(instr.attributes[0], str)
        block_name = instr.attributes[0]
        if getattr(value0, comparison_op)(value1):
            block_id = process.prog_instance.program.get_block_id(block_name)
            self._interface.program_instance_jumps[pid] = block_id
        yield from self._interface.wait(second_half)

    def prepare_lr_call(
        self, process: QoalaProcess, instr: hostlang.RunSubroutineOp
    ) -> LrCallTuple:
        host_mem = process.prog_memory.host_mem

        assert isinstance(instr.arguments[0], hostlang.IqoalaTuple)
        arg_vec: hostlang.IqoalaTuple = instr.arguments[0]
        args = arg_vec.values
        subrt_name = instr.attributes[0]
        assert isinstance(subrt_name, str)

        routine = process.get_local_routine(subrt_name)
        self._logger.info(f"executing subroutine {routine.name}")
        self._logger.debug(f"subroutine contents of {routine.name}: \n{routine}")

        arg_values = {arg: host_mem.read(arg) for arg in args}

        # self._logger.info(f"instantiating subroutine with values {arg_values}")
        # process.instantiate_routine(subrt_name, arg_values)

        shared_mem = process.prog_memory.shared_mem

        # Allocate input memory and write args to it.
        input_addr = shared_mem.allocate_lr_in(len(arg_values))
        shared_mem.write_lr_in(input_addr, list(arg_values.values()))

        # Check if the expected number of results match the number of result variables.
        result_vars_len = self._calculate_result_size(instr.results)
        assert result_vars_len == routine.get_return_size()

        # Allocate result memory.
        result_addr = shared_mem.allocate_lr_out(result_vars_len)

        return LrCallTuple(subrt_name, input_addr, result_addr)

    def post_lr_call(
        self,
        process: QoalaProcess,
        instr: hostlang.RunSubroutineOp,
        lrcall: LrCallTuple,
    ) -> None:
        shared_mem = process.prog_memory.shared_mem

        # Read the results from shared memory.
        result_vars_len = self._calculate_result_size(instr.results)
        result = shared_mem.read_lr_out(lrcall.result_addr, result_vars_len)
        assert len(result) == result_vars_len

        # Copy results to local host variables.
        if isinstance(instr.results, hostlang.IqoalaSingleton):
            for value, var in zip(result, instr.results.name):
                process.host_mem.write(var, value)
        elif isinstance(instr.results, hostlang.IqoalaTuple):
            result_tup: hostlang.IqoalaTuple = instr.results
            for value, var in zip(result, result_tup.values):
                process.host_mem.write(var, value)
        elif isinstance(instr.results, hostlang.IqoalaVector):
            result_vec: hostlang.IqoalaVector = instr.results
            process.host_mem.write_vec(result_vec.name, result)
        elif instr.results is None:
            pass
        else:
            raise RuntimeError

    def prepare_rr_call(
        self, process: QoalaProcess, instr: hostlang.RunRequestOp
    ) -> RrCallTuple:
        host_mem = process.prog_memory.host_mem

        assert isinstance(instr.arguments[0], hostlang.IqoalaTuple)
        arg_vec: hostlang.IqoalaTuple = instr.arguments[0]
        args = arg_vec.values
        routine_name = instr.attributes[0]
        assert isinstance(routine_name, str)

        routine = process.get_request_routine(routine_name)
        self._logger.info(f"executing request routine {routine}")

        arg_values = {arg: host_mem.read(arg) for arg in args}

        shared_mem = process.prog_memory.shared_mem

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
        prog_input = process.prog_instance.inputs.values
        if isinstance(routine.request.num_pairs, Template):
            template_name = routine.request.num_pairs.name
            num_pairs = prog_input[template_name]
        else:
            num_pairs = routine.request.num_pairs

        # Calculate total return size (RR itself + callbacks).
        total_return_size = 0

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
                    cb_results_len = cb_routine.get_return_size()
                    cb_output_addrs.append(shared_mem.allocate_lr_out(cb_results_len))
                    total_return_size += cb_results_len
            elif routine.callback_type == CallbackType.WAIT_ALL:
                cb_routine = process.get_local_routine(routine.callback)
                # Allocate input memory.
                cb_args = cb_routine.subroutine.arguments
                # TODO: can it just be LR_in instead of CR_in?
                cb_input_addrs.append(shared_mem.allocate_cr_in(len(cb_args)))

                # Allocate result memory.
                cb_results_len = cb_routine.get_return_size()
                cb_output_addrs.append(shared_mem.allocate_lr_out(cb_results_len))
                total_return_size += cb_results_len

        # Allocate result memory for RR itself.
        routine_return_size = routine.get_return_size(prog_input)
        result_addr = shared_mem.allocate_rr_out(routine_return_size)
        total_return_size += routine_return_size

        # Check that host variables match the RR total result in length
        result_vars_len = self._calculate_result_size(instr.results, prog_input)
        assert result_vars_len == total_return_size

        return RrCallTuple(
            routine_name, input_addr, result_addr, cb_input_addrs, cb_output_addrs
        )

    def post_rr_call(
        self, process: QoalaProcess, instr: hostlang.RunRequestOp, rrcall: RrCallTuple
    ) -> None:
        shared_mem = process.prog_memory.shared_mem
        routine = process.get_request_routine(rrcall.routine_name)

        # Bit of a hack; see prepare_rr_call comments.
        prog_input = process.prog_instance.inputs.values
        if isinstance(routine.request.num_pairs, Template):
            template_name = routine.request.num_pairs.name
            num_pairs = prog_input[template_name]
        else:
            num_pairs = routine.request.num_pairs

        # Read the RR results from shared memory.
        rr_result = shared_mem.read_rr_out(
            rrcall.result_addr, routine.get_return_size(prog_input)
        )

        # Read the callback results from shared memory.
        cb_results: List[int] = []
        if routine.callback is not None:
            if routine.callback_type == CallbackType.SEQUENTIAL:
                cb_routine = process.get_local_routine(routine.callback)
                for i in range(num_pairs):
                    # Read result memory.
                    cb_results_len = cb_routine.get_return_size()
                    cb_output_addr = rrcall.cb_output_addrs[i]
                    result = shared_mem.read_lr_out(cb_output_addr, cb_results_len)
                    cb_results.extend(result)
            elif routine.callback_type == CallbackType.WAIT_ALL:
                cb_routine = process.get_local_routine(routine.callback)
                # Read result memory.
                cb_results_len = cb_routine.get_return_size()
                cb_output_addr = rrcall.cb_output_addrs[0]
                result = shared_mem.read_lr_out(cb_output_addr, cb_results_len)
                cb_results.extend(result)

        all_results = rr_result + cb_results
        # At this point, `all_results` contains all the results of both the RR itself
        # as well as from all the callbacks, in order.
        result_vars_len = self._calculate_result_size(instr.results, prog_input)
        assert len(all_results) == result_vars_len

        # Collect the host variables to which to copy these results.
        if isinstance(instr.results, hostlang.IqoalaSingleton):
            loc = instr.results.name
            value = all_results[0]
            process.host_mem.write(loc, value)
        elif isinstance(instr.results, hostlang.IqoalaTuple):
            result_tup: hostlang.IqoalaTuple = instr.results
            for value, var in zip(all_results, result_tup.values):
                process.host_mem.write(var, value)
        elif isinstance(instr.results, hostlang.IqoalaVector):
            result_vec: hostlang.IqoalaVector = instr.results
            process.host_mem.write_vec(result_vec.name, all_results)
        elif instr.results is None:
            pass
        else:
            raise RuntimeError

    def _calculate_result_size(
        self,
        results: hostlang.IqoalaVar,
        prog_input: Optional[Dict[str, int]] = None,
    ) -> int:
        if results is None:
            return 0
        elif isinstance(results, hostlang.IqoalaTuple):
            return len(results.values)
        elif isinstance(results, hostlang.IqoalaVector):
            if isinstance(results.size, int):
                return results.size
            else:  # Size is a variable. Get its value from the Program Inputs.
                assert prog_input is not None
                return prog_input[results.size]
        elif isinstance(results, hostlang.IqoalaSingleton):
            return 1
        else:
            raise RuntimeError
