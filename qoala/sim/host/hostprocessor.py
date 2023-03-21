from __future__ import annotations

import logging
from typing import Generator

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.runtime.message import LrCallTuple, Message, RrCallTuple
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

    def assign(
        self, process: IqoalaProcess, instr_idx: int
    ) -> Generator[EventExpression, None, None]:
        csockets = process.csockets
        host_mem = process.prog_memory.host_mem
        program = process.prog_instance.program

        instr = program.instructions[instr_idx]

        self._logger.info(f"Interpreting LHR instruction {instr}")
        if isinstance(instr, hostlang.AssignCValueOp):
            value = instr.attributes[0]
            assert isinstance(value, int)
            loc = instr.results[0]  # type: ignore
            self._logger.info(f"writing {value} to {loc}")
            host_mem.write(loc, value)
        elif isinstance(instr, hostlang.SendCMsgOp):
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.arguments[1], str)

            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            value = host_mem.read(instr.arguments[1])
            self._logger.info(f"sending msg {value}")
            csck.send_int(value)
        elif isinstance(instr, hostlang.ReceiveCMsgOp):
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.results, list)
            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            msg = yield from csck.recv_int()
            yield from self._interface.wait(self._latencies.host_peer_latency)
            host_mem.write(instr.results[0], msg)
            self._logger.info(f"received msg {msg}")
        elif isinstance(instr, hostlang.AddCValueOp):
            assert isinstance(instr.arguments[0], str)
            assert isinstance(instr.arguments[1], str)
            arg0 = host_mem.read(instr.arguments[0])
            arg1 = host_mem.read(instr.arguments[1])
            loc = instr.results[0]  # type: ignore
            result = arg0 + arg1
            self._logger.info(f"computing {loc} = {arg0} + {arg1} = {result}")
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.MultiplyConstantCValueOp):
            assert isinstance(instr.arguments[0], str)
            arg0 = host_mem.read(instr.arguments[0])
            const = instr.attributes[0]
            assert isinstance(const, int)
            loc = instr.results[0]  # type: ignore
            result = arg0 * const
            self._logger.info(f"computing {loc} = {arg0} * {const} = {result}")
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.BitConditionalMultiplyConstantCValueOp):
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
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.RunSubroutineOp):
            lrcall = self.prepare_lr_call(process, instr)

            # Send a message to Qnos asking to run the routine.
            self._interface.send_qnos_msg(Message(lrcall))

            # Wait until Qnos says it has finished.
            yield from self._interface.receive_qnos_msg()

            self.post_lr_call(process, lrcall)
        elif isinstance(instr, hostlang.RunRequestOp):
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
            process.result.values[loc] = value

        yield from self._interface.wait(self._latencies.host_instr_time)

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
        result_addr = shared_mem.allocate_lr_out(len(routine.return_map))

        return LrCallTuple(subrt_name, input_addr, result_addr)

    def post_lr_call(self, process: IqoalaProcess, lrcall: LrCallTuple) -> None:
        shared_mem = process.prog_memory.shared_memmgr
        routine = process.get_local_routine(lrcall.routine_name)

        # Read the results from shared memory.
        result = shared_mem.read_lr_out(lrcall.result_addr, len(routine.return_map))

        # Copy results to local host variables.
        assert len(result) == len(routine.return_map)
        for value, var in zip(result, routine.return_map.keys()):
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

        # Allocate input memory and write args to it.
        input_addr = shared_mem.allocate_rr_in(len(arg_values))
        shared_mem.write_rr_in(input_addr, list(arg_values.values()))

        # Allocate result memory.
        # TODO: implement RR return values.
        result_addr = shared_mem.allocate_rr_out(0)

        return RrCallTuple(routine_name, input_addr, result_addr)
