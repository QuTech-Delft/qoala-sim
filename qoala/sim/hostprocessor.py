from __future__ import annotations

import logging
from copy import deepcopy
from typing import Generator

from netqasm.lang.operand import Register
from netqasm.lang.parsing.text import NetQASMSyntaxError, parse_register

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.sim.hostinterface import HostInterface, HostLatencies
from qoala.sim.logging import LogManager
from qoala.sim.message import Message
from qoala.sim.process import IqoalaProcess, RoutineInstance


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
        pid = process.prog_instance.pid
        program = process.prog_instance.program

        instr = program.instructions[instr_idx]

        self._logger.info(f"Interpreting LHR instruction {instr}")
        if isinstance(instr, hostlang.AssignCValueOp):
            value = instr.attributes[0]
            loc = instr.results[0]
            self._logger.info(f"writing {value} to {loc}")
            host_mem.write(loc, value)
        elif isinstance(instr, hostlang.SendCMsgOp):
            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            value = host_mem.read(instr.arguments[1])
            self._logger.info(f"sending msg {value}")
            csck.send_int(value)
        elif isinstance(instr, hostlang.ReceiveCMsgOp):
            csck_id = host_mem.read(instr.arguments[0])
            csck = csockets[csck_id]
            msg = yield from csck.recv_int()
            yield from self._interface.wait(self._latencies.host_peer_latency)
            host_mem.write(instr.results[0], msg)
            self._logger.info(f"received msg {msg}")
        elif isinstance(instr, hostlang.AddCValueOp):
            arg0 = host_mem.read(instr.arguments[0])
            arg1 = host_mem.read(instr.arguments[1])
            loc = instr.results[0]
            result = arg0 + arg1
            self._logger.info(f"computing {loc} = {arg0} + {arg1} = {result}")
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.MultiplyConstantCValueOp):
            arg0 = host_mem.read(instr.arguments[0])
            const = instr.attributes[0]
            assert isinstance(const, int)
            loc = instr.results[0]
            result = arg0 * const
            self._logger.info(f"computing {loc} = {arg0} * {const} = {result}")
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.BitConditionalMultiplyConstantCValueOp):
            arg0 = host_mem.read(instr.arguments[0])
            cond = host_mem.read(instr.arguments[1])
            const = instr.attributes[0]
            assert isinstance(const, int)
            loc = instr.results[0]
            if cond == 1:
                result = arg0 * const
            else:
                result = arg0
            self._logger.info(f"computing {loc} = {arg0} * {const}^{cond} = {result}")
            host_mem.write(loc, result)
        elif isinstance(instr, hostlang.RunSubroutineOp):
            arg_vec: hostlang.IqoalaVector = instr.arguments[0]
            args = arg_vec.values
            subrt_name = instr.attributes[0]
            assert isinstance(subrt_name, str)

            routine = process.get_local_routine(subrt_name)
            self._logger.info(f"executing subroutine {routine}")

            arg_values = {arg: host_mem.read(arg) for arg in args}

            self._logger.info(f"instantiating subroutine with values {arg_values}")
            process.instantiate_routine(subrt_name, pid, arg_values)

            if self._asynchronous:
                # Send a message to Qnos asking it to execute the subroutine.
                self._interface.send_qnos_msg(Message(subrt_name))
                # Wait for Qnos to finish.
                yield from self._interface.receive_qnos_msg()
                # Qnos should have updated the shared memory with subroutine results.
                self.copy_subroutine_results(process, subrt_name)
            else:
                # Let the scheduler make sure that Qnos executes the subroutine at
                # some point. The scheduler is also responsible for copying subroutine
                # results to the Host memory.
                pass

        elif isinstance(instr, hostlang.ReturnResultOp):
            loc = instr.arguments[0]
            value = host_mem.read(loc)
            self._logger.info(f"returning {loc} = {value}")
            process.result.values[loc] = value

        yield from self._interface.wait(self._latencies.host_instr_time)

    def copy_subroutine_results(self, process: IqoalaProcess, subrt_name: str) -> None:
        routine = process.get_local_routine(subrt_name)

        for key, mem_loc in routine.return_map.items():
            try:
                reg: Register = parse_register(mem_loc.loc)
                value = process.shared_mem.get_register(reg)
                self._logger.debug(
                    f"writing shared memory value {value} from location "
                    f"{mem_loc} to variable {key}"
                )
                # print(f"subrt result {key} = {value}")
                process.host_mem.write(key, value)
            except NetQASMSyntaxError:
                pass  # TODO: needed?
