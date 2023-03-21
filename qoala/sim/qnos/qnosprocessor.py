from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Generator, Optional, Union

import netsquid as ns
from netqasm.lang.instr import NetQASMInstruction, core, nv, vanilla
from netqasm.lang.operand import Register
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_CZ,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.instructions import Instruction as NsInstr
from netsquid.qubits import qubitapi

from pydynaa import EventExpression
from qoala.lang.routine import LocalRoutine
from qoala.runtime.memory import ProgramMemory, RunningLocalRoutine
from qoala.runtime.message import LrCallTuple, Message
from qoala.runtime.sharedmem import MemAddr
from qoala.sim.globals import GlobalSimData
from qoala.sim.memmgr import NotAllocatedError
from qoala.sim.process import IqoalaProcess
from qoala.sim.qdevice import QDevice, QDeviceCommand
from qoala.sim.qnos.qnosinterface import QnosInterface, QnosLatencies
from qoala.sim.requests import (
    NetstackBreakpointCreateRequest,
    NetstackBreakpointReceiveRequest,
)
from qoala.util.constants import PI, PI_OVER_2
from qoala.util.logging import LogManager


class QnosProcessor:
    """Does not have state itself."""

    def __init__(
        self,
        interface: QnosInterface,
        latencies: QnosLatencies,
        asynchronous: bool = False,
    ) -> None:
        self._interface = interface
        self._latencies = latencies
        self._asynchronous = asynchronous

        # TODO: rewrite
        self._name = f"{interface.name}_QnosProcessor"

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}({self._name})"
        )

        # memory of current program, only not-None when processor is active
        self._current_prog_mem: Optional[ProgramMemory] = None

        self._current_routine: Optional[RunningLocalRoutine] = None

    def _prog_mem(self) -> ProgramMemory:
        # May only be called when processor is active
        assert self._current_prog_mem is not None
        return self._current_prog_mem

    def _routine(self) -> RunningLocalRoutine:
        # May only be called when processor is active
        assert self._current_routine is not None
        return self._current_routine

    @property
    def qdevice(self) -> QDevice:
        return self._interface.qdevice

    def instantiate_routine(
        self,
        process: IqoalaProcess,
        routine: LocalRoutine,
        args: Dict[str, Any],
        input_addr: MemAddr,
        result_addr: MemAddr,
    ) -> None:
        """Instantiates and activates routine."""
        instance = deepcopy(routine)
        instance.subroutine.instantiate(process.pid, args)

        running_routine = RunningLocalRoutine(instance, input_addr, result_addr)
        process.qnos_mem.add_running_local_routine(running_routine)

    def await_local_routine_call(
        self, process: IqoalaProcess
    ) -> Generator[EventExpression, None, None]:
        msg = yield from self._interface.receive_host_msg()
        payload: LrCallTuple = msg.content
        yield from self.assign_local_routine(
            process, payload.routine_name, payload.input_addr, payload.result_addr
        )
        # Mock sending signal back to Host that subroutine has finished.
        self._interface.send_host_msg(Message(None))

    def assign_local_routine(
        self,
        process: IqoalaProcess,
        routine_name: str,
        input_addr: MemAddr,
        result_addr: MemAddr,
    ) -> Generator[EventExpression, None, None]:
        routine = process.get_local_routine(routine_name)
        global_args = process.prog_instance.inputs.values

        self.instantiate_routine(process, routine, global_args, input_addr, result_addr)

        netqasm_instrs = routine.subroutine.instructions

        instr_idx = 0
        while instr_idx < len(netqasm_instrs):
            instr_idx = yield from self.assign_routine_instr(
                process, routine_name, instr_idx
            )

    def assign_routine_instr(
        self, process: IqoalaProcess, subrt_name: str, instr_idx: int
    ) -> Generator[EventExpression, None, int]:
        """Assign the processor to one specific instruction in a local routine."""
        running_routine = process.qnos_mem.get_running_local_routine(subrt_name)
        routine = running_routine.routine
        pid = process.prog_instance.pid

        self._current_prog_mem = process.prog_memory
        self._current_routine = running_routine

        subroutine = routine.subroutine

        # TODO: handle program counter and jumping!!
        instr = subroutine.instructions[instr_idx]
        self._logger.debug(
            f"{ns.sim_time()} interpreting instruction {instr} at line {instr_idx}"
        )

        next_instr_idx: int

        if (
            isinstance(instr, core.JmpInstruction)
            or isinstance(instr, core.BranchUnaryInstruction)
            or isinstance(instr, core.BranchBinaryInstruction)
        ):
            if new_line := (yield from self._interpret_branch_instr(pid, instr)):
                next_instr_idx = new_line
            else:
                next_instr_idx = instr_idx + 1
        else:
            generator = self._interpret_instruction(pid, instr)
            if generator:
                yield from generator
            next_instr_idx = instr_idx + 1

        self._current_prog_mem = None
        self._current_routine = None
        return next_instr_idx

    def _interpret_instruction(
        self, pid: int, instr: NetQASMInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        if isinstance(instr, core.SetInstruction):
            return self._interpret_set(pid, instr)
        elif isinstance(instr, core.QAllocInstruction):
            return self._interpret_qalloc(pid, instr)
        elif isinstance(instr, core.QFreeInstruction):
            return self._interpret_qfree(pid, instr)
        elif isinstance(instr, core.StoreInstruction):
            return self._interpret_store(pid, instr)
        elif isinstance(instr, core.LoadInstruction):
            return self._interpret_load(pid, instr)
        elif isinstance(instr, core.LeaInstruction):
            return self._interpret_lea(pid, instr)
        elif isinstance(instr, core.UndefInstruction):
            return self._interpret_undef(pid, instr)
        elif isinstance(instr, core.ArrayInstruction):
            return self._interpret_array(pid, instr)
        elif isinstance(instr, core.InitInstruction):
            return self._interpret_init(pid, instr)
        elif isinstance(instr, core.MeasInstruction):
            return self._interpret_meas(pid, instr)
        elif isinstance(instr, core.CreateEPRInstruction):
            return self._interpret_create_epr(pid, instr)
        elif isinstance(instr, core.RecvEPRInstruction):
            return self._interpret_recv_epr(pid, instr)
        elif isinstance(instr, core.WaitAllInstruction):
            return self._interpret_wait_all(pid, instr)
        elif isinstance(instr, core.RetRegInstruction):
            return None
        elif isinstance(instr, core.RetArrInstruction):
            return None
        elif isinstance(instr, core.SingleQubitInstruction):
            return self._interpret_single_qubit_instr(pid, instr)
        elif isinstance(instr, core.TwoQubitInstruction):
            return self._interpret_two_qubit_instr(pid, instr)
        elif isinstance(instr, core.RotationInstruction):
            return self._interpret_single_rotation_instr(pid, instr)
        elif isinstance(instr, core.ControlledRotationInstruction):
            return self._interpret_controlled_rotation_instr(pid, instr)
        elif isinstance(instr, core.ClassicalOpInstruction) or isinstance(
            instr, core.ClassicalOpModInstruction
        ):
            return self._interpret_binary_classical_instr(pid, instr)
        elif isinstance(instr, core.BreakpointInstruction):
            return self._interpret_breakpoint(pid, instr)
        else:
            raise RuntimeError(f"Invalid instruction {instr}")

    def _interpret_breakpoint(
        self, pid: int, instr: core.BreakpointInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        if instr.action.value == 0:
            self._logger.info("BREAKPOINT: no action taken")
        elif instr.action.value == 1:
            self._logger.info("BREAKPOINT: dumping local state:")
            for i in range(self.qdevice.qprocessor.num_positions):
                if self.qdevice.qprocessor.mem_positions[i].in_use:
                    q = self.qdevice.qprocessor.peek(i)
                    qstate = qubitapi.reduced_dm(q)
                    self._logger.info(f"physical qubit {i}:\n{qstate}")

            # TODO: fix this; GlobalSimData is not static anymore!
            state = GlobalSimData.get_quantum_state(save=True)  # type: ignore
        elif instr.action.value == 2:
            self._logger.info("BREAKPOINT: dumping global state:")
            if instr.role.value == 0:
                self._interface.send_netstack_msg(
                    Message(content=NetstackBreakpointCreateRequest(pid))
                )
                ready = yield from self._interface.receive_netstack_msg()
                assert ready.content == "breakpoint ready"

                # TODO: fix this; GlobalSimData is not static anymore!
                state = GlobalSimData.get_quantum_state(save=True)  # type: ignore
                self._logger.info(state)

                self._interface.send_netstack_msg(Message(content="breakpoint end"))
                finished = yield from self._interface.receive_netstack_msg()
                assert finished.content == "breakpoint finished"
            elif instr.role.value == 1:
                self._interface.send_netstack_msg(
                    Message(content=NetstackBreakpointReceiveRequest(pid))
                )
                ready = yield from self._interface.receive_netstack_msg()
                assert ready.content == "breakpoint ready"
                self._interface.send_netstack_msg(Message(content="breakpoint end"))
                finished = yield from self._interface.receive_netstack_msg()
                assert finished.content == "breakpoint finished"
            else:
                raise ValueError
        else:
            raise ValueError

    def _interpret_set(
        self, pid: int, instr: core.SetInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        self._logger.debug(f"Set register {instr.reg} to {instr.imm}")
        qnos_mem = self._prog_mem().qnos_mem
        qnos_mem.set_reg_value(instr.reg, instr.imm.value)
        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _interpret_qalloc(
        self, pid: int, instr: core.QAllocInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem

        virt_id = qnos_mem.get_reg_value(instr.reg)
        if virt_id is None:
            raise RuntimeError(f"qubit address in register {instr.reg} is not defined")
        self._logger.debug(f"Allocating qubit with virtual ID {virt_id}")
        self._interface.memmgr.allocate(pid, virt_id)
        yield from self._interface.wait(self._latencies.qnos_instr_time)

        return None

    def _interpret_qfree(
        self, pid: int, instr: core.QFreeInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem

        virt_id = qnos_mem.get_reg_value(instr.reg)
        assert virt_id is not None
        self._logger.debug(f"Freeing virtual qubit {virt_id}")
        self._interface.memmgr.free(pid, virt_id)
        self._interface.signal_memory_freed()
        yield from self._interface.wait(self._latencies.qnos_instr_time)

        return None

    def _interpret_store(
        self, pid: int, instr: core.StoreInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem

        new_shared_mem = self._prog_mem().shared_memmgr
        result_addr = self._routine().result_addr

        value = qnos_mem.get_reg_value(instr.reg)
        if value is None:
            raise RuntimeError(f"value in register {instr.reg} is not defined")
        self._logger.debug(
            f"Storing value {value} from register {instr.reg} "
            f"to array entry {instr.entry}"
        )

        addr = instr.entry.address.address
        entry = instr.entry.index
        assert isinstance(entry, Register)
        index = qnos_mem.get_reg_value(entry)
        if addr == 101:  # result region
            new_shared_mem.write_lr_out(result_addr, [value], offset=index)
        else:
            raise NotImplementedError  # TODO: needed?

        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _interpret_load(
        self, pid: int, instr: core.LoadInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem

        new_shared_mem = self._prog_mem().shared_memmgr
        input_addr = self._routine().params_addr

        addr = instr.entry.address.address
        entry = instr.entry.index
        assert isinstance(entry, Register)
        index = qnos_mem.get_reg_value(entry)
        if addr == 100:  # input region
            [value] = new_shared_mem.read_lr_in(input_addr, 1, offset=index)
        else:
            raise NotImplementedError  # TODO: needed?

        if value is None:
            raise RuntimeError(f"array value at {instr.entry} is not defined")
        self._logger.debug(
            f"Storing value {value} from array entry {instr.entry} "
            f"to register {instr.reg}"
        )

        qnos_mem.set_reg_value(instr.reg, value)
        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _interpret_lea(
        self, pid: int, instr: core.LeaInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem
        self._logger.debug(
            f"Storing address of {instr.address} to register {instr.reg}"
        )
        qnos_mem.set_reg_value(instr.reg, instr.address.address)
        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _interpret_undef(
        self, pid: int, instr: core.UndefInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        raise DeprecationWarning

    def _interpret_array(
        self, pid: int, instr: core.ArrayInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem
        shared_mem = self._prog_mem().shared_mem

        length = qnos_mem.get_reg_value(instr.size)
        assert length is not None
        self._logger.debug(
            f"Initializing an array of length {length} at address {instr.address}"
        )

        shared_mem.init_new_array(instr.address.address, length)
        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _interpret_branch_instr(
        self,
        pid: int,
        instr: Union[
            core.BranchUnaryInstruction,
            core.BranchBinaryInstruction,
            core.JmpInstruction,
        ],
    ) -> Generator[EventExpression, None, Optional[int]]:
        """Returns line to jump to, or None if no jump happens."""
        qnos_mem = self._prog_mem().qnos_mem
        a, b = None, None
        registers = []
        if isinstance(instr, core.BranchUnaryInstruction):
            a = qnos_mem.get_reg_value(instr.reg)
            registers = [instr.reg]
        elif isinstance(instr, core.BranchBinaryInstruction):
            a = qnos_mem.get_reg_value(instr.reg0)
            b = qnos_mem.get_reg_value(instr.reg1)
            registers = [instr.reg0, instr.reg1]

        if isinstance(instr, core.JmpInstruction):
            condition = True
        elif isinstance(instr, core.BranchUnaryInstruction):
            condition = instr.check_condition(a)
        elif isinstance(instr, core.BranchBinaryInstruction):
            condition = instr.check_condition(a, b)

        yield from self._interface.wait(self._latencies.qnos_instr_time)
        if condition:
            jump_address = instr.line
            self._logger.debug(
                f"Branching to line {jump_address}, since {instr}(a={a}, b={b}) "
                f"is True, with values from registers {registers}"
            )
            return jump_address.value  # type: ignore
        else:
            self._logger.debug(
                f"Don't branch, since {instr}(a={a}, b={b}) "
                f"is False, with values from registers {registers}"
            )
            return None

    def _interpret_binary_classical_instr(
        self,
        pid: int,
        instr: Union[
            core.ClassicalOpInstruction,
            core.ClassicalOpModInstruction,
        ],
    ) -> Optional[Generator[EventExpression, None, None]]:
        qnos_mem = self._prog_mem().qnos_mem
        mod = None
        if isinstance(instr, core.ClassicalOpModInstruction):
            mod = qnos_mem.get_reg_value(instr.regmod)
        if mod is not None and mod < 1:
            raise RuntimeError(f"Modulus needs to be greater or equal to 1, not {mod}")
        a = qnos_mem.get_reg_value(instr.regin0)
        b = qnos_mem.get_reg_value(instr.regin1)
        assert a is not None
        assert b is not None
        value = self._compute_binary_classical_instr(instr, a, b, mod=mod)
        mod_str = "" if mod is None else f"(mod {mod})"
        self._logger.debug(
            f"Performing {instr} of a={a} and b={b} {mod_str} "
            f"and storing the value {value} at register {instr.regout}"
        )
        qnos_mem.set_reg_value(instr.regout, value)
        yield from self._interface.wait(self._latencies.qnos_instr_time)
        return None

    def _compute_binary_classical_instr(
        self, instr: NetQASMInstruction, a: int, b: int, mod: Optional[int] = 1
    ) -> int:
        if isinstance(instr, core.AddInstruction):
            return a + b
        elif isinstance(instr, core.AddmInstruction):
            assert mod is not None
            return (a + b) % mod
        elif isinstance(instr, core.SubInstruction):
            return a - b
        elif isinstance(instr, core.SubmInstruction):
            assert mod is not None
            return (a - b) % mod
        else:
            raise ValueError(f"{instr} cannot be used as binary classical function")

    def _interpret_init(
        self, pid: int, instr: core.InitInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError

    def _do_single_rotation(
        self,
        pid: int,
        instr: core.RotationInstruction,
        ns_instr: NsInstr,
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.reg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        if phys_id is None:
            raise NotAllocatedError
        if isinstance(instr.angle_num, Register):
            n = qnos_mem.get_reg_value(instr.angle_num)
        else:
            n = instr.angle_num.value
        if isinstance(instr.angle_denom, Register):
            d = qnos_mem.get_reg_value(instr.angle_denom)
        else:
            d = instr.angle_denom.value
        angle = self._get_rotation_angle_from_operands(n=n, d=d)
        self._logger.debug(
            f"Performing {instr} with angle {angle} on virtual qubit "
            f"{virt_id} (physical ID: {phys_id})"
        )
        commands = [QDeviceCommand(ns_instr, [phys_id], angle=angle)]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _interpret_single_rotation_instr(
        self, pid: int, instr: nv.RotXInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError

    def _do_controlled_rotation(
        self,
        pid: int,
        instr: core.ControlledRotationInstruction,
        ns_instr: NsInstr,
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id0 = qnos_mem.get_reg_value(instr.reg0)
        phys_id0 = self._interface.memmgr.phys_id_for(pid, virt_id0)
        if phys_id0 is None:
            raise NotAllocatedError
        virt_id1 = qnos_mem.get_reg_value(instr.reg1)
        phys_id1 = self._interface.memmgr.phys_id_for(pid, virt_id1)
        if phys_id1 is None:
            raise NotAllocatedError
        angle = self._get_rotation_angle_from_operands(
            n=instr.angle_num.value,
            d=instr.angle_denom.value,
        )
        self._logger.debug(
            f"Performing {instr} with angle {angle} on virtual qubits "
            f"{virt_id0} and {virt_id1} (physical IDs: {phys_id0} and {phys_id1})"
        )
        commands = [QDeviceCommand(ns_instr, [phys_id0, phys_id1], angle=angle)]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _interpret_controlled_rotation_instr(
        self, pid: int, instr: core.ControlledRotationInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError

    def _get_rotation_angle_from_operands(self, n: int, d: int) -> float:
        return float(n * PI / (2**d))

    def _interpret_meas(
        self, pid: int, instr: core.MeasInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError

    def _interpret_create_epr(
        self, pid: int, instr: core.CreateEPRInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        return None

    def _interpret_recv_epr(
        self, pid: int, instr: core.RecvEPRInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        return None

    def _interpret_wait_all(
        self, pid: int, instr: core.WaitAllInstruction
    ) -> Generator[EventExpression, None, None]:
        raise DeprecationWarning

    def _interpret_ret_reg(
        self, pid: int, instr: core.RetRegInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        return None

    def _interpret_ret_arr(
        self, pid: int, instr: core.RetArrInstruction
    ) -> Optional[Generator[EventExpression, None, None]]:
        return None

    def _interpret_single_qubit_instr(
        self, pid: int, instr: core.SingleQubitInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError

    def _interpret_two_qubit_instr(
        self, pid: int, instr: core.SingleQubitInstruction
    ) -> Generator[EventExpression, None, None]:
        raise NotImplementedError


class GenericProcessor(QnosProcessor):
    """A `Processor` for nodes with a generic quantum hardware."""

    def _interpret_init(
        self, pid: int, instr: core.InitInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.reg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        if phys_id is None:
            raise NotAllocatedError
        self._logger.debug(
            f"Performing {instr} on virtual qubit "
            f"{virt_id} (physical ID: {phys_id})"
        )
        commands = [QDeviceCommand(INSTR_INIT, [phys_id])]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _interpret_meas(
        self, pid: int, instr: core.MeasInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.qreg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        if phys_id is None:
            raise NotAllocatedError

        self._logger.debug(
            f"Measuring qubit {virt_id} (physical ID: {phys_id}), "
            f"placing the outcome in register {instr.creg}"
        )

        commands = [QDeviceCommand(INSTR_MEASURE, [phys_id])]
        outcome = yield from self.qdevice.execute_commands(commands)
        assert outcome is not None
        qnos_mem.set_reg_value(instr.creg, outcome)
        return None

    def _interpret_single_qubit_instr(
        self, pid: int, instr: core.SingleQubitInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.qreg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        if phys_id is None:
            raise NotAllocatedError
        if isinstance(instr, vanilla.GateXInstruction):
            commands = [QDeviceCommand(INSTR_X, [phys_id])]
            yield from self.qdevice.execute_commands(commands)
        elif isinstance(instr, vanilla.GateYInstruction):
            commands = [QDeviceCommand(INSTR_Y, [phys_id])]
            yield from self.qdevice.execute_commands(commands)
        elif isinstance(instr, vanilla.GateZInstruction):
            commands = [QDeviceCommand(INSTR_Z, [phys_id])]
            yield from self.qdevice.execute_commands(commands)
        elif isinstance(instr, vanilla.GateHInstruction):
            commands = [QDeviceCommand(INSTR_H, [phys_id])]
            yield from self.qdevice.execute_commands(commands)
        else:
            raise RuntimeError(f"Unsupported instruction {instr}")
        return None

    def _interpret_single_rotation_instr(
        self, pid: int, instr: nv.RotXInstruction
    ) -> Generator[EventExpression, None, None]:
        if isinstance(instr, vanilla.RotXInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_X)
        elif isinstance(instr, vanilla.RotYInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_Y)
        elif isinstance(instr, vanilla.RotZInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_Z)
        else:
            raise RuntimeError(f"Unsupported instruction {instr}")
        return None

    def _interpret_controlled_rotation_instr(
        self, pid: int, instr: core.ControlledRotationInstruction
    ) -> Generator[EventExpression, None, None]:
        raise RuntimeError(f"Unsupported instruction {instr}")

    def _interpret_two_qubit_instr(
        self, pid: int, instr: core.SingleQubitInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id0 = qnos_mem.get_reg_value(instr.reg0)
        phys_id0 = self._interface.memmgr.phys_id_for(pid, virt_id0)
        if phys_id0 is None:
            raise NotAllocatedError
        virt_id1 = qnos_mem.get_reg_value(instr.reg1)
        phys_id1 = self._interface.memmgr.phys_id_for(pid, virt_id1)
        if phys_id1 is None:
            raise NotAllocatedError
        if isinstance(instr, vanilla.CnotInstruction):
            commands = [QDeviceCommand(INSTR_CNOT, [phys_id0, phys_id1])]
            yield from self.qdevice.execute_commands(commands)
        elif isinstance(instr, vanilla.CphaseInstruction):
            commands = [QDeviceCommand(INSTR_CZ, [phys_id0, phys_id1])]
            yield from self.qdevice.execute_commands(commands)
        else:
            raise RuntimeError(f"Unsupported instruction {instr}")
        return None


class NVProcessor(QnosProcessor):
    """A `Processor` for nodes with a NV hardware."""

    def _interpret_init(
        self, pid: int, instr: core.InitInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.reg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        if phys_id is None:
            raise NotAllocatedError
        self._logger.debug(
            f"Performing {instr} on virtual qubit "
            f"{virt_id} (physical ID: {phys_id})"
        )
        commands = [QDeviceCommand(INSTR_INIT, [phys_id])]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _measure_electron(self) -> Generator[EventExpression, None, int]:
        commands = [QDeviceCommand(INSTR_MEASURE, [0])]
        outcome = yield from self.qdevice.execute_commands(commands)
        assert outcome is not None
        return outcome  # type: ignore

    def _move_carbon_to_electron_for_measure(
        self, carbon_id: int
    ) -> Generator[EventExpression, None, None]:
        commands = [
            QDeviceCommand(INSTR_INIT, [0]),
            QDeviceCommand(INSTR_ROT_Y, [0], angle=PI_OVER_2),
            QDeviceCommand(INSTR_CYDIR, [0, carbon_id], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_ROT_X, [0], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_CXDIR, [0, carbon_id], angle=PI_OVER_2),
            QDeviceCommand(INSTR_ROT_Y, [0], angle=-PI_OVER_2),
        ]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _move_carbon_to_electron(
        self, carbon_id: int
    ) -> Generator[EventExpression, None, None]:
        # TODO: CHECK SEQUENCE OF GATES!!!
        commands = [
            QDeviceCommand(INSTR_INIT, [0]),
            QDeviceCommand(INSTR_ROT_Y, [0], angle=PI_OVER_2),
            QDeviceCommand(INSTR_CYDIR, [0, carbon_id], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_ROT_X, [0], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_CXDIR, [0, carbon_id], angle=PI_OVER_2),
            QDeviceCommand(INSTR_ROT_Y, [0], angle=-PI_OVER_2),
        ]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _move_electron_to_carbon(
        self, carbon_id: int
    ) -> Generator[EventExpression, None, None]:
        commands = [
            QDeviceCommand(INSTR_INIT, [carbon_id]),
            QDeviceCommand(INSTR_ROT_Y, [0], angle=PI_OVER_2),
            QDeviceCommand(INSTR_CYDIR, [0, carbon_id], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_ROT_X, [0], angle=-PI_OVER_2),
            QDeviceCommand(INSTR_CXDIR, [0, carbon_id], angle=PI_OVER_2),
        ]
        yield from self.qdevice.execute_commands(commands)
        return None

    def _interpret_meas(
        self, pid: int, instr: core.MeasInstruction
    ) -> Generator[EventExpression, None, None]:
        qnos_mem = self._prog_mem().qnos_mem
        virt_id = qnos_mem.get_reg_value(instr.qreg)
        phys_id = self._interface.memmgr.phys_id_for(pid, virt_id)
        assert phys_id is not None

        # Only the electron (phys ID 0) can be measured.
        # Measuring any other physical qubit (i.e one of the carbons) requires
        # freeing up the electron and moving the target qubit to the electron first.

        self._logger.debug(
            f"Measuring qubit {virt_id} (physical ID: {phys_id}), "
            f"placing the outcome in register {instr.creg}"
        )

        memmgr = self._interface.memmgr

        if phys_id == 0:
            # Measuring a comm qubit. This can be done immediately.
            outcome = yield from self._measure_electron()
            qnos_mem.set_reg_value(instr.creg, outcome)
            memmgr.free(pid, virt_id)
            return None
        # else:

        # We want to measure a mem qubit.
        # Move it to the comm qubit first.
        # Check if comm qubit is already used by this process:
        if memmgr.phys_id_for(pid, 0) is not None:
            # Comm qubit is already allocated. Try to move it to a free mem qubit.
            mem_virt_id = memmgr.get_unmapped_non_comm_qubit(pid)
            memmgr.allocate(pid, mem_virt_id)
            mem_phys_id = memmgr.phys_id_for(pid, mem_virt_id)
            assert mem_phys_id is not None

            # Move (temporarily) state from comm qubit to mem qubit.
            yield from self._move_electron_to_carbon(mem_phys_id)

            # Move state from qubit-to-measure to comm qubit.
            yield from self._move_carbon_to_electron_for_measure(phys_id)
            memmgr.free(pid, virt_id)
            self._interface.signal_memory_freed()

            # Measure comm qubit (containing state of qubit-to-measure).
            outcome = yield from self._measure_electron()
            qnos_mem.set_reg_value(instr.creg, outcome)

            # Move temporarily-moved state back to comm qubit.
            yield from self._move_carbon_to_electron(mem_phys_id)

            # Free mem qubit that was temporarily used.
            memmgr.free(pid, mem_virt_id)
            self._interface.signal_memory_freed()
        else:  # comm qubit not in use.
            # Allocate comm qubit.
            memmgr.allocate(pid, 0)

            # Move state-to-measure to comm qubit.
            yield from self._move_carbon_to_electron_for_measure(phys_id)

            # Free qubit that contained state-to-measure.
            memmgr.free(pid, virt_id)
            self._interface.signal_memory_freed()

            # Measure comm qubit.
            outcome = yield from self._measure_electron()
            qnos_mem.set_reg_value(instr.creg, outcome)

            # Free comm qubit.
            memmgr.free(pid, 0)
            self._interface.signal_memory_freed()
        return None

    def _interpret_single_rotation_instr(
        self, pid: int, instr: nv.RotXInstruction
    ) -> Generator[EventExpression, None, None]:
        if isinstance(instr, nv.RotXInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_X)
        elif isinstance(instr, nv.RotYInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_Y)
        elif isinstance(instr, nv.RotZInstruction):
            yield from self._do_single_rotation(pid, instr, INSTR_ROT_Z)
        else:
            raise RuntimeError(f"Unsupported instruction {instr}")

    def _interpret_controlled_rotation_instr(
        self, pid: int, instr: core.ControlledRotationInstruction
    ) -> Generator[EventExpression, None, None]:
        if isinstance(instr, nv.ControlledRotXInstruction):
            yield from self._do_controlled_rotation(pid, instr, INSTR_CXDIR)
        elif isinstance(instr, nv.ControlledRotYInstruction):
            yield from self._do_controlled_rotation(pid, instr, INSTR_CYDIR)
        else:
            raise RuntimeError(f"Unsupported instruction {instr}")
