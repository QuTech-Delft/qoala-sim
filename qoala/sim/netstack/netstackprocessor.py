import logging
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional

from pydynaa import EventExpression
from qoala.lang.request import CallbackType, EprType, QoalaRequest
from qoala.runtime.lhi import INSTR_MEASURE_INSTANT
from qoala.runtime.memory import ProgramMemory, RunningRequestRoutine
from qoala.runtime.message import Message, RrCallTuple
from qoala.sim.entdist.entdist import EntDistRequest
from qoala.sim.netstack.netstackinterface import NetstackInterface, NetstackLatencies
from qoala.sim.process import QoalaProcess
from qoala.sim.qdevice import QDevice, QDeviceCommand
from qoala.sim.qnos.qnosprocessor import QnosProcessor
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

    def _execute_entdist_request(
        self, request: EntDistRequest
    ) -> Generator[EventExpression, None, bool]:
        self._interface.send_entdist_msg(Message(-1, -1, request))
        result = yield from self._interface.receive_entdist_msg()
        self._logger.info(f"got result {result}")
        return result.content is not None

    def _allocate_for_pair(
        self, process: QoalaProcess, request: QoalaRequest, index: int
    ) -> int:
        memmgr = self._interface.memmgr
        pid = process.pid

        virt_id = request.virt_ids.get_id(index)
        memmgr.allocate(pid, virt_id)

        return virt_id

    def _create_entdist_request(
        self, process: QoalaProcess, request: QoalaRequest, virt_id: int
    ) -> EntDistRequest:
        memmgr = self._interface.memmgr
        pid = process.pid
        phys_id = memmgr.phys_id_for(pid, virt_id)

        epr_sck = process.epr_sockets[request.epr_socket_id]

        return EntDistRequest(
            local_node_id=self._interface.node_id,
            remote_node_id=request.remote_id,
            local_qubit_id=phys_id,
            local_pid=epr_sck.local_pid,
            remote_pid=epr_sck.remote_pid,
        )

    def measure_epr_qubit(
        self, process: QoalaProcess, virt_id: int
    ) -> Generator[EventExpression, None, int]:
        phys_id = self._interface.memmgr.phys_id_for(process.pid, virt_id)
        # Should have been allocated by `_handle_multi_pair_md`
        assert phys_id is not None
        # Use the special INSTR_MEASURE_INSTANT instruction so that measuring
        # doesn't take time.
        commands = [QDeviceCommand(INSTR_MEASURE_INSTANT, [phys_id])]
        m = yield from self.qdevice.execute_commands(commands=commands)
        assert isinstance(m, int)
        return m

    def _handle_multi_pair_ck(
        self, process: QoalaProcess, routine_name: str
    ) -> Generator[EventExpression, None, bool]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request
        assert request.typ == EprType.CREATE_KEEP
        num_pairs = request.num_pairs

        for i in range(num_pairs):
            virt_id = self._allocate_for_pair(process, request, i)
            entdist_req = self._create_entdist_request(process, request, virt_id)
            result = yield from self._execute_entdist_request(entdist_req)
            if not result:
                self._interface.memmgr.free(process.pid, virt_id)
                return False
        return True

    def _handle_multi_pair_md(
        self, process: QoalaProcess, routine_name: str
    ) -> Generator[EventExpression, None, bool]:
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
                virt_id = self._allocate_for_pair(process, request, i)
                entdist_req = self._create_entdist_request(process, request, virt_id)
                # Create EPR pair
                result = yield from self._execute_entdist_request(entdist_req)
                if result:
                    # Measure local qubit
                    m = yield from self.measure_epr_qubit(process, virt_id)
                    outcomes.append(m)
                # Free virt qubit
                self._interface.memmgr.free(process.pid, virt_id)
                if not result:
                    return False

        shared_mem = process.prog_memory.shared_mem
        results_addr = running_routine.result_addr
        shared_mem.write_rr_out(results_addr, outcomes)
        return True

    def handle_multi_pair(
        self, process: QoalaProcess, routine_name: str
    ) -> Generator[EventExpression, None, bool]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request

        if request.typ == EprType.CREATE_KEEP:
            result = yield from self._handle_multi_pair_ck(process, routine_name)
        elif request.typ == EprType.MEASURE_DIRECTLY:
            result = yield from self._handle_multi_pair_md(process, routine_name)
        else:
            raise NotImplementedError
        return result

    def handle_multi_pair_callback(
        self, process: QoalaProcess, routine_name: str, qnosprocessor: QnosProcessor
    ) -> Generator[EventExpression, None, None]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine

        assert routine.callback is not None
        cb_routine = process.get_local_routine(routine.callback)
        for virt_id in cb_routine.metadata.qubit_use:
            if self._interface.memmgr.phys_id_for(process.pid, virt_id) is None:
                self._interface.memmgr.allocate(process.pid, virt_id)

        # for WAIT_ALL, there should be at most 1 callback.
        # So we can access index 0 of the cb input/output addresses.
        yield from qnosprocessor.assign_local_routine(
            process=process,
            routine_name=routine.callback,
            input_addr=running_routine.cb_input_addrs[0],
            result_addr=running_routine.cb_output_addrs[0],
        )

        # Free CR qubits
        for virt_id in cb_routine.metadata.qubit_use:
            if virt_id not in cb_routine.metadata.qubit_keep:
                self._interface.memmgr.free(process.pid, virt_id)

    def handle_single_pair(
        self, process: QoalaProcess, routine_name: str, index: int
    ) -> Generator[EventExpression, None, bool]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request

        virt_id = self._allocate_for_pair(process, request, index)
        entdist_req = self._create_entdist_request(process, request, virt_id)
        result = yield from self._execute_entdist_request(entdist_req)
        if not result:
            self._interface.memmgr.free(process.pid, virt_id)
        return result

    def handle_single_pair_callback(
        self,
        process: QoalaProcess,
        routine_name: str,
        qnosprocessor: QnosProcessor,
        index: int,
    ) -> Generator[EventExpression, None, None]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine

        assert routine.callback is not None
        cb_routine = process.get_local_routine(routine.callback)
        for virt_id in cb_routine.metadata.qubit_use:
            if self._interface.memmgr.phys_id_for(process.pid, virt_id) is None:
                self._interface.memmgr.allocate(process.pid, virt_id)

        assert qnosprocessor is not None
        yield from qnosprocessor.assign_local_routine(
            process=process,
            routine_name=routine.callback,
            input_addr=running_routine.cb_input_addrs[index],
            result_addr=running_routine.cb_output_addrs[index],
        )

        # Free CR qubits
        for virt_id in cb_routine.metadata.qubit_use:
            if virt_id not in cb_routine.metadata.qubit_keep:
                self._interface.memmgr.free(process.pid, virt_id)

    def instantiate_routine(
        self,
        process: QoalaProcess,
        rrcall: RrCallTuple,
        args: Dict[str, Any],
    ) -> None:
        """Instantiates and activates routine."""
        routine = process.get_request_routine(rrcall.routine_name)
        instance = deepcopy(routine)
        instance.instantiate(args)

        running_routine = RunningRequestRoutine(
            instance,
            rrcall.input_addr,
            rrcall.result_addr,
            rrcall.cb_input_addrs,
            rrcall.cb_output_addrs,
        )
        process.qnos_mem.add_running_request_routine(running_routine)

    def _handle_req_routine_md(
        self, process: QoalaProcess, routine_name: str
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
                virt_id = self._allocate_for_pair(process, request, i)
                entdist_req = self._create_entdist_request(process, request, virt_id)
                # Create EPR pair
                yield from self._execute_entdist_request(entdist_req)
                # Measure local qubit
                m = yield from self.measure_epr_qubit(process, virt_id)
                # Free virt qubit
                self._interface.memmgr.free(process.pid, virt_id)
                outcomes.append(m)

        shared_mem = process.prog_memory.shared_mem
        results_addr = running_routine.result_addr
        shared_mem.write_rr_out(results_addr, outcomes)

    def _handle_req_routine_ck(
        self,
        process: QoalaProcess,
        routine_name: str,
        qnosprocessor: Optional[QnosProcessor] = None,
    ) -> Generator[EventExpression, None, None]:
        running_routine = process.qnos_mem.get_running_request_routine(routine_name)
        routine = running_routine.routine
        request = routine.request
        num_pairs = request.num_pairs

        if routine.callback_type == CallbackType.SEQUENTIAL:
            for i in range(num_pairs):
                virt_id = self._allocate_for_pair(process, request, i)
                entdist_req = self._create_entdist_request(process, request, virt_id)
                yield from self._execute_entdist_request(entdist_req)

                if routine.callback is not None:
                    # TODO: write CR inputs to shared memory

                    # Allocate qubits for CR
                    cb_routine = process.get_local_routine(routine.callback)
                    for virt_id in cb_routine.metadata.qubit_use:
                        if (
                            self._interface.memmgr.phys_id_for(process.pid, virt_id)
                            is None
                        ):
                            self._interface.memmgr.allocate(process.pid, virt_id)

                    assert qnosprocessor is not None
                    yield from qnosprocessor.assign_local_routine(
                        process=process,
                        routine_name=routine.callback,
                        input_addr=running_routine.cb_input_addrs[i],
                        result_addr=running_routine.cb_output_addrs[i],
                    )

                    # Free CR qubits
                    for virt_id in cb_routine.metadata.qubit_use:
                        if virt_id not in cb_routine.metadata.qubit_keep:
                            self._interface.memmgr.free(process.pid, virt_id)
        else:
            for i in range(num_pairs):
                virt_id = self._allocate_for_pair(process, request, i)
                entdist_req = self._create_entdist_request(process, request, virt_id)
                yield from self._execute_entdist_request(entdist_req)

            if routine.callback is not None:
                # TODO: write CR inputs to shared memory

                # Allocate qubits for CR
                cb_routine = process.get_local_routine(routine.callback)
                for virt_id in cb_routine.metadata.qubit_use:
                    if self._interface.memmgr.phys_id_for(process.pid, virt_id) is None:
                        self._interface.memmgr.allocate(process.pid, virt_id)

                # for WAIT_ALL, there should be at most 1 callback.
                # So we can access index 0 of the cb input/output addresses.

                assert qnosprocessor is not None
                yield from qnosprocessor.assign_local_routine(
                    process=process,
                    routine_name=routine.callback,
                    input_addr=running_routine.cb_input_addrs[0],
                    result_addr=running_routine.cb_output_addrs[0],
                )

                # Free CR qubits
                for virt_id in cb_routine.metadata.qubit_use:
                    if virt_id not in cb_routine.metadata.qubit_keep:
                        self._interface.memmgr.free(process.pid, virt_id)

    def assign_request_routine(
        self,
        process: QoalaProcess,
        rrcall: RrCallTuple,
        qnosprocessor: Optional[QnosProcessor] = None,
    ) -> Generator[EventExpression, None, None]:
        routine = process.get_request_routine(rrcall.routine_name)
        global_args = process.prog_instance.inputs.values
        self.instantiate_routine(process, rrcall, global_args)

        if routine.request.typ == EprType.CREATE_KEEP:
            yield from self._handle_req_routine_ck(
                process, rrcall.routine_name, qnosprocessor
            )
        elif routine.request.typ == EprType.MEASURE_DIRECTLY:
            yield from self._handle_req_routine_md(process, rrcall.routine_name)
        else:
            raise NotImplementedError
