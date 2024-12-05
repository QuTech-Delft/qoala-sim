from __future__ import annotations

import random
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns
from netqasm.lang.operand import Template

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkSchedule, EhiNetworkTimebin
from qoala.lang.request import VirtIdMappingType
from qoala.runtime.task import LocalRoutineTask, MultiPairTask, SinglePairTask
from qoala.sim.driver import QpuDriver
from qoala.sim.events import EVENT_WAIT, SIGNAL_MEMORY_FREED, SIGNAL_TASK_COMPLETED
from qoala.sim.memmgr import AllocError, MemoryManager
from qoala.sim.scheduling.procsched import ProcessorScheduler, SchedulerStatus, Status


class QpuScheduler(ProcessorScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: QpuDriver,
        memmgr: MemoryManager,
        network_schedule: Optional[EhiNetworkSchedule] = None,
        deterministic: bool = True,
        use_deadlines: bool = True,
        prio_epr: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )
        self._network_schedule = network_schedule
        self._prio_epr = prio_epr

    def timebin_for_task(self, tid: int) -> EhiNetworkTimebin:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        assert isinstance(task, SinglePairTask) or isinstance(task, MultiPairTask)
        drv_mem = self._driver._memory
        rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
        process = self._memmgr.get_process(task.pid)
        routine = process.get_request_routine(rrcall.routine_name)
        request = routine.request
        epr_sck = process.epr_sockets[request.epr_socket_id]
        return EhiNetworkTimebin(
            nodes=frozenset({self._node_id, epr_sck.remote_id}),
            pids={
                self._node_id: epr_sck.local_pid,
                epr_sck.remote_id: epr_sck.remote_pid,
            },
        )

    def are_resources_available(self, tid: int) -> bool:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        self._task_logger.debug(f"check if resources available for task {tid} ({task})")
        phys_qubits_in_use = [
            i for i, vmap in self._memmgr._physical_mapping.items() if vmap is not None
        ]
        self._task_logger.debug(f"physical qubits in use: {phys_qubits_in_use}")
        if isinstance(task, SinglePairTask):
            # TODO: refactor
            drv_mem = self._driver._memory
            rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            routine = process.get_request_routine(rrcall.routine_name)

            # Get virt ID which would be need to be allocated
            virt_id = routine.request.virt_ids.get_id(task.pair_index)

            # Check if virt ID is available by trying to allocate
            # (without actually allocating)

            try:
                self._memmgr.allocate(task.pid, virt_id)
                self._memmgr.free(task.pid, virt_id, send_signal=False)
                return True
            except AllocError:
                return False
        elif isinstance(task, MultiPairTask):
            # TODO: refactor
            drv_mem = self._driver._memory
            rrcall = drv_mem.read_shared_rrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            routine = process.get_request_routine(rrcall.routine_name)

            # Hack to get num_pairs (see comment in hostprocessor.py)
            prog_input = process.prog_instance.inputs.values
            if isinstance(routine.request.num_pairs, Template):
                template_name = routine.request.num_pairs.name
                num_pairs = prog_input[template_name]
            else:
                num_pairs = routine.request.num_pairs

            # Get virt IDs which would be need to be allocated
            if routine.request.virt_ids.typ == VirtIdMappingType.EQUAL:
                virt_id = routine.request.virt_ids.single_value  # type: ignore
                assert virt_id is not None and isinstance(virt_id, int)
                virt_ids = [virt_id]
            else:
                virt_ids = [
                    routine.request.virt_ids.get_id(i) for i in range(num_pairs)
                ]

            # Check if virt IDs are available by trying to allocate
            # (without actually allocating)
            try:
                self._task_logger.debug(f"trying to allocate virt IDs {virt_ids}")
                temp_allocated: List[int] = []
                for virt_id in virt_ids:
                    self._memmgr.allocate(task.pid, virt_id)
                    temp_allocated.append(virt_id)  # successful alloc
                # Free all temporarily allocated qubits again
                for virt_id in temp_allocated:
                    self._memmgr.free(task.pid, virt_id, send_signal=False)
                self._task_logger.debug("all virt IDs available")
                return True
            except AllocError:
                # Make sure all qubits that did successfully allocate are freed
                for virt_id in temp_allocated:
                    self._memmgr.free(task.pid, virt_id, send_signal=False)
                self._task_logger.debug("some virt IDs unavailable")
                return False
        elif isinstance(task, LocalRoutineTask):
            drv_mem = self._driver._memory
            lrcall = drv_mem.read_shared_lrcall(task.shared_ptr)
            process = self._memmgr.get_process(task.pid)
            local_routine = process.get_local_routine(lrcall.routine_name)
            virt_ids = local_routine.metadata.qubit_use
            try:
                # get qubit IDs that are not already allocated
                new_ids = [
                    vid
                    for vid in virt_ids
                    if self._memmgr.phys_id_for(task.pid, vid) is None
                ]
                # try to allocate them
                temp_allocated = []
                for virt_id in new_ids:
                    self._memmgr.allocate(task.pid, virt_id)
                    temp_allocated.append(virt_id)  # successful alloc
                # Free all temporarily allocated qubits again
                for virt_id in temp_allocated:
                    self._memmgr.free(task.pid, virt_id, send_signal=False)
                self._task_logger.debug("all virt IDs available")
                return True
            except AllocError:
                # Make sure all qubits that did successfully allocate are freed
                for virt_id in temp_allocated:
                    self._memmgr.free(task.pid, virt_id, send_signal=False)
                self._task_logger.debug("some virt IDs unavailable")
                return False
        else:
            self._logger.info(
                f"Checking if resources are available for task type {type(task)}, "
                "returning True but no actual check is implemented"
            )
            # NOTE: we assume that callback tasks never allocate any additional
            # resources so they can always return `True` here.
            return True

    def update_status(self) -> None:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            self._status = SchedulerStatus(status={Status.GRAPH_EMPTY}, params={})
            return

        # All tasks that have no predecessors, internal nor external.
        no_predecessors = tg.get_roots()
        self._task_logger.debug(
            f"no_predecessors: {[str(tg.get_tinfo(t).task) for t in no_predecessors]}"
        )

        # All tasks that have only external predecessors.
        blocked_on_other_core = tg.get_tasks_blocked_only_on_external()
        self._task_logger.debug(
            f"blocked_on_other_core : {[str(tg.get_tinfo(t).task) for t in blocked_on_other_core]}"
        )

        # All EPR (SinglePair or MultiPair) tasks that have no predecessors,
        # internal nor external.
        epr_no_predecessors = [
            tid for tid in no_predecessors if tg.get_tinfo(tid).task.is_epr_task()
        ]
        self._task_logger.debug(
            f"epr_no_predecessors : {[str(tg.get_tinfo(t).task) for t in epr_no_predecessors]}"
        )

        # All tasks without predecessors for which not all resources are availables.
        blocked_on_resources = [
            tid for tid in no_predecessors if not self.are_resources_available(tid)
        ]
        self._task_logger.debug(
            f"blocked_on_resources : {[str(tg.get_tinfo(t).task) for t in blocked_on_resources]}"
        )

        # All non-EPR tasks that are ready for execution.
        non_epr_ready = [
            tid
            for tid in no_predecessors
            if tid not in epr_no_predecessors and tid not in blocked_on_resources
        ]
        self._task_logger.debug(
            f"non_epr_ready : {[str(tg.get_tinfo(t).task) for t in non_epr_ready]}"
        )

        # All EPR tasks that have no predecessors and are not blocked on resources.
        epr_no_preds_not_blocked = [
            tid for tid in epr_no_predecessors if tid not in blocked_on_resources
        ]
        self._task_logger.debug(
            f"epr_no_preds_not_blocked : {[str(tg.get_tinfo(t).task) for t in epr_no_preds_not_blocked]}"
        )

        # All EPR tasks that can be immediately executed.
        epr_ready = []

        # The next EPR task (if any) that is ready to execute but needs to wait for its
        # corresponding time bin.
        epr_wait_for_bin: Optional[Tuple[int, int]] = None  # (task ID, delta)

        time_until_bin: Dict[int, int] = {}  # task ID -> time until bin

        now = ns.sim_time()
        for e in epr_no_preds_not_blocked:
            if self._network_schedule is not None:
                # First, check if the current bin allows this EPR task.
                bin = self.timebin_for_task(e)
                curr_bin = self._network_schedule.current_bin(now)
                if curr_bin and curr_bin.bin == bin and curr_bin.end - 1 != now:
                    # The current bin allows this task.
                    # The last check (end - 1 != now) is needed since we don't allow
                    # generation to start at the last 'tick' of the bin.
                    epr_ready.append(e)
                else:
                    # Find the time until the next netschedule timebin that allows this EPR task.
                    self._task_logger.info(f"EPR ready: task {e}, bin: {bin}")
                    delta = self._network_schedule.next_specific_bin(now, bin)
                    time_until_bin[e] = delta
                    self._task_logger.info(f"EPR ready: task {e}, delta: {delta}")
                    if delta == 0:
                        epr_ready.append(e)
            else:
                # No network schedule: immediately just execute the EPR task
                epr_ready.append(e)

        epr_non_zero_delta = {
            tid: delta for tid, delta in time_until_bin.items() if delta > 0
        }
        self._task_logger.info(f"epr_non_zero_delta: {epr_non_zero_delta}")
        if len(epr_non_zero_delta) > 0:
            sorted_by_delta = sorted(
                epr_non_zero_delta.items(), key=lambda item: item[1]
            )
            earliest, delta = sorted_by_delta[0]
            epr_wait_for_bin = (earliest, delta)

        self._task_logger.info(f"epr_wait_for_bin: {epr_wait_for_bin}")

        if len(epr_ready) > 0:
            self._task_logger.info(f"epr_ready: {epr_ready}")
            self._status = SchedulerStatus(
                status={Status.EPR_GEN}, params={"task_id": epr_ready[0]}
            )
        elif len(non_epr_ready) > 0:
            with_deadline = [
                t for t in non_epr_ready if tg.get_tinfo(t).deadline is not None
            ]

            if not self._use_deadlines:
                with_deadline = []
            if len(with_deadline) > 0:
                # Sort them by deadline and return the one with the earliest deadline
                deadlines = {t: tg.get_tinfo(t).deadline for t in with_deadline}
                sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
                to_return = sorted_by_deadline[0][0]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
                self._status = SchedulerStatus(
                    status={Status.NEXT_TASK}, params={"task_id": to_return}
                )
            else:
                # No deadlines
                if self._deterministic:
                    index = 0
                else:
                    index = random.randint(0, len(non_epr_ready) - 1)
                to_return = non_epr_ready[index]
                self._logger.debug(f"Return task {to_return}")
                self._task_logger.debug(f"Return task {to_return}")
                self._status = SchedulerStatus(
                    status={Status.NEXT_TASK}, params={"task_id": to_return}
                )
        else:
            if len(blocked_on_other_core) > 0:
                self._logger.debug("Waiting other core")
                self._task_logger.debug("Waiting other core")
                self._status.status.add(Status.WAITING_OTHER_CORE)
            if len(blocked_on_resources) > 0:
                self._logger.debug("Waiting resources")
                self._task_logger.debug("Waiting resources")
                self._status.status.add(Status.WAITING_RESOURCES)
            if epr_wait_for_bin is not None:
                self._logger.debug("Waiting time bin")
                self._task_logger.debug("Waiting time bin")
                task_id, delta = epr_wait_for_bin
                self._status.status.add(Status.WAITING_TIME_BIN)
                self._status.params["delta"] = delta

            if len(self.status.status) == 0:
                raise RuntimeError

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            self._status = SchedulerStatus(status=set(), params={})
            self.update_external_predcessors()
            self.update_status()
            self._task_logger.debug(f"status: {self.status.status}")
            if Status.EPR_GEN in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            elif Status.NEXT_TASK in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            else:
                ev_expr = self.await_port_input(self.node_scheduler_in_port)
                if Status.WAITING_OTHER_CORE in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._other_scheduler,
                        signal_label=SIGNAL_TASK_COMPLETED,
                    )
                if Status.WAITING_RESOURCES in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._memmgr,
                        signal_label=SIGNAL_MEMORY_FREED,
                    )
                if Status.WAITING_TIME_BIN in self.status.status:
                    delta = self.status.params["delta"]
                    self._schedule_after(delta, EVENT_WAIT)
                    ev_timebin = EventExpression(source=self, event_type=EVENT_WAIT)
                    ev_expr = ev_expr | ev_timebin
                self._task_logger.debug(f"yielding on {ev_expr}")
                yield ev_expr
