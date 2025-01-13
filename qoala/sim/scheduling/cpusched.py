from __future__ import annotations

import random
from abc import abstractmethod
from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns

from pydynaa import EventExpression
from qoala.lang import hostlang
from qoala.lang.hostlang import ReceiveCMsgOp
from qoala.runtime.task import HostEventTask
from qoala.sim.driver import CpuDriver
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.host.hostinterface import HostInterface
from qoala.sim.memmgr import MemoryManager
from qoala.sim.scheduling.procsched import ProcessorScheduler, SchedulerStatus, Status


class CpuScheduler(ProcessorScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: CpuDriver,
        memmgr: MemoryManager,
        host_interface: HostInterface,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )
        self._host_interface = host_interface

    def is_message_available(self, tid: int) -> bool:
        assert self._task_graph is not None
        task = self._task_graph.get_tinfo(tid).task
        assert isinstance(task, HostEventTask)
        process = self._memmgr.get_process(task.pid)
        block = process.program.get_block(task.block_name)
        instr = block.instructions[0]
        assert isinstance(instr, ReceiveCMsgOp)
        assert isinstance(instr.arguments[0], hostlang.IqoalaSingleton)
        csck_id = process.host_mem.read(instr.arguments[0].name)
        csck = process.csockets[csck_id]
        remote_name = csck.remote_name
        remote_pid = csck.remote_pid
        self._task_logger.debug(f"checking if msg from {remote_name} is available")
        messages = self._host_interface.get_available_messages(remote_name)
        if (remote_pid, task.pid) in messages:
            self._task_logger.debug(f"task {tid} NOT blocked on message")
            return True
        else:
            self._task_logger.debug(f"task {tid} blocked on message")
            return False

    @abstractmethod
    def choose_next_task(self, ready_tasks: List[int]) -> None:
        raise NotImplementedError

    def update_status(self) -> None:
        tg = self._task_graph

        if tg is None or len(tg.get_tasks()) == 0:
            # No tasks in the task graph.
            # If we were in a critical section, we have now completed it.
            self._critical_section = None
            # Return empty graph status.
            self._status = SchedulerStatus(status={Status.GRAPH_EMPTY}, params={})
            return

        # All tasks that have no predecessors, internal nor external.
        no_predecessors = tg.get_roots()
        # If we are in a CS, only tasks in that CS are eligible, so apply a filter.
        if self._critical_section:
            no_predecessors = [
                t
                for t in no_predecessors
                if tg.get_tinfo(t).task.critical_section == self._critical_section
            ]

        # All tasks that have only external predecessors.
        blocked_on_other_core = tg.get_tasks_blocked_only_on_external()
        # If we are in a CS, only tasks in that CS are eligible, so apply a filter.
        if self._critical_section:
            blocked_on_other_core = [
                t
                for t in blocked_on_other_core
                if tg.get_tinfo(t).task.critical_section == self._critical_section
            ]

        # All "receive message" tasks without predecessors (internal nor external).
        event_no_predecessors = [
            tid for tid in no_predecessors if tg.get_tinfo(tid).task.is_event_task()
        ]

        event_blocked_on_message = [
            tid for tid in event_no_predecessors if not self.is_message_available(tid)
        ]
        self._task_logger.info(f"event_blocked_on_message: {event_blocked_on_message}")

        now = ns.sim_time()
        with_future_start: Dict[int, float] = {
            tid: tg.get_tinfo(tid).start_time  # type: ignore
            for tid in no_predecessors
            if tg.get_tinfo(tid).start_time is not None
            and tg.get_tinfo(tid).start_time > now
        }
        wait_for_start: Optional[Tuple[int, float]] = None  # (task ID, start time)
        if len(with_future_start) > 0:
            sorted_by_start = sorted(
                with_future_start.items(), key=lambda item: item[1]
            )
            wait_for_start = sorted_by_start[0]
        self._task_logger.info(f"wait_for_start: {wait_for_start}")

        ready = [
            tid
            for tid in no_predecessors
            if tid not in event_blocked_on_message and tid not in with_future_start
        ]
        ready_task_dict = {tid: str(tg.get_tinfo(tid).task) for tid in ready}
        self._task_logger.debug(f"ready tasks: {ready}\n{ready_task_dict}")

        if len(ready) > 0:
            # self._task_logger.warning(f"ready tasks: {ready}")
            # From the readily executable tasks, choose which one to execute
            self.choose_next_task(ready)
        else:
            if len(blocked_on_other_core) > 0:
                self._logger.debug("Waiting other core")
                self._task_logger.debug("Waiting other core")
                self._status.status.add(Status.WAITING_OTHER_CORE)
            if len(event_blocked_on_message) > 0:
                self._logger.debug("Waiting message")
                self._task_logger.debug("Waiting message")
                self._status.status.add(Status.WAITING_MSG)
            if wait_for_start is not None:
                _, start = wait_for_start
                self._logger.debug("Waiting Start Time")
                self._task_logger.debug("Waiting Start Time")
                self._status.status.add(Status.WAITING_START_TIME)
                self._status.params["start_time"] = start

            if len(self.status.status) == 0:
                raise RuntimeError

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._task_logger.debug("updating status...")
            self._status = SchedulerStatus(status=set(), params={})
            self.update_external_predcessors()
            self.update_status()
            self._task_logger.debug(f"status: {self.status.status}")
            if Status.NEXT_TASK in self.status.status:
                task_id = self.status.params["task_id"]
                yield from self.handle_task(task_id)
            else:
                ev_expr = self.await_port_input(self.node_scheduler_in_port)
                if Status.WAITING_OTHER_CORE in self.status.status:
                    ev_expr = ev_expr | self.await_signal(
                        sender=self._other_scheduler,
                        signal_label=SIGNAL_TASK_COMPLETED,
                    )
                if Status.WAITING_START_TIME in self.status.status:
                    start_time = self.status.params["start_time"]
                    now = ns.sim_time()
                    delta = start_time - now
                    self._schedule_after(delta, EVENT_WAIT)
                    ev_start_time = EventExpression(source=self, event_type=EVENT_WAIT)
                    ev_expr = ev_expr | ev_start_time

                if Status.WAITING_MSG in self.status.status:
                    ev_msg_arrived = self._host_interface.get_evexpr_for_any_msg()

                    ev_expr = ev_msg_arrived | ev_expr
                    yield ev_expr
                    if len(ev_expr.first_term.triggered_events) > 0:
                        # It was "ev_msg_arrived" that triggered.
                        # Need to process this event (flushing potential other messages)
                        yield from self._host_interface.handle_msg_evexpr(
                            ev_expr.first_term
                        )
                else:
                    yield ev_expr


class CpuEdfScheduler(CpuScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: CpuDriver,
        memmgr: MemoryManager,
        host_interface: HostInterface,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            host_interface=host_interface,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )

    def choose_next_task(self, ready_tasks: List[int]) -> None:
        tg = self._task_graph

        with_deadline = [
            t
            for t in ready_tasks
            if tg.get_tinfo(t).deadline is not None
            or len(tg.get_tinfo(t).rel_deadlines) > 0
            or len(tg.get_tinfo(t).ext_rel_deadlines) > 0
        ]
        if not self._use_deadlines:
            with_deadline = []

        self._task_logger.debug(f"ready tasks with deadline: {with_deadline}")

        to_return: int

        if len(with_deadline) > 0:
            # Sort them by deadline and return the one with the earliest deadline
            deadlines = {t: tg.get_tinfo(t).deadline for t in with_deadline}
            sorted_by_deadline = sorted(deadlines.items(), key=lambda item: item[1])  # type: ignore
            to_return = sorted_by_deadline[0][0]
            self._logger.debug(f"Return task {to_return}")
            self._task_logger.debug(f"Return task {to_return}")
        else:
            # No deadlines
            if self._deterministic:
                to_return = ready_tasks[0]
            else:
                index = random.randint(0, len(ready_tasks) - 1)
                to_return = ready_tasks[index]
            self._logger.debug(f"Return task {to_return}")
            self._task_logger.debug(f"Return task {to_return}")
        self._status = SchedulerStatus(
            status={Status.NEXT_TASK}, params={"task_id": to_return}
        )


class CpuFcfsScheduler(CpuScheduler):
    def __init__(
        self,
        name: str,
        node_id: int,
        driver: CpuDriver,
        memmgr: MemoryManager,
        host_interface: HostInterface,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_id=node_id,
            driver=driver,
            memmgr=memmgr,
            host_interface=host_interface,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
        )

        self._task_queue: List[int] = []  # list of task IDs

    def choose_next_task(self, ready_tasks: List[int]) -> None:
        for tid in ready_tasks:
            if tid not in self._task_queue:
                self._task_queue.append(tid)

        self._task_logger.debug(f"task queue: {self._task_queue}")
        next_task = self._task_queue.pop(0)
        self._task_logger.debug(f"popping: {next_task}")

        self._status = SchedulerStatus(
            status={Status.NEXT_TASK}, params={"task_id": next_task}
        )
