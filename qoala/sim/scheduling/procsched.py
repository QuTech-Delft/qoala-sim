from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import netsquid as ns
from netsquid.components.component import Component, Port
from netsquid.protocols import Protocol

from pydynaa import EventExpression
from qoala.runtime.task import QoalaTask, TaskGraph, TaskInfo
from qoala.sim.driver import Driver
from qoala.runtime.message import Message
from qoala.sim.events import EVENT_WAIT, SIGNAL_TASK_COMPLETED
from qoala.sim.memmgr import MemoryManager
from qoala.sim.scheduling.schedmsg import TaskFinishedMsg
from qoala.util.logging import LogManager


class ProcessorSchedulerComponent(Component):
    """
    NetSquid component representing for the ProcessorScheduler.
    It is used to receive messages from the node scheduler.

    :param name: Name of the component
    """

    def __init__(self, name):
        super().__init__(name=name)
        self.add_ports(["node_scheduler_in", "node_scheduler_out"])

    @property
    def node_scheduler_in_port(self) -> Port:
        """
        Port that the node scheduler uses to send messages to this component.
        """
        return self.ports["node_scheduler_in"]

    @property
    def node_scheduler_out_port(self) -> Port:
        """
        Port that this component uses to send messages to the node scheduler.
        """
        return self.ports["node_scheduler_out"]

    def send_node_scheduler_message(self, msg: Message) -> None:
        """
        Send a message to the CPU scheduler.
        :param msg: Message to send.
        :return: None
        """
        self.node_scheduler_out_port.tx_output(msg)


@dataclass
class ActiveCriticalSection:
    pid: int  # program instance ID
    cs_id: int  # critical section ID


class ProcessorScheduler(Protocol):

    def __init__(
        self,
        name: str,
        node_id: int,
        driver: Driver,
        memmgr: MemoryManager,
        deterministic: bool = True,
        use_deadlines: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.add_signal(SIGNAL_TASK_COMPLETED)

        self._logger: logging.Logger = LogManager.get_stack_logger(  # type: ignore
            f"{self.__class__.__name__}_{driver.__class__.__name__}({name})"
        )
        self._node_id = node_id
        self._task_logger = LogManager.get_task_logger(name)
        self._driver = driver
        self._other_scheduler: Optional[ProcessorScheduler] = None
        self._memmgr = memmgr
        self._deterministic = deterministic
        self._use_deadlines = use_deadlines

        self._task_graph: TaskGraph = TaskGraph()
        self._finished_tasks: List[int] = []

        self._prog_start_timestamps: Dict[int, float] = {}  # program ID -> start time
        self._prog_end_timestamps: Dict[int, float] = {}  # program ID -> end time

        self._tasks_executed: Dict[int, QoalaTask] = {}
        self._task_starts: Dict[int, float] = {}
        self._task_ends: Dict[int, float] = {}
        self.last_finished_task_pid: Tuple[int, int] = (-1, -1)  # (pid, end_time)

        self._comp = ProcessorSchedulerComponent(name + "_comp")

        self._status: SchedulerStatus = SchedulerStatus(status=set(), params={})
        self._critical_section: Optional[ActiveCriticalSection] = None

    @property
    def node_scheduler_in_port(self) -> Port:
        return self._comp.node_scheduler_in_port

    @property
    def node_scheduler_out_port(self) -> Port:
        return self._comp.node_scheduler_out_port

    @property
    def driver(self) -> Driver:
        return self._driver

    @property
    def status(self) -> SchedulerStatus:
        return self._status

    def update_external_predcessors(self) -> None:
        if self._other_scheduler is None:
            return
        assert self._task_graph is not None

        tg = self._task_graph

        for r in tg.get_roots(ignore_external=True):
            ext_preds = tg.get_tinfo(r).ext_predecessors
            new_ext_preds = {
                ext for ext in ext_preds if not self._other_scheduler.has_finished(ext)
            }
            tg.get_tinfo(r).ext_predecessors = new_ext_preds

    def upload_task_graph(self, graph: TaskGraph) -> None:
        """
        Sets the given task graph as the current task graph.

        :param graph: The task graph to upload.
        :return: None
        """
        self._task_graph = graph

    def get_last_finished_task_pid_at(self, time: float) -> int:
        """
        Get the pid of the last finished task at the given time and returns it.
        If there is no task that is finished at the current time, it returns -1

        :param time: The time to check for finished tasks.
        :return: The pid of the last finished task at the given time if such task exists, -1 otherwise.
        """
        if self.last_finished_task_pid[1] == time:
            return self.last_finished_task_pid[0]
        else:
            return -1

    def task_exists_for_pid(self, pid: int) -> bool:
        """
        Checks the current task graph for the existence of a task with the given pid. Returns True if such task exists,
        False otherwise.

        :param pid: The pid to check for.
        :return: True if a task with the given pid exists, False otherwise.
        """
        return self._task_graph.task_exists_for_pid(pid)

    def add_tasks(self, tasks: Dict[int, TaskInfo]) -> None:
        """
        Adds the given tasks to the current task graph.

        :param tasks: The tasks to add.
        :return: None
        """
        self._task_graph.get_tasks().update(tasks)

    def has_finished(self, task_id: int) -> bool:
        return task_id in self._finished_tasks

    def set_other_scheduler(self, other: ProcessorScheduler) -> None:
        self._other_scheduler = other

    def record_start_timestamp(self, pid: int, time: float) -> None:
        # Only write start time for first encountered task.
        if pid not in self._prog_start_timestamps:
            self._prog_start_timestamps[pid] = time

    def record_end_timestamp(self, pid: int, time: float) -> None:
        # Overwrite end time for every task. Automatically the last timestamp remains.
        self._prog_end_timestamps[pid] = time

    def get_timestamps(self, pid: int) -> Optional[Tuple[float, float]]:
        if pid not in self._prog_start_timestamps:
            assert pid not in self._prog_end_timestamps
            return None
        assert pid in self._prog_end_timestamps
        return self._prog_start_timestamps[pid], self._prog_end_timestamps[pid]

    def get_tasks_executed(self) -> Dict[int, QoalaTask]:
        return self._tasks_executed

    def get_task_starts(self) -> Dict[int, float]:
        return self._task_starts

    def get_task_ends(self) -> Dict[int, float]:
        return self._task_ends

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def handle_task(self, task_id: int) -> Generator[EventExpression, None, None]:
        assert self._task_graph is not None
        tinfo = self._task_graph.get_tinfo(task_id)
        task = tinfo.task

        # TODO: do something with critical sections
        # task.critical_section
        if task.critical_section is not None:
            self._critical_section = ActiveCriticalSection(
                task.pid, task.critical_section
            )
            self._task_logger.info(
                f"setting critical_section to {self._critical_section} because starting task {task}"
            )

        self._logger.debug(f"{ns.sim_time()}: {self.name}: checking next task {task}")

        before = ns.sim_time()

        start_time = self._task_graph.get_tinfo(task.task_id).start_time
        is_busy_task = start_time is not None
        self._logger.info(f"executing task {task}")
        if is_busy_task:
            self._task_logger.info(f"BUSY start  {task} (start time: {start_time})")
        else:
            self._task_logger.info(f"start  {task}")
        self._task_starts[task.task_id] = before
        self.record_start_timestamp(task.pid, before)

        # Execute the task
        success = yield from self._driver.handle_task(task)
        if success:
            after = ns.sim_time()

            self.record_end_timestamp(task.pid, after)
            self.last_finished_task_pid = (task.pid, after)
            duration = after - before
            self._task_graph.decrease_deadlines(duration)
            self._task_graph.remove_task(task_id)

            self._finished_tasks.append(task.task_id)

            # Send (1) signal and (2) message saying the task completed.
            # (1) signal: no info other than that "a" task finished;
            #             it is broadcast, so the other proc scheduler sees it
            # (2) Message to Node Scheduler, with info which task finished.
            self.send_signal(SIGNAL_TASK_COMPLETED)
            msg = TaskFinishedMsg(task.processor_type, task.pid, task.task_id)
            self._comp.send_node_scheduler_message(Message(-1, -1, msg))

            self._logger.info(f"finished task {task}")
            if is_busy_task:
                self._task_logger.info(f"BUSY finish {task}")
            else:
                self._task_logger.info(f"finish {task}")

            self._tasks_executed[task.task_id] = task
            self._task_ends[task.task_id] = after
        else:
            self._task_logger.info("task failed")


class Status(Enum):
    GRAPH_EMPTY = auto()
    EPR_GEN = auto()
    NEXT_TASK = auto()
    WAITING_OTHER_CORE = auto()
    WAITING_MSG = auto()
    WAITING_START_TIME = auto()
    WAITING_RESOURCES = auto()
    WAITING_TIME_BIN = auto()


@dataclass
class SchedulerStatus:
    status: Set[Status]
    params: Dict[str, Any]
