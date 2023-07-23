from typing import Dict, Generator

import netsquid as ns

from pydynaa import EventExpression
from qoala.runtime.task import ProcessorType, QoalaTask, TaskGraph
from qoala.sim.driver import Driver
from qoala.sim.events import EVENT_WAIT
from qoala.sim.scheduler import CpuEdfScheduler, Status
from qoala.util.logging import LogManager


class SimpleTask(QoalaTask):
    def __init__(self, task_id: int, duration: int) -> None:
        super().__init__(task_id, ProcessorType.CPU, 0, duration)


class MockDriver(Driver):
    def __init__(self) -> None:
        # Task ID -> timestamp of start
        self._executed_tasks: Dict[int, float] = {}

    def wait(self, delta_time: float) -> Generator[EventExpression, None, None]:
        self._schedule_after(delta_time, EVENT_WAIT)
        event_expr = EventExpression(source=self, event_type=EVENT_WAIT)
        yield event_expr

    def handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, None]:
        now = ns.sim_time()
        self._executed_tasks[task.task_id] = now
        yield from self.wait(task.duration)
        return True
        yield


def test_update_status_one_root():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0, 200), SimpleTask(1, 500)])
    graph.add_precedences([(0, 1)])
    graph.add_rel_deadlines([((0, 1), 100)])
    graph.get_tasks()

    scheduler = CpuEdfScheduler("sched", 0, MockDriver(), None, None)
    scheduler.add_tasks(graph.get_tasks())
    scheduler.update_status()
    assert scheduler.status.status == {Status.NEXT_TASK}
    assert scheduler.status.params == {"task_id": 0}


def test_update_status_two_roots():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0, 200), SimpleTask(1, 500)])
    graph.add_deadlines([(0, 1000), (1, 500)])

    scheduler = CpuEdfScheduler("sched", 0, MockDriver(), None, None)
    scheduler.add_tasks(graph.get_tasks())
    scheduler.update_status()
    assert scheduler.status.status == {Status.NEXT_TASK}
    assert scheduler.status.params == {"task_id": 1}


def test_edf_1():
    LogManager.enable_task_logger(True)
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0, 200), SimpleTask(1, 500)])
    graph.add_precedences([(0, 1)])
    graph.add_rel_deadlines([((0, 1), 100)])

    scheduler = CpuEdfScheduler("sched", 0, MockDriver(), None, None)
    scheduler.add_tasks(graph.get_tasks())

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert scheduler._driver._executed_tasks == {0: 0, 1: 200}


def test_edf_2():
    graph = TaskGraph()
    graph.add_tasks(
        [SimpleTask(1, 500), SimpleTask(2, 80), SimpleTask(3, 300), SimpleTask(4, 100)]
    )
    graph.add_precedences([(1, 2), (1, 3), (2, 4)])
    graph.add_rel_deadlines([((1, 2), 200), ((1, 3), 400), ((2, 4), 100)])

    scheduler = CpuEdfScheduler("sched", 0, MockDriver(), None, None)
    scheduler.add_tasks(graph.get_tasks())

    ns.sim_reset()
    scheduler.start()
    ns.sim_run()

    assert scheduler._driver._executed_tasks == {1: 0, 2: 500, 4: 580, 3: 680}


if __name__ == "__main__":
    test_update_status_one_root()
    test_update_status_two_roots()
    test_edf_1()
    test_edf_2()
