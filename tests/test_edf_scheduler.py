from typing import Generator
from unittest.mock import Mock

from pydynaa import EventExpression
from qoala.runtime.task import HostLocalTask, ProcessorType, QoalaTask, TaskGraph
from qoala.sim.driver import Driver
from qoala.sim.scheduler import EdfScheduler


class SimpleTask(QoalaTask):
    def __init__(self, task_id: int) -> None:
        super().__init__(task_id, ProcessorType.CPU, 0)


class MockDriver(Driver):
    def __init__(self) -> None:
        pass

    def handle_task(self, task: QoalaTask) -> Generator[EventExpression, None, None]:
        return None


def test_next_task():
    tasks = {0: SimpleTask(0), 1: SimpleTask(1)}
    precedences = [(0, 1)]
    rel_deadlines = {1: {0: 100}}
    graph = TaskGraph(tasks, precedences, [], rel_deadlines, {})

    scheduler = EdfScheduler(name="sched", driver=MockDriver())
    scheduler.init_task_graph(graph)
    assert scheduler.next_task() == 0


if __name__ == "__main__":
    test_next_task()
