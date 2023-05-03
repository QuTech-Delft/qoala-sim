from typing import Dict

import pytest

from qoala.runtime.task import (
    HostEventTask,
    HostLocalTask,
    LocalRoutineTask,
    MultiPairCallbackTask,
    MultiPairTask,
    PreCallTask,
    ProcessorType,
    QoalaTask,
    TaskGraph,
    TaskInfo,
)


class SimpleTask(QoalaTask):
    def __init__(self, task_id: int) -> None:
        super().__init__(task_id, ProcessorType.CPU, 0)


def linear():
    tasks = [SimpleTask(i) for i in range(5)]
    precedences = [(i - 1, i) for i in range(1, 5)]
    rel_deadlines = [((i - 1, i), 100) for i in range(1, 5)]
    graph = TaskGraph()
    graph.add_tasks(tasks)
    graph.add_precedences(precedences)
    graph.add_rel_deadlines(rel_deadlines)

    assert graph.get_roots() == [0]
    assert graph.get_tinfo(0).predecessors == []
    assert all(graph.get_tinfo(i).predecessors == [i - 1] for i in range(1, 5))
    assert all(graph.get_tinfo(i).deadline is None for i in range(5))
    assert graph.get_tinfo(0).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(1, 5))

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.get_tinfo(0)

    assert graph.get_roots() == [1]
    assert graph.get_tinfo(1).predecessors == []
    assert all(graph.get_tinfo(i).predecessors == [i - 1] for i in range(2, 5))
    assert graph.get_tinfo(1).deadline == 100
    assert all(graph.get_tinfo(i).deadline is None for i in range(2, 5))
    assert graph.get_tinfo(1).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 5))

    with pytest.raises(AssertionError):
        # not a root
        graph.remove_task(4)


def no_precedence():
    tasks = [SimpleTask(i) for i in range(5)]
    rel_deadlines = [((i - 1, i), 100) for i in range(1, 5)]
    graph = TaskGraph()
    graph.add_tasks(tasks)
    graph.add_rel_deadlines(rel_deadlines)

    assert graph.get_roots() == [i for i in range(5)]
    assert all(graph.get_tinfo(i).predecessors == [] for i in range(5))
    assert all(graph.get_tinfo(i).deadline is None for i in range(5))
    assert graph.get_tinfo(0).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(1, 5))

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.get_tinfo(0)

    assert graph.get_roots() == [i for i in range(1, 5)]
    assert all(graph.get_tinfo(i).predecessors == [] for i in range(1, 5))
    assert graph.get_tinfo(1).deadline == 100
    assert all(graph.get_tinfo(i).deadline is None for i in range(2, 5))
    assert graph.get_tinfo(1).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 5))

    graph.remove_task(4)

    with pytest.raises(AssertionError):
        graph.get_tinfo(4)

    assert graph.get_roots() == [1, 2, 3]
    assert all(graph.get_tinfo(i).predecessors == [] for i in range(1, 4))
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 4))


def test_get_cpu_graph():
    pid = 0
    mp_ptr = 0
    lr_ptr = 1

    class TaskCounter:
        task_id = -1

        def next(self) -> int:
            self.task_id += 1
            return self.task_id

    tc = TaskCounter()
    hl1 = HostLocalTask(tc.next(), pid, "hl1")
    hl2 = HostLocalTask(tc.next(), pid, "hl2")
    hl3 = HostLocalTask(tc.next(), pid, "hl2")
    he1 = HostEventTask(tc.next(), pid, "he1")
    prc1 = PreCallTask(tc.next(), pid, "prc1", mp_ptr)
    prc2 = PreCallTask(tc.next(), pid, "prc2", lr_ptr)
    poc1 = PreCallTask(tc.next(), pid, "poc1", mp_ptr)
    poc2 = PreCallTask(tc.next(), pid, "poc2", lr_ptr)
    mp1 = MultiPairTask(tc.next(), pid, mp_ptr)
    mpc1 = MultiPairCallbackTask(tc.next(), pid, "mpc1", mp_ptr)
    lr1 = LocalRoutineTask(tc.next(), pid, "lr1", lr_ptr)

    precedences = [
        (hl1.task_id, hl2.task_id),
        (hl1.task_id, he1.task_id),
        (hl2.task_id, prc1.task_id),
        (he1.task_id, prc1.task_id),
        (he1.task_id, prc2.task_id),
        (prc1.task_id, mp1.task_id),
        (mp1.task_id, mpc1.task_id),
        (mpc1.task_id, poc1.task_id),
        (prc2.task_id, lr1.task_id),
        (mpc1.task_id, lr1.task_id),
        (lr1.task_id, poc2.task_id),
        (poc1.task_id, hl3.task_id),
        (poc2.task_id, hl3.task_id),
    ]
    graph = TaskGraph()
    graph.add_tasks([hl1, hl2, hl3, he1, prc1, prc2, poc1, poc2, mp1, mpc1, lr1])
    graph.add_precedences(precedences)

    # Test immediate cross-predecessors
    for task in [hl1, hl2, he1, prc1, prc2, mpc1, hl3]:
        assert graph.cross_predecessors(task.task_id) == set()
    assert graph.cross_predecessors(mp1.task_id) == {prc1.task_id}
    assert graph.cross_predecessors(poc1.task_id) == {mpc1.task_id}
    assert graph.cross_predecessors(lr1.task_id) == {prc2.task_id}
    assert graph.cross_predecessors(poc2.task_id) == {lr1.task_id}

    # Test indirect cross-predecessors
    for task in [hl1, hl2, he1, prc1, prc2]:
        assert graph.cross_predecessors(task.task_id, immediate=False) == set()
    assert graph.cross_predecessors(mp1.task_id, immediate=False) == {prc1.task_id}
    assert graph.cross_predecessors(mpc1.task_id, immediate=False) == {prc1.task_id}
    assert graph.cross_predecessors(poc1.task_id, immediate=False) == {mpc1.task_id}
    assert graph.cross_predecessors(lr1.task_id, immediate=False) == {
        prc1.task_id,
        prc2.task_id,
    }
    assert graph.cross_predecessors(poc2.task_id, immediate=False) == {lr1.task_id}
    assert graph.cross_predecessors(hl3.task_id, immediate=False) == {
        mpc1.task_id,
        lr1.task_id,
    }

    assert all(
        graph.double_cross_predecessors(t.task_id) == set()
        for t in [hl1, hl2, hl3, he1, prc1, prc2, mp1, mpc1, lr1]
    )
    assert graph.double_cross_predecessors(poc1.task_id) == {prc1.task_id}
    assert graph.double_cross_predecessors(poc2.task_id) == {prc1.task_id, prc2.task_id}

    # Check CPU graph
    expected_cpu_precedences = [
        (hl1.task_id, hl2.task_id),
        (hl1.task_id, he1.task_id),
        (hl2.task_id, prc1.task_id),
        (he1.task_id, prc1.task_id),
        (he1.task_id, prc2.task_id),
        (prc1.task_id, poc1.task_id),
        (prc1.task_id, poc2.task_id),
        (prc2.task_id, poc2.task_id),
        (poc1.task_id, hl3.task_id),
        (poc2.task_id, hl3.task_id),
    ]
    expected_external_cpu_precedences = [
        (mpc1.task_id, poc1.task_id),
        (lr1.task_id, poc2.task_id),
    ]
    expected_cpu_graph = TaskGraph()
    expected_cpu_graph.add_tasks([hl1, hl2, hl3, he1, prc1, prc2, poc1, poc2])
    expected_cpu_graph.add_precedences(expected_cpu_precedences)
    expected_cpu_graph.add_ext_precedences(expected_external_cpu_precedences)
    cpu_graph = graph.get_cpu_graph()
    assert cpu_graph == expected_cpu_graph
    # assert set(cpu_graph.precedences) == set(expected_cpu_precedences)
    # assert set(cpu_graph.external_precedences) == set(expected_external_cpu_precedences)
    # assert cpu_graph.rel_deadlines == {}
    # assert cpu_graph.external_rel_deadlines == {}

    # Check QPU graph
    expected_qpu_tasks = {t.task_id: t for t in [mp1, mpc1, lr1]}
    expected_qpu_precedences = [
        (mp1.task_id, mpc1.task_id),
        (mpc1.task_id, lr1.task_id),
    ]
    expected_external_qpu_precedences = [
        (prc1.task_id, mp1.task_id),
        (prc2.task_id, lr1.task_id),
    ]
    qpu_graph = graph.get_qpu_graph()
    assert qpu_graph.tasks == expected_qpu_tasks
    assert set(qpu_graph.precedences) == set(expected_qpu_precedences)
    assert set(qpu_graph.external_precedences) == set(expected_external_qpu_precedences)
    assert qpu_graph.rel_deadlines == {}
    assert qpu_graph.external_rel_deadlines == {}


if __name__ == "__main__":
    linear()
    no_precedence()
    test_get_cpu_graph()
