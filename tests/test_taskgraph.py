from typing import List, Optional, Tuple

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
    TaskGraphBuilder,
)


class SimpleTask(QoalaTask):
    def __init__(self, task_id: int) -> None:
        super().__init__(task_id, ProcessorType.CPU, 0)


def test_linear():
    tasks = [SimpleTask(i) for i in range(5)]
    precedences = [(i - 1, i) for i in range(1, 5)]
    rel_deadlines = [((i - 1, i), 100) for i in range(1, 5)]
    graph = TaskGraph()
    graph.add_tasks(tasks)
    graph.add_precedences(precedences)
    graph.add_rel_deadlines(rel_deadlines)

    assert graph.get_roots() == [0]
    assert graph.get_tinfo(0).predecessors == set()
    assert graph.get_tinfo(4).successors == set()
    assert all(graph.get_tinfo(i).predecessors == {i - 1} for i in range(1, 5))
    assert all(graph.get_tinfo(i).successors == {i + 1} for i in range(0, 4))
    assert all(graph.get_tinfo(i).deadline is None for i in range(5))
    assert graph.get_tinfo(0).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(1, 5))
    assert len(graph.get_tasks()) == 5

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.get_tinfo(0)

    assert graph.get_roots() == [1]
    assert graph.get_tinfo(1).predecessors == set()
    assert graph.get_tinfo(4).successors == set()
    assert all(graph.get_tinfo(i).predecessors == {i - 1} for i in range(2, 5))
    assert all(graph.get_tinfo(i).successors == {i + 1} for i in range(1, 4))
    assert graph.get_tinfo(1).deadline == 100
    assert all(graph.get_tinfo(i).deadline is None for i in range(2, 5))
    assert graph.get_tinfo(1).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 5))
    assert len(graph.get_tasks()) == 4

    with pytest.raises(AssertionError):
        # not a root
        graph.remove_task(4)


def test_no_precedence():
    tasks = [SimpleTask(i) for i in range(5)]
    rel_deadlines = [((i - 1, i), 100) for i in range(1, 5)]
    graph = TaskGraph()
    graph.add_tasks(tasks)
    graph.add_rel_deadlines(rel_deadlines)

    assert graph.get_roots() == [i for i in range(5)]
    assert all(graph.get_tinfo(i).predecessors == set() for i in range(5))
    assert all(graph.get_tinfo(i).successors == set() for i in range(5))
    assert all(graph.get_tinfo(i).deadline is None for i in range(5))
    assert graph.get_tinfo(0).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(1, 5))

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.get_tinfo(0)

    assert graph.get_roots() == [i for i in range(1, 5)]
    assert all(graph.get_tinfo(i).predecessors == set() for i in range(1, 5))
    assert all(graph.get_tinfo(i).successors == set() for i in range(1, 5))
    assert graph.get_tinfo(1).deadline == 100
    assert all(graph.get_tinfo(i).deadline is None for i in range(2, 5))
    assert graph.get_tinfo(1).rel_deadlines == {}
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 5))

    graph.remove_task(4)

    with pytest.raises(AssertionError):
        graph.get_tinfo(4)

    assert graph.get_roots() == [1, 2, 3]
    assert all(graph.get_tinfo(i).predecessors == set() for i in range(1, 4))
    assert all(graph.get_tinfo(i).rel_deadlines == {i - 1: 100} for i in range(2, 4))


def test_get_partial_graph():
    pid = 0
    mp_ptr = 0
    lr_ptr = 1

    hl1 = HostLocalTask(0, pid, "hl1")
    hl2 = HostLocalTask(1, pid, "hl2")
    hl3 = HostLocalTask(2, pid, "hl2")
    he1 = HostEventTask(3, pid, "he1")
    prc1 = PreCallTask(4, pid, "prc1", mp_ptr)
    prc2 = PreCallTask(5, pid, "prc2", lr_ptr)
    poc1 = PreCallTask(6, pid, "poc1", mp_ptr)
    poc2 = PreCallTask(7, pid, "poc2", lr_ptr)
    mp1 = MultiPairTask(8, pid, mp_ptr)
    mpc1 = MultiPairCallbackTask(9, pid, "mpc1", mp_ptr)
    lr1 = LocalRoutineTask(10, pid, "lr1", lr_ptr)

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

    # Check QPU graph
    expected_qpu_precedences = [
        (mp1.task_id, mpc1.task_id),
        (mpc1.task_id, lr1.task_id),
    ]
    expected_external_qpu_precedences = [
        (prc1.task_id, mp1.task_id),
        (prc2.task_id, lr1.task_id),
    ]
    qpu_graph = graph.get_qpu_graph()
    expected_qpu_graph = TaskGraph()
    expected_qpu_graph.add_tasks([mp1, mpc1, lr1])
    expected_qpu_graph.add_precedences(expected_qpu_precedences)
    expected_qpu_graph.add_ext_precedences(expected_external_qpu_precedences)
    assert qpu_graph == expected_qpu_graph


def test_dynamic_update():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0), SimpleTask(1)])

    # task 0 should start at <2000 from now
    graph.add_deadlines([(0, 2000)])
    # task 1 should start <100 after task 1 finishes
    graph.add_rel_deadlines([((0, 1), 100)])

    # Mock execution of task 0, taking 500 time units.
    # First decrease all current absolute deadlines since removing task 0 will make
    # the relative deadline of task 1 an absolute deadline, which we do not want to decrease.
    graph.decrease_deadlines(500)
    graph.remove_task(0)
    assert graph.get_tinfo(1).deadline == 100


def test_linear_tasks():
    tasks = [SimpleTask(0), SimpleTask(1), SimpleTask(2), SimpleTask(3), SimpleTask(4)]

    graph = TaskGraphBuilder.linear_tasks(tasks)
    assert graph.get_tinfo(0).task == tasks[0]
    assert graph.get_tinfo(0).predecessors == set()
    assert graph.get_tinfo(len(tasks) - 1).successors == set()
    for i in range(len(tasks)):
        assert graph.get_tinfo(i).task == tasks[i]
        if i > 0:
            assert graph.get_tinfo(i).predecessors == {i - 1}
        if i < len(tasks) - 1:
            assert graph.get_tinfo(i).successors == {i + 1}


def test_linear_tasks_with_timestamps():
    tasks = [SimpleTask(0), SimpleTask(1), SimpleTask(2), SimpleTask(3), SimpleTask(4)]

    start_times: List[Tuple[SimpleTask, Optional[int]]] = [
        (tasks[0], 0),
        (tasks[1], 2000),
        (tasks[2], 3000),
        (tasks[3], 12500),
        (tasks[4], None),
    ]

    graph = TaskGraphBuilder.linear_tasks_with_start_times(start_times)
    assert graph.get_tinfo(0).task == tasks[0]
    assert graph.get_tinfo(0).predecessors == set()
    assert graph.get_tinfo(len(tasks) - 1).successors == set()
    for i in range(len(tasks)):
        assert graph.get_tinfo(i).task == tasks[i]
        if i > 0:
            assert graph.get_tinfo(i).predecessors == {i - 1}
        if i < len(tasks) - 1:
            assert graph.get_tinfo(i).successors == {i + 1}

    for task, start_time in start_times:
        assert graph.get_tinfo(task.task_id).start_time == start_time


def test_merge():
    graph1 = TaskGraph()
    graph1.add_tasks([SimpleTask(0), SimpleTask(1)])
    graph1.add_precedences([(0, 1)])

    graph2 = TaskGraph()
    graph2.add_tasks([SimpleTask(2), SimpleTask(3)])
    graph2.add_precedences([(2, 3)])

    merged = TaskGraphBuilder.merge([graph1, graph2])
    for i in range(4):
        assert merged.get_tinfo(i).task == SimpleTask(i)

    assert merged.get_tinfo(1).predecessors == {0}
    assert merged.get_tinfo(0).successors == {1}
    assert merged.get_tinfo(3).predecessors == {2}
    assert merged.get_tinfo(2).successors == {3}


def test_merge_linear():
    graph1 = TaskGraph()
    graph1.add_tasks([SimpleTask(0), SimpleTask(1)])
    graph1.add_precedences([(0, 1)])

    graph2 = TaskGraph()
    graph2.add_tasks([SimpleTask(2), SimpleTask(3)])
    graph2.add_precedences([(2, 3)])

    merged = TaskGraphBuilder.merge_linear([graph1, graph2])
    for i in range(4):
        assert merged.get_tinfo(i).task == SimpleTask(i)

    assert merged.get_tinfo(1).predecessors == {0}
    assert merged.get_tinfo(0).successors == {1}
    assert merged.get_tinfo(3).predecessors == {2}
    assert merged.get_tinfo(2).successors == {3}

    # Check that there is precedence between two original graphs
    assert merged.get_tinfo(2).predecessors == {1}
    assert merged.get_tinfo(1).successors == {2}


def test_linearize_1():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0), SimpleTask(1)])
    with pytest.raises(RuntimeError):
        graph.linearize()


def test_linearize_2():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0), SimpleTask(1)])
    graph.add_precedences([(0, 1)])
    assert graph.linearize() == [0, 1]


def test_linearize_3():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0), SimpleTask(1), SimpleTask(2)])
    graph.add_precedences([(0, 1)])
    graph.add_precedences([(0, 2)])
    with pytest.raises(RuntimeError):
        graph.linearize()


def test_linearize_4():
    graph = TaskGraph()
    graph.add_tasks([SimpleTask(0), SimpleTask(1), SimpleTask(2)])
    graph.add_precedences([(1, 2)])
    graph.add_precedences([(2, 0)])
    assert graph.linearize() == [1, 2, 0]


if __name__ == "__main__":
    test_linear()
    test_no_precedence()
    test_get_partial_graph()
    test_dynamic_update()
    test_linear_tasks()
    test_linear_tasks_with_timestamps()
    test_merge()
    test_merge_linear()
    test_linearize_1()
    test_linearize_2()
    test_linearize_3()
    test_linearize_4()
