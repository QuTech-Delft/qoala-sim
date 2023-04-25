import pytest

from qoala.runtime.task import QoalaTask, TaskGraph


def linear():
    tasks = {i: QoalaTask(i) for i in range(5)}
    precedences = [(i - 1, i) for i in range(1, 5)]
    relative_deadlines = {i: {i - 1: 100} for i in range(1, 5)}
    graph = TaskGraph(tasks, precedences, relative_deadlines)

    assert graph.roots() == [0]
    assert graph.predecessors(0) == []
    assert all(graph.predecessors(i) == [i - 1] for i in range(1, 5))
    assert graph.deadlines(0) == {}
    assert all(graph.deadlines(i) == {i - 1: 100} for i in range(1, 5))

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.predecessors(0)
    with pytest.raises(AssertionError):
        graph.deadlines(0)

    assert graph.roots() == [1]
    assert graph.predecessors(1) == []
    assert all(graph.predecessors(i) == [i - 1] for i in range(2, 5))
    assert graph.deadlines(1) == {}
    assert all(graph.deadlines(i) == {i - 1: 100} for i in range(2, 5))

    with pytest.raises(AssertionError):
        # not a leaf
        graph.remove_task(4)


def no_precedence():
    tasks = {i: QoalaTask(i) for i in range(5)}
    relative_deadlines = {i: {i - 1: 100} for i in range(1, 5)}
    graph = TaskGraph(tasks, [], relative_deadlines)

    assert graph.roots() == [i for i in range(5)]
    assert all(graph.predecessors(i) == [] for i in range(5))
    assert graph.deadlines(0) == {}
    assert all(graph.deadlines(i) == {i - 1: 100} for i in range(1, 5))

    graph.remove_task(0)

    with pytest.raises(AssertionError):
        graph.predecessors(0)
    with pytest.raises(AssertionError):
        graph.deadlines(0)

    assert graph.roots() == [i for i in range(1, 5)]
    assert all(graph.predecessors(i) == [] for i in range(1, 5))
    assert graph.deadlines(1) == {}
    assert all(graph.deadlines(i) == {i - 1: 100} for i in range(2, 5))

    graph.remove_task(4)

    with pytest.raises(AssertionError):
        graph.predecessors(4)
    with pytest.raises(AssertionError):
        graph.deadlines(4)

    assert graph.roots() == [1, 2, 3]
    assert all(graph.predecessors(i) == [] for i in [1, 2, 3])
    assert all(graph.deadlines(i) == {i - 1: 100} for i in [2, 3])


if __name__ == "__main__":
    linear()
    no_precedence()
