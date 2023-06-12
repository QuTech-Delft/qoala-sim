import os

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.task import TaskGraph, TaskGraphBuilder
from qoala.util.taskgraph import TaskGraphWriter


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def test1():
    program = load_program("test_callbacks_2_pairs.iqoala")
    # graph = TaskGraphBuilder.from_file(program, 0)
    graph = TaskGraphBuilder.from_file_block_tasks(program, 0)

    TaskGraphWriter(graph).draw("graph2.png")


if __name__ == "__main__":
    test1()
