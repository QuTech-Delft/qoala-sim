import os

from qoala.lang.parse import IqoalaParser
from qoala.runtime.taskcreator import (
    CpuTask,
    QpuTask,
    RoutineType,
    TaskCreator,
    TaskExecutionMode,
)


def relative_path(path: str) -> str:
    return os.path.join(os.getcwd(), os.path.dirname(__file__), path)


def test1():
    path = relative_path("integration/bqc/vbqc_client.iqoala")
    with open(path) as file:
        text = file.read()
    program = IqoalaParser(text).parse()

    creator = TaskCreator(TaskExecutionMode.ROUTINE_ATOMIC)
    pid = 3
    cpu_tasks, qpu_tasks = creator.from_program(program, pid)

    assert cpu_tasks == [CpuTask(pid, "b0"), CpuTask(pid, "b5")]
    assert qpu_tasks == [
        QpuTask(pid, RoutineType.REQUEST, "b1"),
        QpuTask(pid, RoutineType.LOCAL, "b2"),
        QpuTask(pid, RoutineType.REQUEST, "b3"),
        QpuTask(pid, RoutineType.LOCAL, "b4"),
    ]


if __name__ == "__main__":
    test1()
