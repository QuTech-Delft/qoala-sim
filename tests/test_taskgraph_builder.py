from typing import List, Optional, Tuple

from qoala.lang.hostlang import BasicBlockType
from qoala.runtime.task import BlockTask, TaskGraphBuilder

CL = BasicBlockType.CL
CC = BasicBlockType.CC
QL = BasicBlockType.QL
QC = BasicBlockType.QC


def test_linear_block_tasks():
    pid = 0
    tasks = [
        BlockTask(0, pid, "blk_host0", CL, 1000),
        BlockTask(1, pid, "blk_recv", CC, 1000),
        BlockTask(2, pid, "blk_add_one", QL, 1000),
        BlockTask(3, pid, "blk_epr_md_1", QC, 1000),
        BlockTask(4, pid, "blk_host1", CL, 1000),
    ]

    graph = TaskGraphBuilder.linear_block_tasks(tasks)
    assert graph.get_tinfo(0).task == tasks[0]
    assert graph.get_tinfo(0).predecessors == []
    for i in range(1, len(tasks)):
        assert graph.get_tinfo(i).task == tasks[i]
        assert graph.get_tinfo(i).predecessors == [i - 1]


def test_linear_block_tasks_with_timestamps():
    pid = 0
    tasks = [
        BlockTask(0, pid, "blk_host0", CL, 1000),
        BlockTask(1, pid, "blk_recv", CC, 1000),
        BlockTask(2, pid, "blk_add_one", QL, 1000),
        BlockTask(3, pid, "blk_epr_md_1", QC, 1000),
        BlockTask(4, pid, "blk_host1", CL, 1000),
    ]

    start_times: List[Tuple[BlockTask, Optional[int]]] = [
        (tasks[0], 0),
        (tasks[1], 2000),
        (tasks[2], 3000),
        (tasks[3], 12500),
        (tasks[4], None),
    ]

    graph = TaskGraphBuilder.linear_block_tasks_with_start_times(start_times)
    assert graph.get_tinfo(0).task == tasks[0]
    assert graph.get_tinfo(0).predecessors == []
    for i in range(1, len(tasks)):
        assert graph.get_tinfo(i).task == tasks[i]
        assert graph.get_tinfo(i).predecessors == [i - 1]

    for task, start_time in start_times:
        assert graph.get_tinfo(task.task_id).start_time == start_time


if __name__ == "__main__":
    test_linear_block_tasks()
    test_linear_block_tasks_with_timestamps()
