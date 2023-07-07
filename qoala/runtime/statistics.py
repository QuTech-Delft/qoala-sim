from typing import Dict, List

from qoala.runtime.task import QoalaTask


class SchedulerStatistics:
    def __init__(
        self,
        cpu_tasks_executed: Dict[int, QoalaTask],
        qpu_tasks_executed: Dict[int, QoalaTask],
        cpu_task_starts: Dict[int, float],
        qpu_task_starts: Dict[int, float],
        cpu_task_ends: Dict[int, float],
        qpu_task_ends: Dict[int, float],
    ) -> None:
        # task ID -> task
        self._cpu_tasks_executed = cpu_tasks_executed
        self._qpu_tasks_executed = qpu_tasks_executed

        # task ID -> start time
        self._cpu_task_starts = cpu_task_starts
        self._qpu_task_starts = qpu_task_starts

        # task ID -> end time
        self._cpu_task_ends: Dict[int, float] = cpu_task_ends
        self._qpu_task_ends: Dict[int, float] = qpu_task_ends

        cpu_pids = set([t.pid for t in self._cpu_tasks_executed.values()])
        qpu_pids = set([t.pid for t in self._qpu_tasks_executed.values()])
        pids = cpu_pids.union(qpu_pids)
        self._pids = list(pids)

        # for pid in self._pids

    @property
    def pids(self) -> List[int]:
        return self._pids

    @property
    def num_tasks_executed(self) -> int:
        return self.num_cpu_tasks_executed + self.num_qpu_tasks_executed

    @property
    def num_cpu_tasks_executed(self) -> int:
        return len(self._cpu_tasks_executed)

    @property
    def num_qpu_tasks_executed(self) -> int:
        return len(self._qpu_tasks_executed)

    def __str__(self) -> str:
        return (
            f"# tasks executed: {self.num_tasks_executed} "
            f"(CPU: {self.num_cpu_tasks_executed}, "
            f"QPU: {self.num_qpu_tasks_executed})"
            f"\npids: {self.pids}"
        )
