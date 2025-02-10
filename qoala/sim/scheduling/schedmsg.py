from dataclasses import dataclass

from qoala.runtime.task import ProcessorType


@dataclass
class TaskFinishedMsg:
    processor: ProcessorType
    # Processor type (CPU or QPU) that finished this task

    pid: int
    # PID of the task that finished

    task_id: int
    # Task ID of the task that finished
