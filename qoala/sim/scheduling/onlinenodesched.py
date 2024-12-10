from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import netsquid as ns

from pydynaa import EventExpression
from qoala.lang.ehi import EhiNetworkInfo, EhiNodeInfo
from qoala.lang.hostlang import BasicBlockType
from qoala.runtime.message import Message
from qoala.runtime.program import ProgramInstance
from qoala.runtime.task import ProcessorType, TaskInfo
from qoala.runtime.taskbuilder import TaskGraphFromBlockBuilder
from qoala.sim.events import SIGNAL_TASK_COMPLETED
from qoala.sim.host.host import Host
from qoala.sim.memmgr import MemoryManager
from qoala.sim.netstack import Netstack
from qoala.sim.qnos import Qnos
from qoala.sim.scheduling.nodesched import NodeScheduler


class OnlineNodeScheduler(NodeScheduler):
    """TODO docstrings"""

    def __init__(
        self,
        node_name: str,
        host: Host,
        qnos: Qnos,
        netstack: Netstack,
        memmgr: MemoryManager,
        local_ehi: EhiNodeInfo,
        network_ehi: EhiNetworkInfo,
        deterministic: bool = True,
        use_deadlines: bool = True,
        fcfs: bool = False,
        prio_epr: bool = False,
    ) -> None:
        super().__init__(
            node_name=node_name,
            host=host,
            qnos=qnos,
            netstack=netstack,
            memmgr=memmgr,
            local_ehi=local_ehi,
            network_ehi=network_ehi,
            deterministic=deterministic,
            use_deadlines=use_deadlines,
            fcfs=fcfs,
            prio_epr=prio_epr,
        )

        # For each program instance, keep track of which block is currently being
        # executed or which one is to be executed next (if the prev block just finished)
        self._curr_blk_idx: Dict[int, int] = {}  # program ID -> block index
        self._task_from_block_builder = TaskGraphFromBlockBuilder()
        self._prog_instance_dependency: Dict[
            int, int
        ] = {}  # program ID -> dependent program ID

    def create_processes_for_batches(
        self,
        remote_pids: Optional[Dict[int, List[int]]] = None,  # batch ID -> PID list
        linear: bool = False,
    ) -> None:
        prev_prog_instance_id = -1
        for batch_id, batch in self._batches.items():
            for i, prog_instance in enumerate(batch.instances):
                if remote_pids is not None and batch_id in remote_pids:
                    remote_pid = remote_pids[batch_id][i]
                else:
                    remote_pid = None
                process = self.create_process(prog_instance, remote_pid)

                self.memmgr.add_process(process)
                self.initialize_process(process)
                self._curr_blk_idx[prog_instance.pid] = 0
                if linear:
                    self._prog_instance_dependency[
                        prog_instance.pid
                    ] = prev_prog_instance_id
                    prev_prog_instance_id = prog_instance.pid
                else:
                    self._prog_instance_dependency[prog_instance.pid] = -1

        if self._const_batch is not None:
            for i, prog_instance in enumerate(self._const_batch.instances):
                process = self.create_process(prog_instance)
                self.memmgr.add_process(process)
                self.initialize_process(process)

    def submit_program_instance(
        self, prog_instance: ProgramInstance, remote_pid: Optional[int] = None
    ) -> None:
        process = self.create_process(prog_instance, remote_pid)
        self.memmgr.add_process(process)
        self.initialize_process(process)
        self._curr_blk_idx[prog_instance.pid] = 0
        self._prog_instance_dependency[prog_instance.pid] = -1

    def start(self) -> None:
        # Processor schedulers start first to ensure that they will start running tasks after they receive the first
        # message from the node scheduler.
        self._cpu_scheduler.start()
        self._qpu_scheduler.start()
        super().start()
        self.schedule_all()

    def stop(self) -> None:
        self._qpu_scheduler.stop()
        self._cpu_scheduler.stop()
        super().stop()

    def run(self) -> Generator[EventExpression, None, None]:
        while True:
            self._logger.debug("main node scheduler loop")

            ev_expr = self.await_signal(self._cpu_scheduler, SIGNAL_TASK_COMPLETED)
            ev_expr = ev_expr | self.await_signal(
                self._qpu_scheduler, SIGNAL_TASK_COMPLETED
            )
            yield ev_expr
            self._logger.debug("got a TASK COMPLETED signal")

            now = ns.sim_time()

            # Gets the pid of the most recently finished task.
            last_cpu_task_pid = self.cpu_scheduler.get_last_finished_task_pid_at(now)
            last_qpu_task_pid = self.qpu_scheduler.get_last_finished_task_pid_at(now)

            # One of the ProcSchedulers must have just finished a task.
            # (It could happen, although rare, that both CPU and QPU schedulers just
            # finished a task at the same time. However, in that case, the 2nd task
            # fired its own TASK COMPLETED signal which will be handled in the next
            # iteration of this main loop.)
            assert last_cpu_task_pid != -1 or last_qpu_task_pid != -1

            # Get the PID of the last recent task.
            pid = last_cpu_task_pid if last_cpu_task_pid != -1 else last_qpu_task_pid

            is_const = self.is_from_const_batch(pid)
            is_finished = self.is_prog_inst_finished(pid)

            if not is_const:
                self.schedule_next_for(pid)

            if is_const or is_finished:
                self.schedule_all()

    def schedule_next_for(self, pid: int) -> None:
        """
        Schedule the tasks of the next block for program instance with given pid
        by assigning respective tasks to CPU and QPU schedulers and send a message
        to schedulers for informing them about the newly assigned tasks.

        :param pid: program instance id
        :return: None
        """
        self._logger.debug("schedule_next_for()")
        new_cpu_tasks, new_qpu_tasks = self.find_new_tasks_for(pid)

        # If there are new tasks, send a message to schedulers
        # Note that find_new_tasks_for() returns None if there are no new tasks for that processor
        if new_cpu_tasks:
            self._logger.debug(
                f"schedule_next_for: adding new cpu tasks: {new_cpu_tasks}"
            )
            self._cpu_scheduler.add_tasks(new_cpu_tasks)
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
        if new_qpu_tasks:
            self._logger.debug(
                f"schedule_next_for: adding new qpu tasks: {new_qpu_tasks}"
            )
            self._qpu_scheduler.add_tasks(new_qpu_tasks)
            self._task_logger.debug("sending 'New Task' msg to QPU scheduler")
            self._comp.send_qpu_scheduler_message(Message(-1, -1, "New Task"))

    def schedule_all(self) -> None:
        """
        Schedules the tasks of the next block for each available program instance in the memory manager
        by assigning respective tasks to CPU and QPU schedulers and sends a message
        to schedulers for informing them about the newly assigned tasks.

        This method is responsible for scheduling tasks for each available program instance in the memory manager.
        A program instance is considered available if it meets two conditions:
        1. It is not finished.
        2. It does not have any dependencies on an unfinished program instance
        (it can have such dependencies if the batch of program instances are submitted to run linearly).

        :return: None
        """
        self._logger.debug("schedule_all()")

        all_new_cpu_tasks: Dict[int, TaskInfo] = {}
        all_new_qpu_tasks: Dict[int, TaskInfo] = {}

        # for pid in self.memmgr.get_all_program_ids():
        for pid in self.get_all_non_const_pids():
            # If there is a dependency, check if it is finished
            dependency_pid = self._prog_instance_dependency[pid]
            if dependency_pid != -1:
                dep_cur_index = self._curr_blk_idx[dependency_pid]
                dep_block_length = len(
                    self.memmgr.get_process(dependency_pid).prog_instance.program.blocks
                )
                if dep_cur_index < dep_block_length:
                    continue

            # Note that find_new_tasks_for() returns None if there are no new tasks for that processor
            new_cpu_tasks, new_qpu_tasks = self.find_new_tasks_for(pid)
            if new_cpu_tasks:
                all_new_cpu_tasks.update(new_cpu_tasks)
            if new_qpu_tasks:
                all_new_qpu_tasks.update(new_qpu_tasks)

        # If there are new tasks, send a message to schedulers
        if len(all_new_cpu_tasks) > 0:
            self._logger.debug(f"schedule_all: adding new cpu tasks: {new_cpu_tasks}")
            self._cpu_scheduler.add_tasks(all_new_cpu_tasks)
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
        if len(all_new_qpu_tasks) > 0:
            self._logger.debug(f"schedule_all: adding new qpu tasks: {new_qpu_tasks}")
            self._qpu_scheduler.add_tasks(all_new_qpu_tasks)
            self._task_logger.debug("sending 'New Task' msg to QPU scheduler")
            self._comp.send_qpu_scheduler_message(Message(-1, -1, "New Task"))

    def find_new_tasks_for(
        self, pid: int
    ) -> Tuple[Optional[Dict[int, TaskInfo]], Optional[Dict[int, TaskInfo]]]:
        """
        Find new tasks, for a specific program instance (with the given PID),
        to add to the CPU task graph and/or QPU task graph.
        If the CPU or QPU task graph still contains tasks for this program instance,
        no new tasks are returned. This is because it means that program instance has
        not fully completed the current block.

        :param pid: The program instance ID for which to find the next tasks.
        :return: A 2-tuple containing the new CPU tasks and new QPU tasks. If no tasks are
             found for the given PID, both elements of the tuple will be set to None.
        """

        # Check if this program instance just executed a jump, in which case we need to
        # update which block is the next one to execute.
        if (
            pid in self.host.interface.program_instance_jumps
            and self.host.interface.program_instance_jumps[pid] != -1
        ):
            self._curr_blk_idx[pid] = self.host.interface.program_instance_jumps[pid]
            self.host.interface.program_instance_jumps[pid] = -1

        # Find the block we're currently executing or (in case we just finished one)
        # the block which is the next one to execute.
        current_block_index = self._curr_blk_idx[pid]
        prog_instance = self.memmgr.get_process(pid).prog_instance
        blocks = prog_instance.program.blocks

        # If the program has finished, return no tasks.
        if current_block_index >= len(blocks):
            return None, None

        # If the CPU or QPU scheduler still has tasks for this pid, it means that
        # execution of the current block has not yet finished, and hence we don't
        # return any new tasks to add.
        cpu_tasks_for_pid: bool = self.cpu_scheduler.task_exists_for_pid(pid)
        qpu_tasks_for_pid: bool = self.qpu_scheduler.task_exists_for_pid(pid)
        if cpu_tasks_for_pid or qpu_tasks_for_pid:
            return None, None

        # If we arrive here, it means that we should find new tasks to be added
        # (to CPU and/or QPU scheduler), based on the next block to execute.
        block = prog_instance.program.blocks[current_block_index]

        # If it is CL or CC it will only have tasks in CPU but others will have tasks in both
        if block.typ in {BasicBlockType.CL, BasicBlockType.CC}:
            graph = self._task_from_block_builder.build(
                prog_instance, current_block_index, self._network_ehi
            )
            cpu_graph = graph.get_tasks()

            # TODO: fix this; not quite correct/intuitive
            self._curr_blk_idx[pid] += 1

            # If scheduler does not have any task send a message to wake it up
            self._comp.send_cpu_scheduler_message(Message(-1, -1, "New Task"))
            return cpu_graph, None
        else:  # block.typ in {QL, QC}
            graph = self._task_from_block_builder.build(
                prog_instance, current_block_index, self._network_ehi
            )
            cpu_graph = graph.partial_graph(ProcessorType.CPU).get_tasks()
            qpu_graph = graph.partial_graph(ProcessorType.QPU).get_tasks()
            self._task_logger.debug(f"adding CPU tasks {cpu_graph}")
            self._task_logger.debug(f"adding QPU tasks {qpu_graph}")

            self._curr_blk_idx[pid] += 1

            return cpu_graph, qpu_graph

    def is_prog_inst_finished(self, pid: int) -> bool:
        return self._curr_blk_idx[pid] >= len(
            self.memmgr.get_process(pid).prog_instance.program.blocks
        )
