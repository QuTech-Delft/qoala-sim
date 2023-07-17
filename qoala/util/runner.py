import gc
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import ProcNodeNetworkConfig  # type: ignore
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.runtime.task import QoalaTask, TaskGraphBuilder
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager


class SchedulerType(Enum):
    NO_SCHED = 0
    FCFS = auto()
    QOALA = auto()


@dataclass
class AppResult:
    batch_results: Dict[str, BatchResult]
    statistics: Dict[str, SchedulerStatistics]
    total_duration: float


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def create_batch(
    program: QoalaProgram,
    unit_module: UnitModule,
    inputs: List[ProgramInput],
    num_iterations: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        unit_module=unit_module,
        inputs=inputs,
        num_iterations=num_iterations,
        deadline=0,
    )


def run_two_node_app_separate_inputs(
    num_iterations: int,
    programs: Dict[str, QoalaProgram],
    program_inputs: Dict[str, List[ProgramInput]],
    network_cfg: ProcNodeNetworkConfig,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    names = list(programs.keys())
    assert len(names) == 2
    other_name = {names[0]: names[1], names[1]: names[0]}
    batches: Dict[str, ProgramBatch] = {}  # node -> batch

    for name in names:
        procnode = network.nodes[name]
        program = programs[name]
        inputs = program_inputs[name]

        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch_info = create_batch(program, unit_module, inputs, num_iterations)
        batches[name] = procnode.submit_batch(batch_info)

    for name in names:
        procnode = network.nodes[name]

        remote_batch = batches[other_name[name]]
        remote_pids = {remote_batch.batch_id: [p.pid for p in remote_batch.instances]}
        procnode.initialize_processes(remote_pids)

        tasks = procnode.scheduler.get_tasks_to_schedule()
        if linear:
            merged = TaskGraphBuilder.merge_linear(tasks)
        else:
            merged = TaskGraphBuilder.merge(tasks)
        procnode.scheduler.upload_task_graph(merged)

        logger = LogManager.get_stack_logger()
        for batch_id, prog_batch in procnode.scheduler.get_batches().items():
            task_graph = prog_batch.instances[0].task_graph
            num = len(prog_batch.instances)
            logger.info(f"batch {batch_id}: {num} instances each with task graph:")
            logger.info(task_graph)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in names:
        procnode = network.nodes[name]
        # only one batch (ID = 0), so get value at index 0
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    total_duration = ns.sim_time()
    return AppResult(results, statistics, total_duration)


def run_two_node_app_separate_inputs_plus_constant_tasks(
    num_iterations: int,
    num_const_tasks: int,
    node1: str,
    node2: str,
    prog_node1: QoalaProgram,
    prog_node1_inputs: Dict[str, List[ProgramInput]],
    prog_node2: QoalaProgram,
    prog_node2_inputs: Dict[str, List[ProgramInput]],
    const_prog_node2: QoalaProgram,
    const_prog_node2_inputs: Dict[str, List[ProgramInput]],
    const_period: int,
    const_start: int,
    network_cfg: ProcNodeNetworkConfig,
    sched_typ: SchedulerType,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    procnode1 = network.nodes[node1]
    unit_module1 = UnitModule.from_full_ehi(procnode1.memmgr.get_ehi())
    batch_info1 = create_batch(
        prog_node1, unit_module1, prog_node1_inputs, num_iterations
    )
    batch_node1 = procnode1.submit_batch(batch_info1)

    procnode2 = network.nodes[node2]
    unit_module2 = UnitModule.from_full_ehi(procnode2.memmgr.get_ehi())
    batch_info2 = create_batch(
        prog_node2, unit_module2, prog_node2_inputs, num_iterations
    )
    batch_node2 = procnode2.submit_batch(batch_info2)

    batch_info_const = create_batch(
        const_prog_node2, unit_module2, const_prog_node2_inputs, num_const_tasks
    )
    batch_const = procnode2.submit_batch(batch_info_const)

    # Init

    remote_pids1 = {batch_node2.batch_id: [p.pid for p in batch_node2.instances]}
    procnode1.initialize_processes(remote_pids1)

    remote_pids2 = {batch_node1.batch_id: [p.pid for p in batch_node1.instances]}
    procnode2.initialize_processes(remote_pids2)

    # Upload tasks

    tasks1 = procnode1.scheduler.get_tasks_to_schedule()
    if linear:
        merged1 = TaskGraphBuilder.merge_linear(tasks1)
    else:
        merged1 = TaskGraphBuilder.merge(tasks1)
    procnode1.scheduler.upload_task_graph(merged1)

    tasks2 = procnode2.scheduler.get_tasks_to_schedule_for(batch_node2.batch_id)
    if linear:
        merged2 = TaskGraphBuilder.merge_linear(tasks2)
    else:
        merged2 = TaskGraphBuilder.merge(tasks2)

    tasks_const = procnode2.scheduler.get_tasks_to_schedule_for(batch_const.batch_id)
    merged_const = TaskGraphBuilder.merge_linear(tasks_const)
    start = const_start
    for tid, tinfo in merged_const.get_tasks().items():
        merged_const.get_tinfo(tid).start_time = start
        if sched_typ == SchedulerType.FCFS:
            # Force the busy tasks to start earlier by setting their start times as deadlines
            merged_const.get_tinfo(tid).deadline = start
        start += const_period
    # for tid, tinfo in merged_const.get_tasks().items():
    #     print(f"tid: {tid}, tinfo: {tinfo}")
    if sched_typ == SchedulerType.NO_SCHED:
        merged_with_const = TaskGraphBuilder.merge_linear([merged2, merged_const])
    else:
        merged_with_const = TaskGraphBuilder.merge([merged2, merged_const])

    procnode2.scheduler.upload_task_graph(merged_with_const)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    # only one batch (ID = 0), so get value at index 0
    results[node1] = procnode1.scheduler.get_batch_results()[0]
    results[node2] = procnode2.scheduler.get_batch_results()[0]
    statistics[node1] = procnode1.scheduler.get_statistics()
    statistics[node2] = procnode2.scheduler.get_statistics()

    total_duration = ns.sim_time()

    return AppResult(results, statistics, total_duration)


def run_1_server_n_clients(
    client_names: List[str],
    client_program: QoalaProgram,
    server_name: str,
    server_program: QoalaProgram,
    client_inputs: Dict[str, List[ProgramInput]],
    server_inputs: List[ProgramInput],
    network_cfg: ProcNodeNetworkConfig,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    server_pids: Dict[str, int] = {}  # client name -> server PID

    for client_name in client_names:
        procnode = network.nodes[client_name]
        inputs = client_inputs[client_name]
        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch_info = create_batch(client_program, unit_module, inputs, 1)
        procnode.submit_batch(batch_info)

        server_procnode = network.nodes[server_name]
        server_unit_module = UnitModule.from_full_ehi(server_procnode.memmgr.get_ehi())
        program = deepcopy(server_program)
        program.meta.csockets[0] = client_name
        program.meta.epr_sockets[0] = client_name
        server_batch_info = create_batch(program, server_unit_module, server_inputs, 1)
        batch = server_procnode.submit_batch(server_batch_info)
        server_pids[client_name] = batch.instances[0].pid

    for client_name in client_names:
        procnode.initialize_processes({0: [server_pids[client_name]]})
        tasks = procnode.scheduler.get_tasks_to_schedule()
        merged = TaskGraphBuilder.merge(tasks)
        procnode.scheduler.upload_task_graph(merged)

    server_procnode.initialize_processes({i: [0] for i in range(len(client_names))})
    server_tasks = server_procnode.scheduler.get_tasks_to_schedule()
    server_merged = TaskGraphBuilder.merge(server_tasks)
    server_procnode.scheduler.upload_task_graph(server_merged)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in client_names:
        procnode = network.nodes[name]
        # only one batch (ID = 0), so get value at index 0
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    server_procnode = network.nodes[server_name]
    results[server_name] = server_procnode.scheduler.get_batch_results()[0]
    statistics[server_name] = server_procnode.scheduler.get_statistics()

    total_duration = ns.sim_time()
    return AppResult(results, statistics, total_duration)


def run_two_node_app(
    num_iterations: int,
    programs: Dict[str, QoalaProgram],
    program_inputs: Dict[str, ProgramInput],
    network_cfg: ProcNodeNetworkConfig,
    linear: bool = False,
) -> AppResult:
    new_inputs: Dict[str, List[ProgramInput]] = {}

    names = list(programs.keys())
    new_inputs = {
        name: [program_inputs[name] for _ in range(num_iterations)] for name in names
    }

    return run_two_node_app_separate_inputs(
        num_iterations, programs, new_inputs, network_cfg, linear
    )


def run_single_node_app_separate_inputs(
    num_iterations: int,
    program_name: str,
    program: QoalaProgram,
    program_input: List[ProgramInput],
    network_cfg: ProcNodeNetworkConfig,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    procnode = list(network.nodes.values())[0]

    unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
    batch_info = create_batch(program, unit_module, program_input, num_iterations)
    procnode.submit_batch(batch_info)

    procnode.initialize_processes()

    tasks = procnode.scheduler.get_tasks_to_schedule()
    if linear:
        merged = TaskGraphBuilder.merge_linear(tasks)
    else:
        merged = TaskGraphBuilder.merge(tasks)
    procnode.scheduler.upload_task_graph(merged)

    logger = LogManager.get_stack_logger()
    for batch_id, prog_batch in procnode.scheduler.get_batches().items():
        task_graph = prog_batch.instances[0].task_graph
        num = len(prog_batch.instances)
        logger.info(f"batch {batch_id}: {num} instances each with task graph:")
        logger.info(task_graph)

    network.start()
    ns.sim_run()

    # only one batch (ID = 0), so get value at index 0
    results = procnode.scheduler.get_batch_results()[0]
    statistics = procnode.scheduler.get_statistics()
    total_duration = ns.sim_time()

    return AppResult(
        {program_name: results}, {program_name: statistics}, total_duration
    )


def run_single_node_app(
    num_iterations: int,
    program_name: str,
    program: QoalaProgram,
    program_input: ProgramInput,
    network_cfg: ProcNodeNetworkConfig,
    linear: bool = False,
) -> AppResult:
    new_inputs = [program_input for _ in range(num_iterations)]

    return run_single_node_app_separate_inputs(
        num_iterations, program_name, program, new_inputs, network_cfg, linear
    )
