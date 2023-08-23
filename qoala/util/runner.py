import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import ProcNodeNetworkConfig  # type: ignore
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.sim.build import build_network_from_config

# from qoala.util.logging import LogManager


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
        procnode.initialize_processes(remote_pids, linear=linear)

        # logger = LogManager.get_stack_logger()
        # for batch_id, prog_batch in procnode.scheduler.get_batches().items():
        #     task_graph = prog_batch.instances[0].task_graph
        #     num = len(prog_batch.instances)
        #     logger.info(f"batch {batch_id}: {num} instances each with task graph:")
        #     logger.info(task_graph)

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
        procnode.initialize_processes({0: [server_pids[client_name]]}, linear=False)

    server_procnode.initialize_processes(
        {i: [0] for i in range(len(client_names))}, linear=False
    )

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

    procnode.initialize_processes(linear=linear)

    # logger = LogManager.get_stack_logger()
    # for batch_id, prog_batch in procnode.scheduler.get_batches().items():
    #     task_graph = prog_batch.instances[0].task_graph
    #     num = len(prog_batch.instances)
    #     logger.info(f"batch {batch_id}: {num} instances each with task graph:")
    #     logger.info(task_graph)

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
