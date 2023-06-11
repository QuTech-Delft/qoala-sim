import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.task import TaskExecutionMode, TaskGraphBuilder
from qoala.sim.build import build_network_from_config


@dataclass
class AppResult:
    batch_results: Dict[str, BatchResult]


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


def run_application(
    num_iterations: int,
    programs: Dict[str, QoalaProgram],
    program_inputs: Dict[str, ProgramInput],
    network_cfg: ProcNodeNetworkConfig,
):
    ns.sim_reset()

    network = build_network_from_config(network_cfg)

    names = list(programs.keys())

    for name in names:
        procnode = network.nodes[name]
        program = programs[name]
        inputs = [program_inputs[name] for _ in range(num_iterations)]

        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch = create_batch(program, unit_module, inputs, num_iterations)
        procnode.submit_batch(batch)
        procnode.initialize_processes()

        tasks = procnode.scheduler.get_tasks_to_schedule()
        merged = TaskGraphBuilder.merge(tasks)
        procnode.scheduler.upload_task_graph(merged)

    network.start()
    ns.sim_run()

    results: Dict[str, BatchResult] = {}

    for name in names:
        procnode = network.nodes[name]
        # only one batch (ID = 0), so get value at index 0
        results[name] = procnode.scheduler.get_batch_results()[0]

    return AppResult(results)
