from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import netsquid as ns
from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramBatch, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.runtime.task import QoalaGraphFromProgramBuilder, TaskGraphBuilder
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager
from qoala.util.runner import (
    AppResult,
    create_batch,
    run_two_node_app_separate_inputs_plus_constant_tasks,
)


def run_two_node_app_separate_inputs_plus_3rd_program(
    num_iterations: int,
    programs: Dict[str, QoalaProgram],
    program_inputs: Dict[str, List[ProgramInput]],
    third_program: QoalaProgram,
    third_program_input: ProgramInput,
    network_cfg: ProcNodeNetworkConfig,
    hog_prob: float,
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

    # 3rd program goes on second node
    procnode = network.nodes[names[1]]
    unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
    inputs = [third_program_input]
    batch_info = create_batch(third_program, unit_module, inputs, 1)
    procnode.submit_batch(batch_info)

    for name in names:
        procnode = network.nodes[name]

        remote_batch = batches[other_name[name]]
        remote_pids = {remote_batch.batch_id: [p.pid for p in remote_batch.instances]}
        if name == names[1]:
            remote_pids[1] = [0]
        procnode.initialize_processes(remote_pids)

        # Only linearize the actual app, not the "3rd program"
        if name == names[1] and linear:  # 2nd node
            to_linearize = procnode.scheduler.get_tasks_to_schedule_for(0)
            linearized = TaskGraphBuilder.merge_linear(to_linearize)
            third_prog_tasks = procnode.scheduler.get_tasks_to_schedule_for(1)
            hogging_task_id = third_prog_tasks[0].get_roots()[0]
            hogging_task = third_prog_tasks[0].get_tinfo(hogging_task_id).task
            # all_tasks = third_prog_tasks + [linearized]
            all_tasks = [linearized]
            merged = TaskGraphBuilder.merge(all_tasks)
        elif linear:  # first node
            tasks = procnode.scheduler.get_tasks_to_schedule()
            merged = TaskGraphBuilder.merge_linear(tasks)
        else:  # not linear
            tasks = procnode.scheduler.get_tasks_to_schedule()
            merged = TaskGraphBuilder.merge(tasks)
        procnode.scheduler.upload_task_graph(merged)

        logger = LogManager.get_stack_logger()
        for batch_id, prog_batch in procnode.scheduler.get_batches().items():
            task_graph = prog_batch.instances[0].task_graph
            num = len(prog_batch.instances)
            logger.info(f"batch {batch_id}: {num} instances each with task graph:")
            logger.info(task_graph)

    procnode = network.nodes[names[1]]
    procnode.scheduler.cpu_scheduler.set_hogging_task(hogging_task, 1000, hog_prob)

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

    del network
    gc.collect()

    return AppResult(results, statistics, total_duration)


def create_procnode_cfg(
    name: str, id: int, t1: float, t2: float, determ: bool, deadlines: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.uniform_t1t2_qubits_perfect_gates_default_params(
            5, t1, t2
        ),
        latencies=LatenciesConfig(qnos_instr_time=1000, host_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
        use_deadlines=deadlines,
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class ProgResult:
    alice_results: BatchResult
    bob_results: BatchResult
    qpu_waits: List[float]
    total_duration: float


def run_apps(
    num_const_tasks: int,
    t1: float,
    t2: float,
    cc_latency: float,
    busy_duration: float,
    determ_sched: bool,
    deadlines: bool,
    const_period: int,
    const_start: int,
    state: int,
    fcfs: bool,
    no_tasks: bool,
) -> ProgResult:
    ns.sim_reset()

    alice_id = 1
    bob_id = 0

    alice_node_cfg = create_procnode_cfg(
        "alice", alice_id, t1, t2, determ=determ_sched, deadlines=deadlines
    )
    bob_node_cfg = create_procnode_cfg(
        "bob", bob_id, t1, t2, determ=determ_sched, deadlines=deadlines
    )

    cconn = ClassicalConnectionConfig.from_nodes(alice_id, bob_id, cc_latency)
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[alice_node_cfg, bob_node_cfg], link_duration=1000
    )
    network_cfg.cconns = [cconn]

    alice_program = load_program("programs/controller.iqoala")
    bob_program = load_program("programs/interactive_quantum.iqoala")
    busy_program = load_program("programs/cpu_busy.iqoala")

    alice_inputs = [ProgramInput({"bob_id": bob_id})]

    def measure_state(prepare_state: int) -> int:
        return {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}[prepare_state]

    bob_inputs = [
        ProgramInput(
            {
                "alice_id": alice_id,
                "prepare_state": state,
                "measure_state": measure_state(state),
            },
        )
    ]

    busy_inputs = [
        ProgramInput({"duration": busy_duration}) for _ in range(num_const_tasks)
    ]

    app_result = run_two_node_app_separate_inputs_plus_constant_tasks(
        num_iterations=1,
        num_const_tasks=num_const_tasks,
        node1="alice",
        node2="bob",
        prog_node1=alice_program,
        prog_node1_inputs=alice_inputs,
        prog_node2=bob_program,
        prog_node2_inputs=bob_inputs,
        const_prog_node2=busy_program,
        const_prog_node2_inputs=busy_inputs,
        const_period=const_period,
        const_start=const_start,
        network_cfg=network_cfg,
        fcfs=fcfs,
        no_tasks=no_tasks,
        linear=True,
    )

    alice_result = app_result.batch_results["alice"]
    bob_result = app_result.batch_results["bob"]

    bob_stats = app_result.statistics["bob"]

    return ProgResult(alice_result, bob_result, [], app_result.total_duration)


@dataclass
class DataPoint:
    t2: float
    fcfs: bool
    no_tasks: bool
    latency_factor: float
    num_const_tasks: int
    const_period_factor: int
    busy_factor: float
    succ_prob: float
    succ_prob_lower: float
    succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    num_iterations: int


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def wilson_score_interval(p_hat, n, z):
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p_hat + z**2 / (2 * n)
    adjusted_standard_deviation = z * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n)) / n
    )

    lower_bound = (
        centre_adjusted_probability - adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + adjusted_standard_deviation
    ) / denominator

    return (lower_bound, upper_bound)


def get_metrics(
    start_time: float,
    t1: int,
    t2: int,
    determ: bool,
    use_deadlines: bool,
    cc_latency: int,
    num_const_tasks: int,
    const_period: int,
    const_start: int,
    busy_duration: int,
    state: int,
    fcfs: bool,
    no_tasks: bool,
) -> float:
    successes: List[bool] = []
    qpu_waits: List[List[float]] = []
    result = run_apps(
        num_const_tasks=num_const_tasks,
        t1=t1,
        t2=t2,
        cc_latency=cc_latency,
        busy_duration=busy_duration,
        determ_sched=determ,
        deadlines=use_deadlines,
        const_period=const_period,
        const_start=const_start,
        state=state,
        fcfs=fcfs,
        no_tasks=no_tasks,
    )
    program_results = result.bob_results.results
    outcomes = [result.values["outcome"] for result in program_results]
    assert len(outcomes) == 1
    successes = [outcome == 1 for outcome in outcomes]
    qpu_waits.append(result.qpu_waits)
    makespan = result.total_duration
    avg_succ = len([s for s in successes if s]) / len(successes)
    curr_time = round(time.time() - start_time, 3)

    return avg_succ, makespan


def run(deadlines: bool, output_dir: str):
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("classical_multitasking.log")
    # LogManager.enable_task_logger(True)
    # LogManager.log_tasks_to_file("classical_multitasking_tasks.log")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(timestamp)

    t1 = 1e10
    t2 = 1e8

    determ = True
    use_deadlines = deadlines

    latency_factor = 0.1

    num_const_tasks = 10
    const_period_factor = 1 / 3
    busy_factor = 10.2

    num_iterations = 1000
    succ_probs: List[float] = []
    makespans: List[float] = []
    fcfs = False
    no_tasks = False

    data_points: List[DataPoint] = []

    for (fcfs, no_tasks) in [(False, False), (True, False), (False, True)]:
        for i in range(num_iterations):
            print(i)
            cc_latency = latency_factor * t2
            const_period = const_period_factor * cc_latency
            const_start = cc_latency
            busy_duration = busy_factor * const_period
            succ_prob, makespan = get_metrics(
                start_time=start_time,
                t1=t1,
                t2=t2,
                determ=determ,
                use_deadlines=use_deadlines,
                cc_latency=cc_latency,
                num_const_tasks=num_const_tasks,
                const_period=const_period,
                const_start=const_start,
                busy_duration=busy_duration,
                state=i % 6,
                fcfs=fcfs,
                no_tasks=no_tasks,
            )
            succ_probs.append(succ_prob)
            makespans.append(makespan)

        avg_succ_prob = sum(succ_probs) / len(succ_probs)
        print(f"succ prob: {avg_succ_prob}")
        succ_prob_lower, succ_prob_upper = wilson_score_interval(
            p_hat=avg_succ_prob, n=len(succ_probs), z=1.96
        )
        print(succ_prob_lower, succ_prob_upper)

        data_points.append(
            DataPoint(
                t2=t2,
                fcfs=fcfs,
                no_tasks=no_tasks,
                latency_factor=latency_factor,
                num_const_tasks=num_const_tasks,
                const_period_factor=const_period,
                busy_factor=busy_factor,
                succ_prob=avg_succ_prob,
                succ_prob_lower=succ_prob_lower,
                succ_prob_upper=succ_prob_upper,
                makespan=makespan,
            )
        )

        end_time = time.time()
        sim_duration = end_time - start_time
        print(f"total duration: {sim_duration}s")

    meta = DataMeta(
        timestamp=timestamp,
        sim_duration=sim_duration,
        num_iterations=num_iterations,
    )
    data = Data(meta=meta, data_points=data_points)

    json_data = asdict(data)

    abs_dir = relative_to_cwd(f"data/{output_dir}")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, f"LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}.json")
    with open(last_path, "w") as datafile:
        json.dump(json_data, datafile)
    with open(timestamp_path, "w") as datafile:
        json.dump(json_data, datafile)


if __name__ == "__main__":
    run(deadlines=True, output_dir="data")
