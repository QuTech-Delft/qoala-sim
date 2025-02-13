import os
import random
from dataclasses import dataclass
from typing import Dict, List
import yaml

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import ProcNodeNetworkConfig, ProcNodeConfig, TopologyConfig, LatenciesConfig, \
    NtfConfig, ClassicalConnectionConfig
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager


def create_procnode_cfg(
    name: str, id: int, num_qubits: int, determ: bool
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
        determ_sched=determ,
    )


@dataclass
class AppResult:
    batch_results: Dict[str, BatchResult]
    statistics: Dict[str, SchedulerStatistics]
    total_duration: float


def load_program(name: str, interactions: dict) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), f"{name}.iqoala")
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()
    #for i, remote_name in enumerate(interactions['nodes'][name]):
    #    program.meta.csockets[i] = remote_name
    #    program.meta.epr_sockets[i] = remote_name
    return program


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
        deadline=1e8
    )


def run_n_clients(
    node_names: List[str],
    node_programs: Dict[str, QoalaProgram],
    node_inputs: Dict[str, List[ProgramInput]],
    network_cfg: ProcNodeNetworkConfig,
    interactions: Dict[str, dict],
    num_iterations: int = 1,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    remotes = {}

    for name in node_names:
        procnode = network.nodes[name]
        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        batch_info = create_batch(node_programs[name], unit_module, node_inputs[name], num_iterations)
        batch = procnode.submit_batch(batch_info)
        remotes[name] = batch

    for name in node_names:
        pids = []
        for remote in interactions[name]:
            pids.extend([
            inst.pid for inst in remotes[remote].instances
        ])
        network.nodes[name].initialize_processes(
            remote_pids={remotes[name].batch_id: pids},
            linear=linear)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in node_names:
        procnode = network.nodes[name]
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    return AppResult(results, statistics, makespan)


def run_sim(num_iterations: int, interactions: dict, programs: dict, node_ids: dict) -> AppResult:
    LogManager.set_log_level("WARN")
    # LogManager.log_to_file("ghz.log")
    # LogManager.log_tasks_to_file("ghz_tasks.log")
    # num_iterations = 2

    ns.sim_reset()

    num_qubits = 20
    node_configs = {}
    connections = []
    for name in interactions['nodes']:
        node_configs[name] = create_procnode_cfg(f"{name}", node_ids[name], num_qubits, determ=True)

    for conn in interactions['connections']:
        connections.append(ClassicalConnectionConfig.from_nodes(node_ids[conn[0]], node_ids[conn[1]], 1e9))

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=list(node_configs.values()), link_duration=1000
    )
    network_cfg.cconns = connections

    inputs = {}
    for name in node_ids:
        inputs[name] = [ProgramInput({f"{n}_id": node_ids[n] for n in interactions['nodes'][name]})]

    return run_n_clients(list(node_ids.keys()), programs, inputs, network_cfg, interactions['nodes'], num_iterations)


"""
Three nodes. Node0 prepares one qubit in the up state.
Node0 performs a remote cnot with Node1.
Node1 performs a remote cnot with Node2.
The three node should now share a GHZ state.
Every node measures at the end.
"""

n_iterations = 10

results = {}

with open('interactions.yaml', 'r') as file:
    interactions = yaml.safe_load(file)
print('Interactions',interactions)

node_ids = {}
for name in interactions['nodes']:
    node_ids.update({name: len(node_ids)})
print('node_ids', node_ids)
    
programs = {}
for name in node_ids:
    programs[name] = load_program(name, interactions)

for iteration in range(n_iterations):
    iter_results = run_sim(1, interactions, programs, node_ids)
    bitstring = ''.join([str(v) for v in iter_results.batch_results['node0'].results[0].values.values()])
    bitstring += ''.join([str(v) for v in iter_results.batch_results['node1'].results[0].values.values()])
    bitstring += ''.join([str(v) for v in iter_results.batch_results['node2'].results[0].values.values()])
    if bitstring not in results:
        results[bitstring] = 0
    results[bitstring] += 1

print(results)
