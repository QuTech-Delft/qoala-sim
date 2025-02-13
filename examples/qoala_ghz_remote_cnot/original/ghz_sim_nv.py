import os
import random
from dataclasses import dataclass
from typing import Dict, List
import yaml

import netsquid as ns
from netqasm.lang.instr import TrappedIonFlavour, NVFlavour, core
from qoala.sim.network import ProcNodeNetwork
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from qoala.sim.build import build_network_from_lhi
from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import ProcNodeNetworkConfig, ProcNodeConfig, TopologyConfig, LatenciesConfig, \
    NtfConfig, ClassicalConnectionConfig, NvParams
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiGateInfo,
    LhiLatencies,
    LhiProcNodeInfo,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiQubitInfo,
    LhiTopology,
    LhiTopologyBuilder,
)
from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)
from qoala.sim.qdevice import QDevice
from netsquid.nodes import Node
from qoala.sim.build import build_qprocessor_from_topology
from typing import Dict, List, Optional, Tuple
from qoala.sim.qnos import (
    GenericProcessor,
    NVProcessor,
    QnosComponent,
    QnosInterface,
    QnosLatencies,
    QnosProcessor,
)
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.ntf import GenericNtf, NvNtf, TrappedIonNtf
from qoala.sim.memmgr import AllocError, MemoryManager, NotAllocatedError
from qoala.sim.qnos.qnosprocessor import (
    IonTrapProcessor,
    UnsupportedNetqasmInstructionError,
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
    program = QoalaParser(text, flavour=NVFlavour(), transpiler=True).parse()
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
    nodes,
    node_names: List[str],
    node_programs: Dict[str, QoalaProgram],
    node_inputs: Dict[str, List[ProgramInput]],
    network: ProcNodeNetwork,
    interactions: Dict[str, dict],
    num_iterations: int = 1,
    linear: bool = False,
) -> AppResult:
    ns.sim_reset()
    #ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)
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

    num_qubits = 4
    node_configs = {}
    connections = []
    nodes = []

    #topology = TopologyConfig.from_nv_params(num_qubits=num_qubits, params=NvParams())
    
    topology = LhiTopologyBuilder.perfect_star(num_qubits=num_qubits,
        comm_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
            INSTR_MEASURE,
        ],
        comm_duration=5e3,
        mem_instructions=[
            INSTR_INIT,
            INSTR_ROT_X,
            INSTR_ROT_Y,
            INSTR_ROT_Z,
        ],
        mem_duration=1e4,
        two_instructions=[INSTR_CXDIR, INSTR_CYDIR],
        two_duration=1e5,)
    latencies = LhiLatencies(
        host_instr_time=1000, qnos_instr_time=2000, host_peer_latency=3000
    )
    node0 = LhiProcNodeInfo(name='node0', id=0, topology=topology, latencies=latencies)
    node1 = LhiProcNodeInfo(name='node1', id=1, topology=topology, latencies=latencies)
    node2 = LhiProcNodeInfo(name='node2', id=2, topology=topology, latencies=latencies)
    node_configs = {0: "node0", 1: "node1" , 2: "node2"}
        
    for conn in interactions['connections']:
        connections.append(ClassicalConnectionConfig.from_nodes(node_ids[conn[0]], node_ids[conn[1]], 1e9))

    network_lhi = LhiNetworkInfo.fully_connected(node_configs, LhiLinkInfo.perfect(duration=1000))

    inputs = {}
    for name in node_ids:
        inputs[name] = [ProgramInput({f"{n}_id": node_ids[n] for n in interactions['nodes'][name]})]

    ntfs = [NvNtf(), NvNtf(), NvNtf()]
    network = build_network_from_lhi([node0,node1,node2],ntfs,network_lhi)

    
    return run_n_clients(nodes,list(node_ids.keys()), programs, inputs, network, interactions['nodes'], num_iterations)


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

node_ids = {}
for name in interactions['nodes']:
    node_ids.update({name: len(node_ids)})
print(node_ids)
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