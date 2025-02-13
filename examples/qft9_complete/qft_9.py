
import random
from typing import Dict, List

from qoala.runtime.config import ProcNodeNetworkConfig, ProcNodeConfig, TopologyConfig, LatenciesConfig, \
    NtfConfig, ClassicalConnectionConfig
from qoala.lang.program import QoalaProgram

from qoala.lang.ehi import UnitModule
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
import os
from qoala.lang.parse import QoalaParser
import netsquid as ns
from qoala.runtime.statistics import SchedulerStatistics
from dataclasses import dataclass
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.sim.build import (build_network_from_config)

@dataclass
class AppResult:
    batch_results: Dict[str, BatchResult]
    statistics: Dict[str, SchedulerStatistics]
    total_duration: float

def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()
    return program

def qft():
    top_cfg = TopologyConfig.perfect_config_uniform_default_params(num_qubits=6)
    cfg_node0 = ProcNodeConfig(
        node_name="node0",
        node_id=0,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg_node1 = ProcNodeConfig(
        node_name="node1",
        node_id=1,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg_node2 = ProcNodeConfig(
        node_name="node2",
        node_id=2,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    cfg_node3 = ProcNodeConfig(
        node_name="node3",
        node_id=3,
        topology=top_cfg,
        latencies=LatenciesConfig(),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )
    
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=[cfg_node0, cfg_node1, cfg_node2, cfg_node3], link_duration=1000
    )
    
    programs = {}
    for name in ["node0.iqoala", "node1.iqoala", "node2.iqoala", "node3.iqoala"]:
        programs[name[:5]] = load_program(name)

    
    connections = []
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node0.node_id, cfg_node1.node_id, 1e9))
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node0.node_id, cfg_node2.node_id, 1e9))
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node0.node_id, cfg_node3.node_id, 1e9))
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node1.node_id, cfg_node2.node_id, 1e9))
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node1.node_id, cfg_node3.node_id, 1e9))
    connections.append(ClassicalConnectionConfig.from_nodes(cfg_node2.node_id, cfg_node3.node_id, 1e9))
    
    network_cfg.cconns = connections

    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)
    
    network = build_network_from_config(network_cfg)
    
    node_ids = {'node0':0,'node1':1,'node2':2,'node3':3}
    interactions = {'node0':['node1','node2','node3'],'node1':['node0','node2','node3'],'node2':['node1','node0','node3'],'node3':['node1','node2','node0']}
    inputs = {}
    for name in node_ids:
        inputs[name] = [ProgramInput({f"{n}_id": node_ids[n] for n in interactions[name]})]

    remotes = {}

    for name in node_ids:
        unit_module = UnitModule.from_full_ehi(network.nodes[name].memmgr.get_ehi())
        batch_info = BatchInfo(program=programs[name], unit_module=unit_module, inputs=inputs[name], num_iterations=1, deadline=1e8)
        remotes[name] = network.nodes[name].submit_batch(batch_info)
        
    for name in node_ids:
        pids = []
        for remote in interactions[name]:
            pids.extend([
            inst.pid for inst in remotes[remote].instances
        ])
        network.nodes[name].initialize_processes(
            remote_pids={remotes[name].batch_id: pids},
            linear=False)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    results: Dict[str, BatchResult] = {}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in node_ids:
        procnode = network.nodes[name]
        results[name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()
        print(name)
        print(procnode.scheduler.get_batch_results())
        print(procnode.scheduler.get_statistics())
        print('------------')
    print('time',makespan)

qft()