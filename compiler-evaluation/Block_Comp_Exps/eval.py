from __future__ import annotations

import datetime
import ent_rate
import json
import math
import os
import random
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import netsquid as ns

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramBatch, ProgramInput
from qoala.sim.build import build_network_from_config
from qoala.sim.network import ProcNodeNetwork
from qoala.util.logging import LogManager


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def get_node_config(
    name: str,
    id: int,
    num_qubits: int,
    t1: int = 1e9,
    t2: int = 1e8,
    single_gate_dur: int = 5e3,
    two_gate_dur: int = 5e3,
    all_gate_dur: int = 5e3,
    single_gate_fid: float = 1.0,
    two_gate_fid: float = 1.0,
    all_gate_fid: float = 1.0,
    qnos_instr_proc_time: int = 50e3,
    host_instr_time: int = 0,
    host_peer_latency: int = 0,
    internal_sched_latency: int = 0,
) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.uniform_t1t2_qubits_uniform_any_gate_duration_and_noise(
            num_qubits=num_qubits,
            t1=t1,
            t2=t2,
            single_gate_duration=single_gate_dur,
            two_gate_duration=two_gate_dur,
            single_gate_fid=single_gate_fid,
            two_gate_fid=two_gate_fid,
        ),
        latencies=LatenciesConfig(
            host_instr_time=host_instr_time,
            host_peer_latency=host_peer_latency,
            qnos_instr_time=qnos_instr_proc_time,
            internal_sched_latency=internal_sched_latency
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_configs: List[ProcNodeConfig],
    num_clients: int,
    client_num_iterations: List[int],
    cc: float,
    use_netschedule: bool,
    bin_length: float,
    link_duration: int = 1000,
    link_fid: float = 1.0,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    node_cfgs = [server_cfg] + client_configs
    network_cfg = ProcNodeNetworkConfig.from_nodes_imperfect_links(
        nodes=node_cfgs, link_fid=link_fid, link_duration=link_duration
    )

    # pattern = []
    # server_pid_index = 0
    # for i in range(num_clients):
    #     # TODO Rework this whol func
    #     for j in range(num_iterations[i]):
    #         # client node id, client pid, server node id, server pid
    #         pattern.append((i + 1, j, 0, server_pid_index))
    #         server_pid_index += 1

    # Create netschedule pattern (client node, client pid, server node, server pid)
    # Above we have the netschedule divided into blocks of
    # [client 1 pid 1, client 1 pid 2, ..., client 1 pid n, ..., client c pid 1, client c pid 2, ..., client c pid n]
    # When we create the programs we do
    # Server Proc Prog 1 Client 1, Server Proc Prog 1 Client 2, ..., Server Proc Prog p Client 1, Server Proc Prog p Client 2,...
    pattern = []
    server_pid_index = 0
    client_pid_index = 0
    for client_prog in range(len(client_num_iterations)):
        for client_index in range(1, num_clients + 1):
            for iteration in range(client_num_iterations[client_prog]):
                pattern.append(
                    (client_index, client_pid_index + iteration, 0, server_pid_index)
                )
                server_pid_index += 1

        client_pid_index += client_num_iterations[client_prog]
    
    # Randomize the order of the network schedule for robustness of the experiment    
    first_bin = random.randint(0,len(pattern)-1)
    random.shuffle(pattern)
    # print(pattern)

    if use_netschedule:
        network_cfg.netschedule = NetworkScheduleConfig(
            bin_length=bin_length,
            first_bin=first_bin,
            bin_pattern=pattern,
            repeat_period=bin_length
            * num_clients
            * len(client_num_iterations)
            * max(client_num_iterations),
        )

    cconns = [
        ClassicalConnectionConfig.from_nodes(i, 0, cc)
        for i in range(1, num_clients + 1)
    ]
    network_cfg.cconns = cconns
    return build_network_from_config(network_cfg)


def load_server_program(program_name: str, remote_name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), program_name)
    with open(path) as file:
        server_text = file.read()
    program = QoalaParser(server_text).parse()

    # Replace "client" by e.g. "client_1"
    program.meta.csockets[0] = remote_name
    program.meta.epr_sockets[0] = remote_name

    return program


def load_client_program(program_name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), program_name)
    with open(path) as file:
        client_text = file.read()
    return QoalaParser(client_text).parse()


def create_server_batch(
    client_id: int,
    inputs: List[ProgramInput],
    program_name: str,
    unit_module: UnitModule,
    num_iterations: int,
) -> BatchInfo:
    server_program = load_server_program(
        program_name=program_name, remote_name=f"client_{client_id}"
    )
    return BatchInfo(
        program=server_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )


def create_client_batch(
    inputs: List[ProgramInput],
    program_name: str,
    unit_module: UnitModule,
    num_iterations: int,
) -> BatchInfo:
    client_program = load_client_program(program_name=program_name)
    return BatchInfo(
        program=client_program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )


@dataclass
class EvalExpResult:
    client_results: BatchResult
    server_results: BatchResult
    total_duration: float


def run_eval_programs(
    client_progs: List[str],
    server_progs: List[str],
    client_prog_args: List[dict],
    server_prog_args: List[dict],
    client_num_iterations: List[int],
    server_num_iterations: List[int],
    num_clients: int = 1,
    linear: bool = True,
    cc: int = 0,
    t1: int = 1e9,
    t2: int = 1e8,
    single_gate_dur: int = 5e3,
    two_gate_dur: int = 5e3,
    all_gate_dur: int = 5e3,
    single_gate_fid: float = 1.0,
    two_gate_fid: float = 1.0,
    all_gate_fid: float = 1.0,
    qnos_instr_proc_time: int = 50e3,
    host_instr_time: int = 0,
    host_peer_latency: int = 0,
    internal_sched_latency=0,
    server_num_qubits: int = 2,
    client_num_qubits: int = 20,
    use_netschedule: bool = False,
    bin_length: int = 0,
    link_duration: int = 1000,
    link_fid: float = 1.0,
    scheduling_alg: str = "",
    random_client_inputs: bool = False,
    client_input_func: List[function] = [],
    random_server_inputs: bool = False,
    server_input_func: List[function] = [],
    seed = 0,
):
    """

    :param client_progs: List of programs for the client to run
    :param server_progs: List of programs for the server to run
    :param client_prog_args: List of dictionaries containing the client program arguments. client_prog_args[i] are the arguments for client_progs[i]
    :param server_prog_args: List of dictionaries containing the server program arguments. server_prog_args[i] are the arguments for server_progs[i]
    :param server_num_iterations: List of number of iterations for each program. server_num_iterations[i] is the number of iterations for server_progs[i].
    :param client_num_iterations: List of number of iterations for each program. client_num_iterations[i] is the number of iterations for client_progs[i].
    :param num_clients: defaults to 1
    :param linear: defaults to True
    :param cc: _description_, defaults to 0
    :param t1: _description_, defaults to 1e9
    :param t2: _description_, defaults to 1e8
    :param single_gate_dur: _description_, defaults to 5e3
    :param two_gate_dur: _description_, defaults to 5e3
    :param all_gate_dur: _description_, defaults to 5e3
    :param single_gate_fid: _description_, defaults to 1.0
    :param two_gate_fid: _description_, defaults to 1.0
    :param all_gate_fid: _description_, defaults to 1.0
    :param qnos_instr_proc_time: _description_, defaults to 50e3
    :param host_instr_time: _description_, defaults to 0
    :param host_peer_latency: _description_, defaults to 0
    :param server_num_qubits: _description_, defaults to 2
    :param client_num_qubits: defaults to 20
    :param use_netschedule: _description_, defaults to False
    :param bin_length: _description_, defaults to 0
    :param scheduling_alg: _description_, defaults to ""
    :param random_client_inputs: randomize client inputs
    :param client_input_func: function used to randomize client inputs
    :return: _description_
    """
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    ns.set_random_state(seed=seed)
    random.seed(seed)

    # Configure server node
    server_config = get_node_config(
        name="server",
        id=0,
        num_qubits=server_num_qubits,
        t1=t1,
        t2=t2,
        single_gate_dur=single_gate_dur,
        two_gate_dur=two_gate_dur,
        all_gate_dur=all_gate_dur,
        single_gate_fid=single_gate_fid,
        two_gate_fid=two_gate_fid,
        all_gate_fid=all_gate_fid,
        qnos_instr_proc_time=qnos_instr_proc_time,
        host_instr_time=host_instr_time,
        host_peer_latency=host_peer_latency,
        internal_sched_latency=internal_sched_latency
    )
    # Configure client nodes
    client_configs = [
        get_node_config(
            name=f"client_{i}",
            id=i,
            num_qubits=client_num_qubits,
            t1=t1,
            t2=t2,
            single_gate_dur=single_gate_dur,
            two_gate_dur=two_gate_dur,
            all_gate_dur=all_gate_dur,
            single_gate_fid=single_gate_fid,
            two_gate_fid=two_gate_fid,
            all_gate_fid=all_gate_fid,
            qnos_instr_proc_time=qnos_instr_proc_time,
            host_instr_time=host_instr_time,
            host_peer_latency=host_peer_latency,
            internal_sched_latency=internal_sched_latency
        )
        for i in range(1, num_clients + 1)
    ]

    # Create the network
    network = create_network(
        server_config,
        client_configs,
        num_clients,
        client_num_iterations,
        cc,
        use_netschedule,
        bin_length,
        link_duration,
        link_fid
    )

    server_procnode = network.nodes["server"]

    # client ID -> list of server batches where each index is the next program
    # key -1 represents server only programs, where index 0 is program_index len(client_progs)
    server_batches: Dict[int, List[ProgramBatch]] = {}

    # client ID -> list of client batches where each index is the next program
    client_batches: Dict[int, List[ProgramBatch]] = {}

    # Initialize server_batches and client_batches to empty lists
    for client_id in range(1, num_clients + 1):
        server_batches[client_id] = []
        client_batches[client_id] = []

    # Local server programs
    server_batches[-1] = []

    # Create the server program batches
    for prog_index in range(len(server_progs)):
        # This is a client<->server application
        if prog_index < len(client_progs):
            for client_id in range(1, num_clients + 1):
                server_inputs = []
                if random_server_inputs:
                    server_inputs = [server_input_func[prog_index](client_id)
                                     for _ in range(server_num_iterations[prog_index])]
                else:
                    server_inputs = [
                        ProgramInput(
                            {"client_id": client_id} | server_prog_args[prog_index]
                        )
                        for _ in range(server_num_iterations[prog_index])
                    ]

                server_unit_module = UnitModule.from_full_ehi(
                    server_procnode.memmgr.get_ehi()
                )
                server_batch_info = create_server_batch(
                    client_id=client_id,
                    inputs=server_inputs,
                    program_name=server_progs[prog_index],
                    unit_module=server_unit_module,
                    num_iterations=server_num_iterations[prog_index],
                )

                # Creates a batch of program instances, so each program now has a pid
                server_batches[client_id] += [
                    server_procnode.submit_batch(server_batch_info)
                ]

        # This is a server only application
        else:
            server_inputs = [
                ProgramInput(server_prog_args[prog_index])
                for _ in range(server_num_iterations[prog_index])
            ]

            server_unit_module = UnitModule.from_full_ehi(
                server_procnode.memmgr.get_ehi()
            )
            server_batch_info = create_server_batch(
                client_id=client_id,
                inputs=server_inputs,
                program_name=server_progs[prog_index],
                unit_module=server_unit_module,
                num_iterations=server_num_iterations[prog_index],
            )

            # Creates a batch of program instances, so each program now has a pid
            server_batches[-1] += [server_procnode.submit_batch(server_batch_info)]

    # Create the client batches
    for prog_index in range(len(client_progs)):
        for client_id in range(1, num_clients + 1):
            
            client_inputs = []

            if random_client_inputs:
                # Use the input function to generate the input for each program instance
                client_inputs = [client_input_func[prog_index](0) 
                                 for _ in range(client_num_iterations[prog_index])]
            else:
                client_inputs = [
                    ProgramInput({"server_id": 0} | client_prog_args[prog_index])
                    for _ in range(client_num_iterations[prog_index])
                ]

            client_procnode = network.nodes[f"client_{client_id}"]

            client_unit_module = UnitModule.from_full_ehi(
                client_procnode.memmgr.get_ehi()
            )
            client_batch_info = create_client_batch(
                inputs=client_inputs,
                program_name=client_progs[prog_index],
                unit_module=client_unit_module,
                num_iterations=client_num_iterations[prog_index],
            )

            client_batches[client_id] += [
                client_procnode.submit_batch(client_batch_info)
            ]

    # Need to initialize all of the client processes AFTER they have all been created...
    for client_id in range(1, num_clients + 1):
        client_procnode = network.nodes[f"client_{client_id}"]

        remote_pids: Dict[int, List[int]] = {}

        for prog_index in range(len(client_progs)):
            # Get the batch ID
            batch_id = client_batches[client_id][prog_index].batch_id
            server_pids = [
                inst.pid for inst in server_batches[client_id][prog_index].instances
            ]
            # print(
            #      f"client ID: {client_id}, batch ID: {batch_id}, server PIDs: {server_pids}"
            # )
            remote_pids[batch_id] = server_pids

        client_procnode.initialize_processes(remote_pids=remote_pids, linear=linear)

    # Need to initialize all of the server processes now
    remote_pids: Dict[int, List[int]] = {}
    for client_id in range(1, num_clients + 1):
        for prog_index in range(len(client_progs)):
            batch_id = server_batches[client_id][prog_index].batch_id
            client_pids = [
                inst.pid for inst in client_batches[client_id][prog_index].instances
            ]
            # print(
            #       f"client ID: {client_id}, batch ID: {batch_id}, client PIDs: {client_pids}"
            # )
            remote_pids[batch_id] = client_pids

    
    # If there are local only server programs we need to give them dummy remote pids
    num_local_progs = len(server_progs) - len(client_progs)
    for local_prog_index in range(num_local_progs):
        batch = server_batches[-1][local_prog_index]
        batch_id = batch.batch_id 
        remote_pids[batch_id] = [-1 for inst in batch.instances] 
    
    server_procnode.initialize_processes(remote_pids=remote_pids, linear=linear)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    client_procnodes = [network.nodes[f"client_{i}"] for i in range(1, num_clients + 1)]
    client_batches = [node.get_batches() for node in client_procnodes]

    client_results: List[Dict[int, BatchResult]]
    client_results = [node.scheduler.get_batch_results() for node in client_procnodes]

    server_results: Dict[int, BatchResult]
    server_results = server_procnode.scheduler.get_batch_results()
    # print(client_results[0])
    return EvalExpResult(
        server_results=server_results,
        client_results=client_results,
        total_duration=makespan,
    )


def run_eval_exp(
    client_progs: List[str],
    server_progs: List[str],
    client_prog_args: List[dict],
    server_prog_args: List[dict],
    client_num_iterations: List[int],
    server_num_iterations: List[int],
    num_clients: int = 1,
    linear: bool = True,
    cc: int = 0,
    t1: int = 1e9,
    t2: int = 1e8,
    single_gate_dur: int = 5e3,
    two_gate_dur: int = 5e3,
    all_gate_dur: int = 5e3,
    single_gate_fid: float = 1.0,
    two_gate_fid: float = 1.0,
    all_gate_fid: float = 1.0,
    qnos_instr_proc_time: int = 50e3,
    host_instr_time: int = 0,
    host_peer_latency: int = 0,
    internal_sched_latency: int = 0,
    server_num_qubits: int = 2,
    client_num_qubits: int = 20,
    use_netschedule: bool = False,
    bin_length: int = 0,
    link_duration: int = 1000,
    link_fid: float = 1.0,
    scheduling_alg: str = "",
    compute_succ_probs: List[function] = [],
    random_client_inputs: bool = False,
    client_input_func: List[function] = [],
    random_server_inputs: bool = False,
    server_input_func: List[function] = [],
    seed = 0
):
    exp_results = run_eval_programs(
        client_progs=client_progs,
        server_progs=server_progs,
        client_prog_args=client_prog_args,
        server_prog_args=server_prog_args,
        client_num_iterations=client_num_iterations,
        server_num_iterations=server_num_iterations,
        num_clients=num_clients,
        linear=linear,
        cc=cc,
        t1=t1,
        t2=t2,
        single_gate_dur=single_gate_dur,
        two_gate_dur=two_gate_dur,
        all_gate_dur=all_gate_dur,
        single_gate_fid=single_gate_fid,
        two_gate_fid=two_gate_fid,
        all_gate_fid=all_gate_fid,
        qnos_instr_proc_time=qnos_instr_proc_time,
        host_instr_time=host_instr_time,
        host_peer_latency=host_peer_latency,
        internal_sched_latency=internal_sched_latency,
        server_num_qubits=server_num_qubits,
        client_num_qubits=client_num_qubits,
        use_netschedule=use_netschedule,
        bin_length=bin_length,
        link_duration=link_duration,
        link_fid=link_fid,
        scheduling_alg=scheduling_alg,
        random_client_inputs=random_client_inputs,
        client_input_func=client_input_func,
        random_server_inputs= random_server_inputs,
        server_input_func = server_input_func,
        seed=seed
    )

    total_makespan = exp_results.total_duration
    
    # Need to compute success probabilities
    server_results = exp_results.server_results
    client_results = exp_results.client_results
    
    # (client_id, program #) -> succ probability
    # A client_id of -1 corresponds to a server only program
    succ_probs: Dict[(int, int), float] = {}
    # (client_id, progam #) -> avg makespan
    makespans: Dict[(int,int), float] = {}

    if compute_succ_probs is not None:
        for prog_index in range(len(compute_succ_probs)):
            # There will be one function to compute success probabilities for each program
            compute_succ_func = compute_succ_probs[prog_index]
            
            if not callable(compute_succ_func):
                continue
            
            # This function computes the success probs and makespans for a client-server app
            if prog_index < len(client_progs):
                # Need to compute the success probabilities for each client
                for client_index in range(0, num_clients):
                    # Results for client index + 1 for program prog_index
                    # Results for the client are indexed first by client, and then by batch
                    # The ith batch on the client is the ith program
                    client_prog_results = client_results[client_index][
                        prog_index
                    ].results

                    # Need to get corresponding server program results
                    # The server results are a little different than the client results
                    # We first iterate over programs, and then clients
                    # So for two programs and two clients we have
                    # batch 0: program 0, client 0
                    # batch 1: program 0, client 1
                    # batch 2: program 1, client 0
                    # batch 3: program 1, client 1
                    # So we have batch index = num_clients*program index + 1*client index
                    server_batch_index = num_clients * prog_index + client_index
                    server_prog_results = server_results[server_batch_index].results

                    # Compute the success probability and store it
                    succ_probs[(client_index + 1, prog_index + 1)] = compute_succ_func(
                        server_prog_results,
                        client_prog_results,
                    )

                    client_timestamps = client_results[client_index][prog_index].timestamps
                    raw_makespans = [ts[1]-ts[0] for ts in client_timestamps]
                    makespans[(client_index+1, prog_index+1)] = sum(raw_makespans) / len(raw_makespans)

            # This function computes the success probs for a server only app
            else:
                # Compute the batch index
                # We know that the batches for server programs are after all of the client programs
                # So we need to offset by the number of clients executing client programs (num_clients * len(client_progs))
                # Then we need to get the relative index of the server only program
                # There are len(client_progs) server-client programs, so we subtract that from the program index to achieve this
                server_batch_index = num_clients * len(client_progs) + (prog_index - len(client_progs)) 
                server_batch_results = server_results[server_batch_index]
                server_prog_results = server_batch_results.results
                server_prog_timestamps = server_batch_results.timestamps

                succ_probs[-1, prog_index+1] = compute_succ_func(server_prog_results)

                raw_makespans = [ts[1]-ts[0] for ts in server_prog_timestamps]
                makespans[(-1, prog_index+1)] = sum(raw_makespans) / len(raw_makespans)

    return total_makespan, succ_probs, makespans


@dataclass
class DataPoint:
    selfish_bqc_makespan: float
    selfish_local_makespan: float
    cooperative_bqc_makespan: float
    cooperative_local_makespan: float
    selfish_bqc_succ_prob: float
    selfish_local_succ_prob: float
    cooperative_bqc_succ_prob: float
    cooperative_local_succ_prob: float   
    prog_size: int
    num_clients: int
    param_name: str  # Name of param being varied
    param_value: float  # Value of the varied param

@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    hardware: str
    qia_sga: int
    scenario: int
    prog_sizes: List[int]
    num_iterations: List[int]
    num_trials: int
    linear: bool
    cc: int
    t1: int
    t2: int
    single_gate_dur: int
    two_gate_dur: int
    all_gate_dur: int
    single_gate_fid: float
    two_gate_fid: float
    all_gate_fid: float
    qnos_instr_proc_time: int
    host_instr_time: int
    host_peer_latency: int
    internal_sched_latency: int
    client_num_qubits: int
    server_num_qubits: int
    use_netschedule: bool
    bin_length: int
    param_name: str  # The parameter being varied
    link_duration: int
    link_fid: float
    seed: int

@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]

def rotation_server_input(client_id):
    return ProgramInput({
        "client_id" : client_id,
        "state": random.randint(0,5)
    })

def rotation_client_input_generator(prog_size): 
    def ret_input(server_id):
        thetas = [random.randint(1,32) for i in range(0,prog_size-1)]
        inputs = { 
            f"theta{i}" : thetas[i] for i in range(0,prog_size-1)
        }
        inputs[f"theta{prog_size-1}"] = -1 * sum(thetas)
        inputs["server_id"] = server_id
        return ProgramInput(inputs)
    return ret_input

def rotation_compute_succ_prob(server_batch_results, client_batch_results):
    results = [result.values["result"] for result in client_batch_results]
    num_iterations = len(results)
    # A success is when the program measures a 0
    # (the measurement outcome is the same as the initial state)
    successes = 0
    for res in results:
        if res == 0:
            successes += 1
    return (successes / num_iterations)

def bqc_compute_succ_prob_3(
    server_batch_results, client_batch_results
):
    num_iterations = len(client_batch_results)
    meas_outcomes = [
        result.values["m2"] for result in client_batch_results
    ]
    successes = [1 if (outcome == 0) else 0 for outcome in meas_outcomes]

    return sum(successes) / num_iterations

def bqc_compute_succ_prob_5(
    server_batch_results, client_batch_results
):
    num_iterations = len(client_batch_results)
    meas_outcomes = [
        result.values["m4"] for result in client_batch_results
    ]
    successes = [1 if (outcome == 1) else 0 for outcome in meas_outcomes]

    return sum(successes) / num_iterations

def bqc_compute_succ_prob_10(
    server_batch_results, client_batch_results
):
    num_iterations = len(client_batch_results)
    meas_outcomes = [
        (result.values["m4"], result.values["m9"]) for result in client_batch_results
    ]
    successes = [1 if ((outcome[0] == outcome[1]) and (outcome[0] == 1)) else 0 for outcome in meas_outcomes]

    return sum(successes) / num_iterations

def bqc_inputs_3(server_id):
    # Input for a 3 qubit BQC app
    # The qubit is initialized to the |-> state
    # Then a Z gates is applied so that the qubit is in the |+> state
    # The finally output will be |+>
    # The thetas are randomized to hide the input state
    inputs = {
        "server_id" : server_id,
        "input0": 1,
        "x0": 0,
        "angle0": 16,
        "angle1": 0,
        "angle2": 0,
        "theta0": random.randint(0,32),
        "theta1": random.randint(0,32),
        "theta2": random.randint(0,32),
        "dummy0": 0,
        "dummy1": 0,
        "dummy2": 0
    }
    return ProgramInput(inputs)

def bqc_inputs_5(server_id):
    # Input for a 5 qubit BQC app
    # The qubit is initialized to the |1> state
    # Then an H gate is applied using three pi/2 measurements
    # So the final qubit will be in the |-> state 
    # The thetas are randomized to hide the input state
    inputs = {
        "server_id" : server_id,
        "input0": 5,
        "x0": 0,
        "angle0": 8,
        "angle1": 8,
        "angle2": 8,
        "angle3": 0,
        "angle4": 0,
        "theta0": random.randint(0,32),
        "theta1": random.randint(0,32),
        "theta2": random.randint(0,32),
        "theta3": random.randint(0,32),
        "theta4": random.randint(0,32),
        "dummy0": 0,
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "dummy4": 0,
    }
    return ProgramInput(inputs)

def bqc_inputs_10(server_id):
    # Input for a 10 qubit BQC app
    # The two input qubits are initialized to |1> and |0>
    # Then a CNOT gate is applied
    # So the final state will be |11>
    # The thetas are randomized to hide the input state
    inputs = {
        "server_id" : server_id,
        "input0": 5,
        "x0": 0,
        "input5": 4,
        "x5": 0,
        "angle0": 0,
        "angle1": 0,
        "angle2": 8,
        "angle3": 0,
        "angle4": 0,
        "angle5": 0,
        "angle6": 8,
        "angle7": 0,
        "angle8": -8,
        "angle9": 0,
        "theta0": 0,
        "theta1": random.randint(0,32),
        "theta2": random.randint(0,32),
        "theta3": random.randint(0,32),
        "theta4": random.randint(0,32),
        "theta5": random.randint(0,32),
        "theta6": random.randint(0,32),
        "theta7": random.randint(0,32),
        "theta8": random.randint(0,32),
        "theta9": random.randint(0,32),
        "dummy0": 0,
        "dummy1": 0,
        "dummy2": 0,
        "dummy3": 0,
        "dummy4": 0,
        "dummy5": 0,
        "dummy6": 0,
        "dummy7": 0,
        "dummy8": 0,
        "dummy9": 0,
    }
    return ProgramInput(inputs)

def local_compute_succ_prob_generator(scenario):
    def local_compute_succ_prob(server_prog_results):
        if scenario == 1: 
            successes = [(1-result.values["outcome"]/100) for result in server_prog_results]
            return sum(successes) / len(successes)
        else: 
            successes = [1 if (result.values["outcome"] == 0) else 0 for result in server_prog_results]
            return sum(successes) / len(successes)
    return local_compute_succ_prob

if __name__ == "__main__":
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("eval_debug.log")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_tasks_to_file("eval_debug_TASKS.log")

    # Succ prob function map
    succ_prob_func = {
        "bqc_compute_succ_prob_3" : bqc_compute_succ_prob_3,
        "bqc_compute_succ_prob_5" : bqc_compute_succ_prob_5,
        "bqc_compute_succ_prob_10" : bqc_compute_succ_prob_10,
    }

    # Input function map
    client_input_func = {
        "bqc_inputs_3" : bqc_inputs_3,
        "bqc_inputs_5" : bqc_inputs_5,
        "bqc_inputs_10" : bqc_inputs_10,
    }

    # Default values
    params = { 
        "scenario": 1,
        "hardware": "NV",
        "qia_sga": 1,
        "distance": 0, 
        "client_num_iterations": [],
        "server_num_iterations": [],
        "num_clients": 1,
        "linear": False,
        "cc": 1e5,
        "t1": 0,
        "t2": 7.5e9,
        "single_gate_dur": 20e3,
        "two_gate_dur": 500e3,
        "all_gate_dur": 500e3,
        "single_gate_fid": 0.97,
        "two_gate_fid": 0.97,
        "all_gate_fid": 0.97,
        "qnos_instr_proc_time": 50e3,
        "host_instr_time": 15,
        "host_peer_latency": 150,
        "client_num_qubits": 2,
        "use_netschedule": True,
        "bin_length": 1,
        "link_fid": 1.0
    }

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--param_sweep_list", type=str, nargs="+", required=False)
    parser.add_argument("--num_trials", type=int, required=False, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--num_clients", "-c", required=True, type=int)

    # Parse arguments
    args = parser.parse_args()
    config = args.config
    param_sweep_list = args.param_sweep_list
    save = args.save
    seed = args.seed
    num_trials = args.num_trials
    num_clients = args.num_clients

    if config is not None:
        # Load param values from .json file
        config_file = open(config)
        config_obj = json.load(config_file)
        for key, val in config_obj.items():
            #print(key, val)
            params[key] = val

    param_name = "cc"
    param_vals = [params["cc"]]
    # Load the configuration file
    if param_sweep_list is not None:
        param_name = param_sweep_list[0]
        param_vals = [float(val) for val in param_sweep_list[1:]]
    
    # Load the seed
    seed_value = 0
    if seed is not None:
        # Load seed file
        seed_file = open("configs/qrng_seeds.json")
        seed_obj = json.load(seed_file)
        seed_str = seed_obj["seeds"][seed]
        seed_value = int(seed_str, 16)

    program_sizes = [3,5]#,10]
    # Load program inputs and other values based on the scenario
    if params["scenario"] == 1 or params["scenario"] == 2:
        params["client_progs"] = [[f"programs/bqc/vbqc_client_{i}.iqoala"] for i in program_sizes]
        
        params["server_progs_self"] = [[f"programs/bqc/vbqc_server_{i}.iqoala",f"programs/local_prog/local_prog_scen{params['scenario']}_self.iqoala"] for i in program_sizes]
        params["server_progs_coop"] = [[f"programs/bqc/vbqc_server_{i}.iqoala",f"programs/local_prog/local_prog_scen{params['scenario']}_coop.iqoala"] for i in program_sizes]
        
        params["random_client_inputs"] = True
        params["client_input_func"] = [[client_input_func[f"bqc_inputs_{i}"]] for i in program_sizes]
        params["client_prog_args"] = [[{}] for i in program_sizes]

        params["random_server_inputs"] = False
        params["server_input_func"] = [[] for i in program_sizes]
        params["server_prog_args"] = [[{},{"iterations":200}] for i in program_sizes]

        params["compute_succ_probs"] = [[succ_prob_func[f"bqc_compute_succ_prob_{i}"], local_compute_succ_prob_generator(params['scenario'])] for i in program_sizes]
    elif params["scenario"] == 3:
        params["client_progs"] = [[f"programs/bqc/vbqc_client_{i}.iqoala"] for i in program_sizes]
        
        params["server_progs_self"] = [[f"programs/bqc/vbqc_server_{i}_selfish.iqoala"] for i in program_sizes]
        params["server_progs_coop"] = [[f"programs/bqc/vbqc_server_{i}_1coop.iqoala"] for i in program_sizes]
        
        params["random_client_inputs"] = True
        params["client_input_func"] = [[client_input_func[f"bqc_inputs_{i}"]] for i in program_sizes]
        params["client_prog_args"] = [[{}] for i in program_sizes]

        params["random_server_inputs"] = False
        params["server_input_func"] = [[] for i in program_sizes]
        params["server_prog_args"] = [[{}] for i in program_sizes]

        params["compute_succ_probs"] = [[succ_prob_func[f"bqc_compute_succ_prob_{i}"]] for i in program_sizes]

    datapoints: List[DataPoint] = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()
    for prog_size_index in range(len(program_sizes)):
        print(f"Program size: {program_sizes[prog_size_index]}")
        # for num_clients in range(2,params["num_clients"]+1):
            # print(f"Num Clients: {num_clients}")
        num_qubits = num_clients*program_sizes[prog_size_index] + 1
        for param_val in param_vals:
            print(f"Results for {param_name} : {params[param_name]}")
            params[param_name] = param_val 
            hardware = params["hardware"]
            params["cc"] = ent_rate.cc_time(distance=params["distance"])
            
            if hardware == "TI":
                link_duration, link_fid = ent_rate.trapped_ion_epr(params["distance"], QIA_SGA=params["qia_sga"])
                params["link_duration"] = link_duration
                if param_name=="single_gate_fid":
                    params["link_fid"] = 1 
            elif hardware == "NV":
                link_duration, link_fid = ent_rate.nv_epr(params["distance"], optimism=params["qia_sga"])
                params["link_duration"] = link_duration
                if param_name=="single_gate_fid":
                    params["link_fid"] = 1

            # Need to keep track of success probability and makespan for BQC and local prog 
            avg_self_bqc_makespan = 0
            avg_self_local_makespan = 0
            avg_self_bqc_succ_prob = 0
            avg_self_local_succ_prob = 0
            avg_coop_bqc_makespan = 0
            avg_coop_local_makespan = 0
            avg_coop_bqc_succ_prob = 0
            avg_coop_local_succ_prob = 0
            for trial in range(num_trials):
                # Run selfish version
                selfish_total_makespan, selfish_succ_probs, selfish_makespans = run_eval_exp(
                    client_progs=params["client_progs"][prog_size_index],
                    server_progs=params["server_progs_self"][prog_size_index],
                    client_prog_args=params["client_prog_args"][prog_size_index],
                    server_prog_args=params["server_prog_args"][prog_size_index],
                    client_num_iterations=params["client_num_iterations"],
                    server_num_iterations=params["server_num_iterations"],
                    num_clients=num_clients,
                    server_num_qubits=num_qubits,
                    client_num_qubits=params["client_num_qubits"],
                    linear=params["linear"],
                    use_netschedule=params["use_netschedule"],
                    bin_length=params["bin_length"]*params["link_duration"]+params["bin_length"],
                    cc=params["cc"],
                    t1=params["t1"],
                    t2=params["t2"],
                    host_instr_time=params["host_instr_time"],
                    host_peer_latency=params["host_peer_latency"],
                    qnos_instr_proc_time=params["qnos_instr_proc_time"],
                    internal_sched_latency=params["internal_sched_latency"],
                    single_gate_dur=params["single_gate_dur"],
                    two_gate_dur=params["two_gate_dur"],
                    single_gate_fid=params["single_gate_fid"],
                    two_gate_fid=params["two_gate_fid"],
                    link_duration=params["link_duration"],
                    link_fid=params["link_fid"],
                    seed = seed_value+trial,
                    client_input_func=params["client_input_func"][prog_size_index],
                    random_client_inputs=True,
                    compute_succ_probs=params["compute_succ_probs"][prog_size_index],
                )
                print(f"Selfish Results\t Total Makespan:{selfish_total_makespan}\tMakespan: {selfish_makespans}\tSuccess Prob: {selfish_succ_probs}")

                # Run coop version
                coop_total_makespan, coop_succ_probs, coop_makespans = run_eval_exp(
                    client_progs=params["client_progs"][prog_size_index],
                    server_progs=params["server_progs_coop"][prog_size_index],
                    client_prog_args=params["client_prog_args"][prog_size_index],
                    server_prog_args=params["server_prog_args"][prog_size_index],
                    client_num_iterations=params["client_num_iterations"],
                    server_num_iterations=params["server_num_iterations"],
                    num_clients=num_clients,
                    server_num_qubits=num_qubits,
                    client_num_qubits=params["client_num_qubits"],
                    linear=params["linear"],
                    use_netschedule=params["use_netschedule"],
                    bin_length=params["bin_length"]*params["link_duration"]+params["bin_length"],
                    cc=params["cc"],
                    t1=params["t1"],
                    t2=params["t2"],
                    host_instr_time=params["host_instr_time"],
                    host_peer_latency=params["host_peer_latency"],
                    qnos_instr_proc_time=params["qnos_instr_proc_time"],
                    internal_sched_latency=params["internal_sched_latency"],
                    single_gate_dur=params["single_gate_dur"],
                    two_gate_dur=params["two_gate_dur"],
                    single_gate_fid=params["single_gate_fid"],
                    two_gate_fid=params["two_gate_fid"],
                    link_duration=params["link_duration"],
                    link_fid=params["link_fid"],
                    seed = seed_value+trial,
                    client_input_func=params["client_input_func"][prog_size_index],
                    random_client_inputs=True,
                    compute_succ_probs=params["compute_succ_probs"][prog_size_index],
                )
                print(f"Cooperative Results\tTotal Makespan: {coop_total_makespan}\tMakespan: {coop_makespans}\tSuccess Prob: {coop_succ_probs}")

                # Compute the average success probability and makespan for local and bqc programs

                for key in selfish_makespans.keys():
                    # This is the local program results
                    if key[0] == -1:
                        avg_self_local_makespan += selfish_makespans[key]
                        avg_coop_local_makespan += coop_makespans[key]
                        avg_self_local_succ_prob += selfish_succ_probs[key]
                        avg_coop_local_succ_prob += coop_succ_probs[key]
                    # This is one of the bqc program results
                    else:
                        avg_self_bqc_makespan += selfish_makespans[key]
                        avg_coop_bqc_makespan += coop_makespans[key]
                        avg_self_bqc_succ_prob += selfish_succ_probs[key]
                        avg_coop_bqc_succ_prob += coop_succ_probs[key]

            avg_self_bqc_makespan /= (num_trials*num_clients) 
            avg_self_local_makespan /= num_trials 
            avg_self_bqc_succ_prob /= (num_trials*num_clients)
            avg_self_local_succ_prob /= num_trials
            avg_coop_bqc_makespan /= (num_trials*num_clients)
            avg_coop_local_makespan /= num_trials
            avg_coop_bqc_succ_prob /= (num_trials*num_clients)
            avg_coop_local_succ_prob /= num_trials
            datapoints.append(DataPoint(
                selfish_bqc_makespan=avg_self_bqc_makespan,
                selfish_local_makespan=avg_self_local_makespan,
                cooperative_bqc_makespan=avg_coop_bqc_makespan,
                cooperative_local_makespan=avg_coop_local_makespan,
                selfish_bqc_succ_prob=avg_self_bqc_succ_prob,
                selfish_local_succ_prob=avg_self_local_succ_prob,
                cooperative_bqc_succ_prob=avg_coop_bqc_succ_prob,
                cooperative_local_succ_prob=avg_coop_local_succ_prob,
                prog_size=program_sizes[prog_size_index],
                num_clients=num_clients,
                param_name=param_name,
                param_value=param_val
            ))
            print(datapoints[-1])

    
    # Finish computing how long the experiment took to run
    end = time.time()
    duration = round(end - start, 2)

    # compute the path to the directory for storing data
    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}_{param_name}_scen{params['scenario']}_{params['hardware']}_numclients{num_clients}_seed{seed}.json")

    metadata = DataMeta(
        timestamp=timestamp,
        sim_duration=duration,
        scenario=params["scenario"],
        hardware=params["hardware"],
        qia_sga=params["qia_sga"],
        prog_sizes=program_sizes,
        num_iterations=params["server_num_iterations"],
        num_trials=num_trials,
        linear=params["linear"],
        cc=params["cc"],
        t1=params["t1"],
        t2=params["t2"],
        single_gate_dur=params["single_gate_dur"],
        two_gate_dur=params["two_gate_dur"],
        all_gate_dur=params["all_gate_dur"],
        single_gate_fid=params["single_gate_fid"],
        two_gate_fid=params["two_gate_fid"],
        all_gate_fid=params["all_gate_fid"],
        qnos_instr_proc_time=params["qnos_instr_proc_time"],
        host_instr_time=params["host_instr_time"],
        host_peer_latency=params["host_peer_latency"],
        internal_sched_latency=params["internal_sched_latency"],
        link_duration=params["link_duration"],
        link_fid=params["link_fid"],
        client_num_qubits=params["client_num_qubits"],
        server_num_qubits=num_qubits,
        param_name=param_name,
        seed = seed_value,
        use_netschedule=params["use_netschedule"],
        bin_length=params["bin_length"]
    )

    # Format the metadata and datapoints into a json object
    data = Data(meta=metadata, data_points=datapoints)
    json_data = asdict(data)

    if save:
        # Write the data
        with open(last_path, "w") as datafile:
            json.dump(json_data, datafile)
        with open(timestamp_path, "w") as datafile:
            json.dump(json_data, datafile)