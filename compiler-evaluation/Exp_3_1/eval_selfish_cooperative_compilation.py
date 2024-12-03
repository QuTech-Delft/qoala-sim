from __future__ import annotations

import datetime
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
    link_duration: int=1000,
) -> ProcNodeNetwork:
    assert len(client_configs) == num_clients

    node_cfgs = [server_cfg] + client_configs
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=link_duration
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

    # print(pattern)

    if use_netschedule:
        network_cfg.netschedule = NetworkScheduleConfig(
            bin_length=bin_length,
            first_bin=0,
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
    server_num_qubits: int = 2,
    client_num_qubits: int = 20,
    use_netschedule: bool = False,
    bin_length: int = 0,
    link_duration: int=1000,
    scheduling_alg: str = "",
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
    :return: _description_
    """
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    # seed = random.randint(0, 1000)
    ns.set_random_state(seed=1)

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
        )
        for i in range(1, num_clients + 1)
    ]

    # Create the network
    # needs rewrite...
    network = create_network(
        server_config,
        client_configs,
        num_clients,
        client_num_iterations,
        cc,
        use_netschedule,
        bin_length,
        link_duration,
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

    # Create the server program batches
    for prog_index in range(len(server_progs)):
        # This is a client<->server application
        if prog_index < len(client_progs):
            for client_id in range(1, num_clients + 1):
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
    server_num_qubits: int = 2,
    client_num_qubits: int = 20,
    use_netschedule: bool = False,
    bin_length: int = 0,
    link_duration: int=1000,
    scheduling_alg: str = "",
    compute_succ_probs: List[function] = [],
):
    exp_results = run_eval_programs(
        client_progs = client_progs,
        server_progs = server_progs,
        client_prog_args = client_prog_args,
        server_prog_args = server_prog_args,
        client_num_iterations = client_num_iterations,
        server_num_iterations = server_num_iterations,
        num_clients = num_clients,
        linear = linear,
        cc = cc,
        t1 = t1,
        t2 = t2,
        single_gate_dur = single_gate_dur,
        two_gate_dur = two_gate_dur,
        all_gate_dur = all_gate_dur,
        single_gate_fid = single_gate_fid,
        two_gate_fid = two_gate_fid,
        all_gate_fid = all_gate_fid,
        qnos_instr_proc_time = qnos_instr_proc_time,
        host_instr_time = host_instr_time,
        host_peer_latency = host_peer_latency,
        server_num_qubits = server_num_qubits,
        client_num_qubits = client_num_qubits,
        use_netschedule = use_netschedule,
        bin_length = bin_length,
        link_duration=link_duration,
        scheduling_alg = scheduling_alg
    ) 

    makespan = exp_results.total_duration

    # Need to compute success probabilities
    server_results = exp_results.server_results
    client_results = exp_results.client_results
    # print(client_results)
    

    # (client_id, program #) -> succ probability
    # A client_id of -1 corresponds to a server only program
    succ_probs : Dict[(int, int), float] = {}

    if compute_succ_probs is not None:
        for prog_index in range(len(compute_succ_probs)):
            # There will be one function to compute success probabilities for each program
            compute_succ_func = compute_succ_probs[prog_index]
            if not callable(compute_succ_func):
                continue

            # This function computes the success probs for a client-server app
            if prog_index < len(client_progs):
                # Need to compute the success probabilities for each client
                for client_index in range(0, num_clients):
                    # Results for client index + 1 for program prog_index
                    # Results for the client are indexed first by client, and then by batch
                    # The ith batch on the client is the ith program
                    client_prog_results = client_results[client_index][prog_index].results

                    # Need to get corresponding server program results
                    # The server results are a little different than the client results
                    # We first iterate over programs, and then clients
                    # So for two programs and two clients we have 
                    # batch 0: program 0, client 0
                    # batch 1: program 0, client 1
                    # batch 2: program 1, client 0
                    # batch 3: program 1, client 1
                    # So we have batch index = num_clients*program index + 1*client index
                    server_batch_index = num_clients*prog_index + client_index
                    server_prog_results = server_results[server_batch_index].results

                    # Compute the success probability and store it 
                    succ_probs[(client_index+1, prog_index+1)] = compute_succ_func(server_prog_results, client_prog_results, server_prog_args[prog_index], client_prog_args[prog_index])
                    
            # This function computes the success probs for a server only app
            else:
                #TODO
                continue
    
    return succ_probs, makespan

@dataclass
class DataPoint:
    num_qubits_server: int
    makespan_selfish: float
    makespan_cooperative: float
    succ_prob_selfish: float
    succ_prob_cooperative: float
    param_name: str # Name of param being varied
    param_value: float # Value of the varied param 

@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    client_progs: List[str]
    server_progs: List[str]
    client_prog_args: List[dict]
    server_prog_args: List[dict]
    client_num_iterations: List[int]
    server_num_iterations: List[int]
    num_clients: int
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
    client_num_qubits: int
    use_netschedule: bool
    bin_length: int 
    param_name: str # The parameter being varied

@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]

def bqc_compute_succ_prob(server_batch_results, client_batch_results, server_prog_args, client_prog_args):
    dummy0 = client_prog_args["dummy0"]

    num_iterations = len(client_batch_results)
    
    p1s = [result.values["p1"] for result in client_batch_results]
    p2s = [result.values["p2"] for result in client_batch_results]
    m1s = [result.values["m1"] for result in client_batch_results]
    m2s = [result.values["m2"] for result in client_batch_results]

    if dummy0 == 0:
        num_fails = len([(p, m) for (p, m) in zip(p1s, m2s) if p != m])
    else:  # dummy0 = 1
        num_fails = len([(p, m) for (p, m) in zip(p2s, m1s) if p != m])

    frac_fail = round(num_fails / num_iterations, 2)
    return (1 - frac_fail)


if __name__ == "__main__":
    # LogManager.set_log_level("DEBUG")
    # LogManager.log_to_file("eval_debug.log")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_tasks_to_file("eval_debug_TASKS.log")
    
    # Default values
    params = {
        "client_progs": ["programs/vbqc_client.iqoala", "programs/vbqc_client.iqoala"],
        "server_progs": ["programs/vbqc_server.iqoala", "programs/vbqc_server.iqoala"],
        "client_prog_args": [
                {
                    "alpha": 8,
                    "beta": 24,
                    "theta1": 2,
                    "theta2": 22,
                    "dummy0": 0,
                    "dummy1": 1,
                },
                {
                    "alpha": 8,
                    "beta": 24,
                    "theta1": 2,
                    "theta2": 22,
                    "dummy0": 0,
                    "dummy1": 1,
                },
            ],
        "server_prog_args": [{},{}],
        "client_num_iterations": [],
        "server_num_iterations": [],
        "num_clients": 1,
        "linear": False, 
        "cc": 1e5, 
        "t1": 1e9,
        "t2": 1e8,
        "single_gate_dur": 1e3,
        "two_gate_dur": 100e3,
        "all_gate_dur": 100e3, 
        "single_gate_fid": 1.0,
        "two_gate_fid": 1.0,
        "all_gate_fid": 1.0,
        "qnos_instr_proc_time": 1e3,
        "host_instr_time": 500,
        "host_peer_latency": 1e3,
        "client_num_qubits": 20,
        "use_netschedule": True,
        "bin_length": 1e5,
        "link_duration": 1e3,
        "compute_succ_probs": [bqc_compute_succ_prob, bqc_compute_succ_prob]
    }

    parser = ArgumentParser()
    parser.add_argument("--num_clients", "-c", type=int, required=True)
    parser.add_argument("--num_iterations", "-n", type=int, nargs="+", required=True)
    parser.add_argument("--default_params_file", type=str, required=False)
    parser.add_argument("--param_name", type=str, required=False)
    parser.add_argument("--param_values", type=float, nargs="+", required=False)
    parser.add_argument("--param_sweep_list", type=str, nargs="+", required=False)
    parser.add_argument("--save", action="store_true")

    # Parse arguments
    args = parser.parse_args()
    num_clients = args.num_clients
    num_iterations = args.num_iterations
    params_filename = args.default_params_file 
    param_name = args.param_name 
    param_vals = args.param_values 
    param_sweep_list = args.param_sweep_list
    save = args.save

    if params_filename is not None:
        # Load param values from .json file
        params_file = open(params_filename) 
        params_obj = json.load(params_file)
        for key, val in params_obj.items():
            print(key, val)
            params[key] = val 

    params["num_clients"] = num_clients
    params["client_num_iterations"] = num_iterations
    params["server_num_iterations"] = num_iterations

    if param_name is None:
        if param_sweep_list is not None:
            param_name = param_sweep_list[0]
            param_vals = [float(val) for val in param_sweep_list[1:]]
        else: 
            param_name = "cc"
            param_vals = [params["cc"]]
    
    datapoints: List[DataPoint] = [] 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start = time.time()
    for num_qubits in [2, 5, 10]:
        for param_val in param_vals:
            print(f"Num Qubits: {num_qubits}\t {param_name}: {param_val}")
            params[param_name] = param_val
            # Run selfish version
            params["client_progs"]=["programs/vbqc_client.iqoala"]
            params["server_progs"]=["programs/vbqc_server_selfish_block.iqoala"]
            selfish_succ_probs, selfish_makespan = run_eval_exp(
                client_progs=params["client_progs"],
                server_progs=params["server_progs"],
                client_prog_args=params["client_prog_args"],
                server_prog_args=params["server_prog_args"],
                client_num_iterations=params["client_num_iterations"],
                server_num_iterations=params["server_num_iterations"],
                num_clients=params["num_clients"],
                server_num_qubits=num_qubits,
                client_num_qubits=params["client_num_qubits"],
                linear=params["linear"],
                use_netschedule=params["use_netschedule"],
                bin_length=params["bin_length"],
                cc=params["cc"],
                t1=params["t1"],
                t2=params["t2"],
                host_instr_time=params["host_instr_time"],
                host_peer_latency=params["host_peer_latency"],
                qnos_instr_proc_time=params["qnos_instr_proc_time"],
                single_gate_dur=params["single_gate_dur"],
                two_gate_dur=params["two_gate_dur"],
                single_gate_fid=params["single_gate_fid"],
                two_gate_fid=params["two_gate_fid"],
                link_duration=params["link_duration"],
                compute_succ_probs=params["compute_succ_probs"]
            )
            print(f"Selfish Results\tMakespan: {selfish_makespan}\tSuccess Prob: {selfish_succ_probs}")

            # Run 1-cooperative version
            params["client_progs"]=["programs/vbqc_client.iqoala"]
            params["server_progs"]=["programs/vbqc_server_1cooperative_block.iqoala"]
            cooperative_succ_probs, cooperative_makespan = run_eval_exp(
                client_progs=params["client_progs"],
                server_progs=params["server_progs"],
                client_prog_args=params["client_prog_args"],
                server_prog_args=params["server_prog_args"],
                client_num_iterations=params["client_num_iterations"],
                server_num_iterations=params["server_num_iterations"],
                num_clients=params["num_clients"],
                server_num_qubits=num_qubits,
                client_num_qubits=params["client_num_qubits"],
                linear=params["linear"],
                use_netschedule=params["use_netschedule"],
                bin_length=params["bin_length"],
                cc=params["cc"],
                t1=params["t1"],
                t2=params["t2"],
                host_instr_time=params["host_instr_time"],
                host_peer_latency=params["host_peer_latency"],
                qnos_instr_proc_time=params["qnos_instr_proc_time"],
                single_gate_dur=params["single_gate_dur"],
                two_gate_dur=params["two_gate_dur"],
                single_gate_fid=params["single_gate_fid"],
                two_gate_fid=params["two_gate_fid"],
                link_duration=params["link_duration"],
                compute_succ_probs=params["compute_succ_probs"]
            )
            print(f"1-Cooperative Results\tMakespan: {cooperative_makespan}\tSuccess Prob: {cooperative_succ_probs}\n\n")
            
            # Average succ_probs for each program
            prog_selfish_succ_probs = []
            prog_cooperative_succ_probs = []
            for prog_index in range(len(params["server_progs"])):
                # This is a server-client app
                if prog_index < len(params["client_progs"]):
                    self_avg = 0
                    coop_avg = 0
                    for client_index in range(num_clients):
                        self_avg += selfish_succ_probs[(client_index+1,prog_index+1)]
                        coop_avg += cooperative_succ_probs[(client_index+1,prog_index+1)]
                    prog_selfish_succ_probs += [self_avg / num_clients]
                    prog_cooperative_succ_probs += [coop_avg / num_clients]
            
            # Create datapoint
            datapoints.append(DataPoint(
                num_qubits_server=num_qubits,
                makespan_selfish=selfish_makespan,
                makespan_cooperative=cooperative_makespan,
                succ_prob_selfish=prog_selfish_succ_probs[0],
                succ_prob_cooperative=prog_cooperative_succ_probs[0],
                param_name=param_name,
                param_value=param_val
            ))

    # Finish computing how long the experiment took to run
    end = time.time()
    duration = round(end - start, 2)

    # compute the path to the directory for storing data
    abs_dir = relative_to_cwd(f"data")
    Path(abs_dir).mkdir(parents=True, exist_ok=True)
    last_path = os.path.join(abs_dir, "LAST.json")
    timestamp_path = os.path.join(abs_dir, f"{timestamp}_{param_name}_nclients_{num_clients}.json")

    # Store the metadata about the experiment
    meta = DataMeta(
        client_progs=params["client_progs"],
        server_progs=params["server_progs"],
        client_prog_args=params["client_prog_args"],
        server_prog_args=params["server_prog_args"],
        client_num_iterations=params["client_num_iterations"],
        server_num_iterations=params["server_num_iterations"],
        num_clients=params["num_clients"],
        client_num_qubits=params["client_num_qubits"],
        linear=params["linear"],
        use_netschedule=params["use_netschedule"],
        bin_length=params["bin_length"],
        cc=params["cc"],
        t1=params["t1"],
        t2=params["t2"],
        host_instr_time=params["host_instr_time"],
        host_peer_latency=params["host_peer_latency"],
        qnos_instr_proc_time=params["qnos_instr_proc_time"],
        single_gate_dur=params["single_gate_dur"],
        two_gate_dur=params["two_gate_dur"],
        all_gate_dur=params["all_gate_dur"],
        single_gate_fid=params["single_gate_fid"],
        two_gate_fid=params["two_gate_fid"],
        all_gate_fid=params["all_gate_fid"],
        param_name=param_name,
        sim_duration=duration,
        timestamp=timestamp,
    )

    # Format the metadata and datapoints into a json object
    data = Data(meta=meta, data_points=datapoints)
    json_data = asdict(data)
    
    if save:
        # Write the data
        with open(last_path, "w") as datafile:
            json.dump(json_data, datafile)
        with open(timestamp_path, "w") as datafile:
            json.dump(json_data, datafile)
