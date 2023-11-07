from __future__ import annotations

import math
import os
import random
import time
from curses.ascii import alt
from dataclasses import dataclass
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


def get_client_config(id: int) -> ProcNodeConfig:
    # client only needs 1 qubit
    return ProcNodeConfig(
        node_name=f"client",
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(20),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def get_server_config(id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name="server",
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(
            host_instr_time=500, host_peer_latency=30_000, qnos_instr_time=1000
        ),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def create_network(
    server_cfg: ProcNodeConfig,
    client_cfg: ProcNodeConfig,
    num_iterations: int,
    qkd_num_iterations: int,
    cc: float,
    use_netschedule: bool,
    alternating: bool,
    bin_length: float,
) -> ProcNodeNetwork:
    node_cfgs = [server_cfg, client_cfg]
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_cfgs, link_duration=1000
    )

    pattern: List[Tuple[int, int, int, int]] = []
    if alternating:
        # start index of PIDs per app
        bqc_start = 0
        qkd_start = num_iterations

        bqc_idx = 0
        qkd_idx = 0

        while bqc_idx < num_iterations or qkd_idx < qkd_num_iterations:
            if bqc_idx < num_iterations:
                pattern.append((0, bqc_start + bqc_idx, 1, bqc_start + bqc_idx))
            if qkd_idx < qkd_num_iterations:
                pattern.append((0, qkd_start + qkd_idx, 1, qkd_start + qkd_idx))
            bqc_idx += 1
            qkd_idx += 1
    else:
        for i in range(num_iterations + qkd_num_iterations):
            pattern.append((0, i, 1, i))

    if use_netschedule:
        network_cfg.netschedule = NetworkScheduleConfig(
            bin_length=bin_length,
            first_bin=0,
            bin_pattern=pattern,
            repeat_period=bin_length * (num_iterations + qkd_num_iterations),
        )

    cconns = [ClassicalConnectionConfig.from_nodes(0, 1, cc)]
    network_cfg.cconns = cconns
    return build_network_from_config(network_cfg)


@dataclass
class BqcQkdResult:
    client_results: BatchResult
    alice_results: BatchResult
    bob_results: BatchResult


@dataclass
class BqcParams:
    alpha: int
    beta: int
    theta1: int
    theta2: int
    dummy0: int
    dummy1: int


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


def create_batch(
    program: QoalaProgram,
    inputs: List[ProgramInput],
    unit_module: UnitModule,
    num_iterations: int,
) -> BatchInfo:
    return BatchInfo(
        program=program,
        inputs=inputs,
        unit_module=unit_module,
        num_iterations=num_iterations,
        deadline=0,
    )


def create_client_inputs(
    bqc_params: BqcParams, server_id: int, num_iterations: int
) -> List[ProgramInput]:
    return [
        ProgramInput(
            {
                "server_id": server_id,
                "alpha": bqc_params.alpha,
                "beta": bqc_params.beta,
                "theta1": bqc_params.theta1,
                "theta2": bqc_params.theta2,
                "dummy0": bqc_params.dummy0,
                "dummy1": bqc_params.dummy1,
            }
        )
        for _ in range(num_iterations)
    ]


def run_bqc(
    bqc_params: BqcParams,
    num_iterations: int,
    qkd_num_iterations: int,
    qkd_num_pairs: int,
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    alternating: bool,
    bin_length: float,
) -> Tuple[BqcQkdResult, float]:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000)
    ns.set_random_state(seed=seed)

    # Create network
    server_id = 0
    client_id = 1
    server_config = get_server_config(id=server_id, num_qubits=server_num_qubits)
    client_config = get_client_config(id=client_id)

    network = create_network(
        server_config,
        client_config,
        num_iterations,
        qkd_num_iterations,
        cc,
        use_netschedule,
        alternating,
        bin_length,
    )

    # Load programs for client node
    client_procnode = network.nodes[f"client"]

    # Load BQC client program
    client_program = load_program("vbqc_client.iqoala")
    client_inputs = create_client_inputs(bqc_params, server_id, num_iterations)
    client_unit_module = UnitModule.from_full_ehi(client_procnode.memmgr.get_ehi())
    client_batch_info = create_batch(
        program=client_program,
        inputs=client_inputs,
        unit_module=client_unit_module,
        num_iterations=num_iterations,
    )
    client_batch = client_procnode.submit_batch(client_batch_info)

    # Load QKD alice program
    alice_program = load_program("qkd_npairs_MD_alice.iqoala")
    alice_inputs = [
        ProgramInput({"bob_id": server_id, "N": qkd_num_pairs})
        for _ in range(qkd_num_iterations)
    ]
    alice_unit_module = UnitModule.from_full_ehi(client_procnode.memmgr.get_ehi())
    alice_batch_info = create_batch(
        program=alice_program,
        inputs=alice_inputs,
        unit_module=alice_unit_module,
        num_iterations=qkd_num_iterations,
    )
    alice_batch = client_procnode.submit_batch(alice_batch_info)

    # Load programs for server node
    server_procnode = network.nodes["server"]

    # Load BQC server program
    server_program = load_program("vbqc_server.iqoala")
    server_inputs = [
        ProgramInput({"client_id": client_id}) for _ in range(num_iterations)
    ]
    server_unit_module = UnitModule.from_full_ehi(server_procnode.memmgr.get_ehi())
    server_batch_info = create_batch(
        program=server_program,
        inputs=server_inputs,
        unit_module=server_unit_module,
        num_iterations=num_iterations,
    )
    server_batch = server_procnode.submit_batch(server_batch_info)

    # Load QKD bob program
    bob_program = load_program("qkd_npairs_MD_bob.iqoala")
    bob_inputs = [
        ProgramInput({"alice_id": client_id, "N": qkd_num_pairs})
        for _ in range(qkd_num_iterations)
    ]
    bob_unit_module = UnitModule.from_full_ehi(server_procnode.memmgr.get_ehi())
    bob_batch_info = create_batch(
        program=bob_program,
        inputs=bob_inputs,
        unit_module=bob_unit_module,
        num_iterations=qkd_num_iterations,
    )
    bob_batch = server_procnode.submit_batch(bob_batch_info)

    # Match PIDs
    remote_pids_for_server_node = {
        server_batch.batch_id: [inst.pid for inst in client_batch.instances],
        bob_batch.batch_id: [inst.pid for inst in bob_batch.instances],
    }
    remote_pids_for_client_node = {
        client_batch.batch_id: [inst.pid for inst in server_batch.instances],
        alice_batch.batch_id: [inst.pid for inst in alice_batch.instances],
    }
    # print(f"remote PIDs for server node: {remote_pids_for_server_node}")
    # print(f"remote PIDs for client node: {remote_pids_for_client_node}")
    client_procnode.initialize_processes(remote_pids_for_client_node, linear=linear)
    server_procnode.initialize_processes(remote_pids_for_server_node, linear=linear)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run()
    end_time = ns.sim_time()
    makespan = end_time - start_time

    all_client_results = client_procnode.scheduler.get_batch_results()
    client_results = all_client_results[client_batch.batch_id]
    alice_results = all_client_results[alice_batch.batch_id]
    all_server_results = server_procnode.scheduler.get_batch_results()
    bob_results = all_server_results[bob_batch.batch_id]

    return BqcQkdResult(client_results, alice_results, bob_results), makespan


def bqc_computation(
    num_iterations: int,
    qkd_num_iterations: int,
    qkd_num_pairs: int,
    linear: bool,
    cc: float,
    server_num_qubits: int,
    use_netschedule: bool,
    alternating: bool,
    bin_length: float,
) -> Tuple[float, float]:
    ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.DM)

    bqc_params = BqcParams(alpha=8, beta=24, theta1=2, theta2=22, dummy0=0, dummy1=0)
    expected = 1

    bqc_result, makespan = run_bqc(
        bqc_params=bqc_params,
        num_iterations=num_iterations,
        qkd_num_iterations=qkd_num_iterations,
        qkd_num_pairs=qkd_num_pairs,
        linear=linear,
        cc=cc,
        server_num_qubits=server_num_qubits,
        use_netschedule=use_netschedule,
        alternating=alternating,
        bin_length=bin_length,
    )

    client_results = bqc_result.client_results
    alice_results = bqc_result.alice_results
    bob_results = bqc_result.bob_results

    m2s = [result.values["m2"] for result in client_results.results]
    correct_outcomes = len([m2 for m2 in m2s if m2 == expected])
    succ_prob = round(correct_outcomes / num_iterations, 2)

    alice_outcomes = [result.values["outcomes"][0] for result in alice_results.results]
    bob_outcomes = [result.values["outcomes"][0] for result in bob_results.results]
    # print(f"alice: {alice_outcomes}")
    # print(f"bob: {bob_outcomes}")
    assert alice_outcomes == bob_outcomes

    return makespan


if __name__ == "__main__":
    start = time.time()

    # LogManager.set_log_level("INFO")
    # LogManager.log_to_file("bqc_qkd.log")
    # LogManager.set_task_log_level("DEBUG")
    # LogManager.log_tasks_to_file("bqc_qkd_tasks.log")

    num_iterations = 50
    qkd_num_iterations = 50
    qkd_num_pairs = 1
    cc = 1e5
    num_qubits = 5
    bin_length = 5e4

    linear_makespan = bqc_computation(
        num_iterations,
        qkd_num_iterations=qkd_num_iterations,
        qkd_num_pairs=qkd_num_pairs,
        linear=True,
        cc=cc,
        server_num_qubits=num_qubits,
        use_netschedule=False,
        alternating=False,
        bin_length=bin_length,
    )
    print(f"linear: makespan: {linear_makespan:_}")

    interleaved_seq_makespan = bqc_computation(
        num_iterations,
        qkd_num_iterations=qkd_num_iterations,
        qkd_num_pairs=qkd_num_pairs,
        linear=False,
        cc=cc,
        server_num_qubits=num_qubits,
        use_netschedule=True,
        alternating=False,
        bin_length=bin_length,
    )
    print(f"interleaved, sequential bins: makespan: {interleaved_seq_makespan:_}")

    interleaved_alt_makespan = bqc_computation(
        num_iterations,
        qkd_num_iterations=qkd_num_iterations,
        qkd_num_pairs=qkd_num_pairs,
        linear=False,
        cc=cc,
        server_num_qubits=num_qubits,
        use_netschedule=True,
        alternating=True,
        bin_length=bin_length,
    )
    print(f"interleaved, alternating bins: makespan: {interleaved_alt_makespan:_}")

    seq_improv = round(1 - interleaved_seq_makespan / linear_makespan, 3)
    alt_improv = round(1 - interleaved_alt_makespan / linear_makespan, 3)
    print(f"improvement for sequential netschedule: {seq_improv}")
    print(f"improvement for alternating netschedule: {alt_improv}")

    end = time.time()
    print(f"duration: {round(end - start, 2)} s")
