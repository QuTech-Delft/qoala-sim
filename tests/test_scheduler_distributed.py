"""
The test cases on this class try to trigger a border case in the CPU scheduler of
the server program, where the scheduler is waiting for a message (a pydynaa event)
from bob, but fails to wait (for the correct pydynaa event) since there is a message
from alice available. The bug triggers due to the fact that waiting for the message
from bob would allow other tasks to execute, but there is literally nothing else
to execute.
"""

import random
from dataclasses import dataclass
from typing import Dict, List

import netsquid as ns
from netsquid.util import simlog

from qoala.lang.ehi import UnitModule
from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    ClassicalConnectionConfig,
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchInfo, BatchResult, ProgramInput
from qoala.runtime.statistics import SchedulerStatistics
from qoala.runtime.task import TaskGraph
from qoala.runtime.taskbuilder import TaskGraphBuilder
from qoala.sim.build import build_network_from_config
from qoala.util.logging import LogManager


@dataclass
class AppResult:
    batch_results: Dict[int, Dict[str, BatchResult]]
    statistics: Dict[str, SchedulerStatistics]
    total_duration: float


# IMPORTANT: In all the programs here; timing is REALLY important
# These programs (including the network latency configuration)
# try to replicate this scenario (Similar numbers happen "simultaneously"):
# 1. Alice sends 1 message
# 1. Bob sends 1 message
# 1. Server starts executing a long block
# 2. Alice executes something, delaying a bit
# 2: Bob returns
# 3. Alice sends a second message
# 4. Alice returns
# 5. Server receives (in queue) Bob's and Alice first message
# 6. Server reads Alice's first message
# 7. Server tries to read Alice's second message
# 8. Since there is a message in the queue (Bob's), qoala-sim stalls the execution
# 9. Alice's second message arrives
# 10. Server reads Bob's message
# 11. Server returns


def get_client_alice_program() -> QoalaProgram:
    program_text = """
META_START
    name: alice
    parameters: server_id
    csockets: 0 -> server
    epr_sockets:
META_END

^b0 {type = CL}:
    csock = assign_cval() : 0
    ret = assign_cval() : 0
    msg = assign_cval() : 3
    send_cmsg(csock, msg)
^b1 {type = QL}:
    // Alice does something between the first and second msg
    run_subroutine() : delay_subroutine
^b2 {type = CL}:
    msg = assign_cval() : 5
    send_cmsg(csock, msg)
    return_result(ret)
SUBROUTINE delay_subroutine
params:
returns:
uses: 1
keeps:
request:
NETQASM_START
set Q0 1
init Q0
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
NETQASM_END
"""
    return QoalaParser(program_text).parse()


def get_client_bob_program() -> QoalaProgram:
    program_text = """
META_START
    name: bob
    parameters: server_id
    csockets: 0 -> server
    epr_sockets:
META_END

^b1 {type = CL}:
    csock = assign_cval() : 0
    ret = assign_cval() : 0
    msg = assign_cval() : 7
    // Bob sends one single message, immediately
    send_cmsg(csock, msg)
    return_result(ret)
SUBROUTINE delay_subroutine
params:
returns:
uses: 1
keeps:
request:
NETQASM_START
set Q0 1
init Q0
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
NETQASM_END
"""
    return QoalaParser(program_text).parse()


def get_server_program() -> QoalaProgram:
    program_text = """
META_START
    name: server
    parameters: alice_id, bob_id
    csockets: 0 -> alice, 1 -> bob
    epr_sockets:
META_END

^b0 {type = QL}:
    run_subroutine() : delay_start
^b1 {type = CL}:
    zero = assign_cval() : 0
    csock = assign_cval() : 0
^b2 {type = CC}:
    // Server receives a single message from Alice - The message
    // should be on the receive queue when this recv is executed
    p0 = recv_cmsg(csock)
^b3 {type = CL}:
    add_a = add_cval_c(p0, zero)
^b4 {type = QL}:
    // Then it does something, quite long - In this time, the message
    // from bob should arrive
    tuple<m0> = run_subroutine() : long_subroutine
^b5 {type = CL}:
    csock = assign_cval() : 0
// By this time, both alice and bob should have finished executing, BUT
// Alice's second message is still "in transmission".
^b6 {type = CC}:
    // This time, we try to receive a message from alice. This should put
    // the server to sleep until a message from alice arrives. However,
    // the message has NOT arrived yet. Despite this, there is a message
    // which arrived... from bob. This leads the qoala sim to yield
    // erronously on a different pydynaa event.
    p1 = recv_cmsg(csock)
^b7 {type = CL}:
    csock = assign_cval() : 1
^b8 {type = CC}:
    // This receives a message from bob. This call should NOT put the
    // server to sleep, since the message form bob should have arrived
    // even before receiving the last message from Alice
    p2 = recv_cmsg(csock)
^b9 {type = CL}:
    add_d = add_cval_c(p2, add_a)
^b10 {type = CL}:
    ret = assign_cval() : 3
    return_result(ret)
SUBROUTINE delay_start
params:
returns:
uses: 1
keeps:
request:
NETQASM_START
set Q0 1
init Q0
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
x Q0
rot_x Q0 1 2
rot_z Q0 3 2
rot_y Q0 2 2
NETQASM_END
SUBROUTINE long_subroutine
params:
returns: m0
uses: 1
keeps:
request:
NETQASM_START
set Q0 1
init Q0
x Q0
rot_x Q0 1 2
store M0 @output[0]
NETQASM_END
"""
    return QoalaParser(program_text).parse()


def run_n_clients(
    node_names: List[str],
    node_programs: Dict[str, QoalaProgram],
    node_inputs: Dict[str, List[ProgramInput]],
    network_cfg: ProcNodeNetworkConfig,
    interactions: Dict[str, List[str]],
    num_iterations: int = 1,
    num_batches: int = 1,
    linear: bool = False,
    manual_sched: bool = False,
) -> AppResult:
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    seed = random.randint(0, 1000000)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(network_cfg)

    remotes = {name: [] for name in node_names}

    for name in node_names:
        procnode = network.nodes[name]
        unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
        for _ in range(num_batches):
            batch_info = BatchInfo(
                node_programs[name],
                unit_module,
                node_inputs[name],
                num_iterations,
                deadline=1e8,
            )
            batch = procnode.submit_batch(batch_info)
            remotes[name].append(batch)

    for name in node_names:
        remote_pids = {}
        for i in range(num_batches):
            pids = []
            for remote in interactions[name]:
                pids.append(remotes[remote][i].instances[0].pid)
            remote_pids[remotes[name][i].batch_id] = pids
        network.nodes[name].initialize_processes(remote_pids=remote_pids, linear=linear)

    if manual_sched:
        # If manual scheduling is enabled, we need to create the task graph for each node
        nodes_counter = {n: 0 for n in node_names}
        nodes_task: Dict[str, List[TaskGraph]] = {n: [] for n in node_names}
        for node in node_names:
            for i, batch in enumerate(remotes[node]):
                tasks = TaskGraphBuilder.from_program(
                    node_programs[node],
                    batch.instances[0].pid,
                    network.nodes[node].local_ehi,
                    network.nodes[node].network_ehi,
                    first_task_id=nodes_counter[node],
                    prog_input=node_inputs[node][0].values,
                )
                nodes_task[node].append(tasks)
                nodes_counter[node] += len(tasks.get_tasks())
            node_full_graph = TaskGraphBuilder.merge_linear(nodes_task[node])
            # And also submit it to the scheduler
            network.nodes[node].scheduler.upload_task_graph(node_full_graph)

    network.start()
    start_time = ns.sim_time()
    ns.sim_run(end_time=1000000, magnitude=ns.SECOND)
    end_time = ns.sim_time()
    makespan = end_time - start_time

    results: Dict[int, Dict[str, BatchResult]] = {0: {}}
    statistics: Dict[str, SchedulerStatistics] = {}

    for name in node_names:
        procnode = network.nodes[name]
        results[0][name] = procnode.scheduler.get_batch_results()[0]
        statistics[name] = procnode.scheduler.get_statistics()

    return AppResult(results, statistics, makespan)


def configure_network(
    hw_num_qubits: int, nodes_spec: Dict[int, str], manual_sched: bool
) -> ProcNodeNetworkConfig:
    node_configs = []
    for node_id, name in nodes_spec.items():
        node_configs.append(
            ProcNodeConfig(
                node_name=name,
                node_id=node_id,
                topology=TopologyConfig.perfect_config_uniform_default_params(
                    hw_num_qubits
                ),
                latencies=LatenciesConfig(qnos_instr_time=1e4),
                ntf=NtfConfig.from_cls_name("GenericNtf"),
                determ_sched=True,
                is_predictable=manual_sched,
            )
        )
    # Communication links alice <-> server, bob <-> server
    connections = [
        # Since timing is important, the latencies are pretty important
        # If these values are are changed, the programs for Alice, Bob and the server
        # NEED to be adjusted to recreate the timing issue.
        ClassicalConnectionConfig.from_nodes(0, 2, 1e4),
        ClassicalConnectionConfig.from_nodes(1, 2, 1e4),
    ]

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=node_configs, link_duration=1000
    )
    network_cfg.cconns = connections

    return network_cfg


def test_automatic_scheduling():
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("DEBUG")
    simlog.logger.setLevel("DEBUG")

    hw_num_qubits = 3
    nodes_spec = {0: "alice", 1: "bob", 2: "server"}
    nodes_programs = {
        "alice": get_client_alice_program(),
        "bob": get_client_bob_program(),
        "server": get_server_program(),
    }
    nodes_interactions = {
        "alice": ["server"],
        "bob": ["server"],
        "server": ["alice", "bob"],
    }

    nodes_inputs = {
        "alice": [ProgramInput({"server_id": 2})],
        "bob": [ProgramInput({"server_id": 2})],
        "server": [ProgramInput({"alice_id": 0, "bob_id": 1})],
    }

    network_cfg = configure_network(hw_num_qubits, nodes_spec, manual_sched=False)

    app_result = run_n_clients(
        node_names=list(nodes_spec.values()),
        node_programs=nodes_programs,
        node_inputs=nodes_inputs,
        network_cfg=network_cfg,
        interactions=nodes_interactions,
    )

    print(app_result)
    assert app_result.batch_results is not None
    assert app_result.batch_results[0]["alice"].results[0].values["ret"] == 0
    assert app_result.batch_results[0]["bob"].results[0].values["ret"] == 0
    assert app_result.batch_results[0]["server"].results[0].values["ret"] == 3


def test_manual_scheduling():
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("DEBUG")
    simlog.logger.setLevel("DEBUG")

    hw_num_qubits = 3
    nodes_spec = {0: "alice", 1: "bob", 2: "server"}
    nodes_programs = {
        "alice": get_client_alice_program(),
        "bob": get_client_bob_program(),
        "server": get_server_program(),
    }
    nodes_interactions = {
        "alice": ["server"],
        "bob": ["server"],
        "server": ["alice", "bob"],
    }

    nodes_inputs = {
        "alice": [ProgramInput({"server_id": 2})],
        "bob": [ProgramInput({"server_id": 2})],
        "server": [ProgramInput({"alice_id": 0, "bob_id": 1})],
    }

    network_cfg = configure_network(hw_num_qubits, nodes_spec, manual_sched=True)

    app_result = run_n_clients(
        node_names=list(nodes_spec.values()),
        node_programs=nodes_programs,
        node_inputs=nodes_inputs,
        network_cfg=network_cfg,
        interactions=nodes_interactions,
        manual_sched=True,
    )

    print(app_result)
    assert app_result.batch_results is not None
    assert app_result.batch_results[0]["alice"].results[0].values["ret"] == 0
    assert app_result.batch_results[0]["bob"].results[0].values["ret"] == 0
    assert app_result.batch_results[0]["server"].results[0].values["ret"] == 3
