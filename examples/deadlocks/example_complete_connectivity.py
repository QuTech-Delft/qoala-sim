"""Examples where the memory topology is assumed to be fully connected.
All qubits can interact with 2-qubit gates and used as connection or storage qubits"""

from __future__ import annotations

import os
from dataclasses import dataclass

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NetworkScheduleConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import BatchResult, ProgramCopies, ProgramInput
from qoala.util.logging import LogManager
from qoala.util.runner import BatchRunner


def create_procnode_cfg(name: str, id: int, num_qubits: int) -> ProcNodeConfig:
    return ProcNodeConfig(
        node_name=name,
        node_id=id,
        topology=TopologyConfig.perfect_config_uniform_default_params(num_qubits),
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(path: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path) as file:
        text = file.read()
    return QoalaParser(text).parse()


@dataclass
class QkdResult:
    alice_result: BatchResult
    bob_result: BatchResult


def run_deadlock(
    num_iterations: int,
    alice1_file: str,
    alice2_file: str,
    bob1_file: str,
    bob2_file: str,
):
    alice_id = 0
    bob_id = 1
    num_qubits = 2

    alice_node_cfg = create_procnode_cfg("alice", alice_id, num_qubits)
    bob_node_cfg = create_procnode_cfg("bob", bob_id, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        [alice_node_cfg, bob_node_cfg], link_duration=1000
    )

    pattern = [(alice_id, i, bob_id, i) for i in range(num_iterations)]
    NetworkScheduleConfig(
        bin_length=1_500, first_bin=0, bin_pattern=pattern, repeat_period=20_000
    )

    alice1_program = load_program(alice1_file)
    alice2_program = load_program(alice2_file)
    bob1_program = load_program(bob1_file)
    bob2_program = load_program(bob2_file)

    alice1_input = ProgramInput({"bob_id": bob_id})
    alice2_input = ProgramInput({"bob_id": bob_id})
    bob1_input = ProgramInput({"alice_id": alice_id})
    bob2_input = ProgramInput({"alice_id": alice_id})

    # Runner
    runner = BatchRunner(network_cfg, num_iterations)
    alice1_program_w_inputs = ProgramCopies.from_input_copies(
        alice1_program, alice1_input, num_iterations
    )
    alice2_program_w_inputs = ProgramCopies.from_input_copies(
        alice2_program, alice2_input, num_iterations
    )
    bob1_program_w_inputs = ProgramCopies.from_input_copies(
        bob1_program, bob1_input, num_iterations
    )
    bob2_program_w_inputs = ProgramCopies.from_input_copies(
        bob2_program, bob2_input, num_iterations
    )
    runner.register_program("alice", alice1_program_w_inputs)
    runner.register_program("alice", alice2_program_w_inputs)
    runner.register_program("bob", bob1_program_w_inputs)
    runner.register_program("bob", bob2_program_w_inputs)


    results = runner.simulate_batches()
    return results


def simple_deadlock():
    num_iterations = 1
    alice1_file = "alice1_2_qubits.iqoala"
    alice2_file = "alice2_2_qubits.iqoala"
    bob1_file = "bob1_2_qubits.iqoala"
    bob2_file = "bob2_2_qubits.iqoala"

    results = run_deadlock(
        num_iterations,
        alice1_file,
        alice2_file,
        bob1_file,
        bob2_file,
    )

    print("Deadlock example completed unsuccessfully")
    print(results)


if __name__ == "__main__":
    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("DEBUG")

    simple_deadlock()

