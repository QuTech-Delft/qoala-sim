"""Examples where the memory topology is assumed to be fully connected.
All qubits can interact with 2-qubit gates and used as connection or storage qubits"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

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
from qoala.runtime.program import BatchResult, IteratedProgram, ProgramInput
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

def create_network_config(node_name: str, num_qubits: int) -> ProcNodeNetworkConfig:
    alice_node_cfg = create_procnode_cfg(node_name, 0, num_qubits)

    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        [alice_node_cfg], link_duration=1000
    )
    return network_cfg


@dataclass
class QkdResult:
    alice_result: BatchResult
    bob_result: BatchResult

def simple_deadlock():
    num_iterations = 2  # > 1 to create deadlock
    num_qubits = 2  # Program allocates 2 qubits
    node_name = "alice"

    alice_file = "2_qubits_local_only.iqoala"

    network_cfg = create_network_config(node_name, num_qubits)

    alice_program_w_inputs = IteratedProgram.from_input_copies(
        load_program(alice_file), ProgramInput.empty(), num_iterations
    )

    runner = BatchRunner(network_cfg, num_iterations)
    runner.register_program(node_name, alice_program_w_inputs)

    results = runner.simulate_batches()

    print("==== Deadlock example completed ====")
    print(results)


def setup_logger() -> str:
    logs_folder = Path("logs")
    logs_folder.mkdir(exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(logs_folder / f"deadlock_just_local_{now}.log")

    LogManager.set_log_level("DEBUG")
    LogManager.set_task_log_level("DEBUG")
    LogManager.log_to_file(log_file)
    LogManager.log_tasks_to_file(log_file)

    return log_file


if __name__ == "__main__":
    log_file = setup_logger()

    simple_deadlock()

    print("\nLogging to file:", log_file)
