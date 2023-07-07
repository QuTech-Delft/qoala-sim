from __future__ import annotations

import os

from netqasm.lang.instr.flavour import NVFlavour

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    NvParams,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import ProgramInput
from qoala.util.runner import run_single_node_app


def get_config() -> ProcNodeConfig:
    params = NvParams()
    topology = TopologyConfig.from_nv_params(num_qubits=5, params=params)
    return ProcNodeConfig(
        node_name="alice",
        node_id=0,
        topology=topology,
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("NvNtf"),
    )


def load_program(name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text, flavour=NVFlavour()).parse()

    return program


def run_program(num_iterations: int, node_cfg: ProcNodeConfig):
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("move_qubit_nv.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
    )

    all_results = app_results.batch_results["alice"].results
    outcomes = [result.values["m"] for result in all_results]
    assert all(outcome == 1 for outcome in outcomes)


def test_move_qubit():
    node_cfg = get_config()
    run_program(10, node_cfg)


if __name__ == "__main__":
    test_move_qubit()
