from __future__ import annotations

import os

import netsquid as ns

from qoala.lang.parse import QoalaParser
from qoala.lang.program import QoalaProgram
from qoala.runtime.config import (
    LatenciesConfig,
    NtfConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
    TopologyConfig,
)
from qoala.runtime.program import ProgramInput
from qoala.util.runner import run_single_node_app


def get_config() -> ProcNodeConfig:
    topology = TopologyConfig.perfect_config_uniform_default_params(5)
    return ProcNodeConfig(
        node_name="alice",
        node_id=0,
        topology=topology,
        latencies=LatenciesConfig(qnos_instr_time=1000),
        ntf=NtfConfig.from_cls_name("GenericNtf"),
    )


def load_program(name: str) -> QoalaProgram:
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path) as file:
        text = file.read()
    program = QoalaParser(text).parse()

    return program


def test_qft():
    ns.sim_reset()

    num_iterations = 1

    node_cfg = get_config()
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("qft.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
        linear=True,
    )
    print(app_results)


if __name__ == "__main__":
    test_qft()
