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
    topology = TopologyConfig.perfect_config_uniform_default_params(1)
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


def test_simple_program():
    ns.sim_reset()

    num_iterations = 100

    node_cfg = get_config()
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("simple_program.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
        linear=True,
    )

    all_results = app_results.batch_results["alice"].results
    outcomes = [result.values["m"] for result in all_results]
    print(outcomes)
    assert all(outcome == 1 for outcome in outcomes)


def test_return_vector():
    ns.sim_reset()

    num_iterations = 100

    node_cfg = get_config()
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("return_vector.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
    )

    all_results = app_results.batch_results["alice"].results
    outcomes = [result.values["outcomes"] for result in all_results]
    print(outcomes)
    assert all(outcome == [1, 1, 0] for outcome in outcomes)


def test_return_vector_loop():
    ns.sim_reset()

    num_iterations = 1

    node_cfg = get_config()
    network_cfg = ProcNodeNetworkConfig(nodes=[node_cfg], links=[])
    program = load_program("return_vector_loop.iqoala")

    app_results = run_single_node_app(
        num_iterations=num_iterations,
        program_name="alice",
        program=program,
        program_input=ProgramInput.empty(),
        network_cfg=network_cfg,
    )

    all_results = app_results.batch_results["alice"].results
    outcomes = [result.values["outcomes"] for result in all_results]
    print(outcomes)
    expected = [1] * 100
    assert all(outcome == expected for outcome in outcomes)


if __name__ == "__main__":
    test_simple_program()
    test_return_vector()
    test_return_vector_loop()
