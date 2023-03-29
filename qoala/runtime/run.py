from __future__ import annotations

from typing import Dict, List

import netsquid as ns

# Ignore type since whole 'config' module is ignored by mypy
from qoala.runtime.config import ProcNodeNetworkConfig  # type: ignore
from qoala.runtime.context import SimulationContext
from qoala.runtime.environment import NetworkEhi
from qoala.runtime.program import BatchResult, ProgramInstance
from qoala.sim.build import build_network
from qoala.sim.globals import GlobalSimData
from qoala.sim.network import ProcNodeNetwork


def _run(network: ProcNodeNetwork) -> List[Dict[int, BatchResult]]:
    """Run the protocols of a network and programs running in that network.

    :param network: `ProcNodeNetwork` representing the nodes and links
    :return: final results of the programs
    """

    # Start all the protocols.
    network.start()

    # Start the NetSquid simulation.
    ns.sim_run()

    return [node.scheduler.get_batch_results() for _, node in network.nodes.items()]


def run(
    config: ProcNodeNetworkConfig,
    programs: Dict[str, List[ProgramInstance]],
    # schedules: Optional[Dict[str, Schedule]] = None,
    num_times: int = 1,
) -> List[Dict[int, BatchResult]]:
    """Run programs on a network specified by a network configuration.

    :param config: configuration of the network
    :param programs: dictionary of node names to programs
    :param num_times: numbers of times to run the programs, defaults to 1
    :return: program results
    """
    # Create global runtime environment.
    # TODO: use new way of creating NetworkEhi objects
    rte = NetworkEhi.with_nodes_no_links({})

    # Build the network. Info about created nodes will be added to the runtime environment.
    network = build_network(config, rte)

    ###########################################
    # TODO: pass context to simulation objects!
    ###########################################
    sim_data = GlobalSimData()
    sim_data.set_network(network)
    context = SimulationContext(network_ehi=rte, global_sim_data=sim_data)
    print(context)

    for name, program_list in programs.items():
        for program in program_list:
            network.nodes[name]._local_env.register_program(program)

    # if schedules is not None:
    #     for name, schedule in schedules.items():
    #         network.nodes[name]._local_env.install_local_schedule(schedule)

    for name in programs.keys():
        network.nodes[name].install_environment()

    results = _run(network)
    return results
