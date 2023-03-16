import itertools
from typing import Dict, List

import numpy as np
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_CZ,
    INSTR_H,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
    INSTR_X,
    INSTR_Y,
    INSTR_Z,
)
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)
from netsquid.components.qprocessor import PhysicalInstruction, QuantumProcessor
from netsquid.qubits.operators import Operator
from netsquid_magic.link_layer import (
    MagicLinkLayerProtocolWithSignaling,
    SingleClickTranslationUnit,
)
from netsquid_magic.magic_distributor import (
    DepolariseWithFailureMagicDistributor,
    DoubleClickMagicDistributor,
    PerfectStateMagicDistributor,
)
from netsquid_magic.state_delivery_sampler import PerfectStateSamplerFactory
from netsquid_physlayer.heralded_connection import MiddleHeraldedConnection

# Ignore type since whole 'config' module is ignored by mypy
from qoala.runtime.config import (  # type: ignore
    DepolariseLinkConfig,
    GenericQDeviceConfig,
    HeraldedLinkConfig,
    LinkConfig,
    NVQDeviceConfig,
    ProcNodeConfig,
    ProcNodeNetworkConfig,
)
from qoala.runtime.environment import GlobalEnvironment
from qoala.runtime.lhi import LhiLatencies, LhiTopology, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface
from qoala.sim.entdist.entdist import EntDist
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.network import ProcNodeNetwork
from qoala.sim.procnode import ProcNode


# TODO: move this to somewhere else
def fidelity_to_prob_max_mixed(fid: float) -> float:
    return (1 - fid) * 4.0 / 3.0


def build_qprocessor_from_topology(
    name: str, topology: LhiTopology
) -> QuantumProcessor:
    num_qubits = len(topology.qubit_infos)

    mem_noise_models: List[QuantumErrorModel] = []
    for i in range(num_qubits):
        info = topology.qubit_infos[i]
        noise_model = info.error_model(**info.error_model_kwargs)
        mem_noise_models.append(noise_model)

    phys_instructions: List[PhysicalInstruction] = []
    # single-qubit gates
    for qubit_id, gate_infos in topology.single_gate_infos.items():
        for gate_info in gate_infos:
            phys_instr = PhysicalInstruction(
                instruction=gate_info.instruction,
                duration=gate_info.duration,
                topology=[qubit_id],
                quantum_noise_model=gate_info.error_model(
                    **gate_info.error_model_kwargs
                ),
            )
            phys_instructions.append(phys_instr)

    # multi-qubit gates
    for multi_qubit, gate_infos in topology.multi_gate_infos.items():
        qubit_ids = tuple(multi_qubit.qubit_ids)
        for gate_info in gate_infos:
            phys_instr = PhysicalInstruction(
                instruction=gate_info.instruction,
                duration=gate_info.duration,
                topology=[qubit_ids],
                quantum_noise_model=gate_info.error_model(
                    **gate_info.error_model_kwargs
                ),
            )
            phys_instructions.append(phys_instr)

    return QuantumProcessor(
        name=name,
        num_positions=num_qubits,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )


def build_generic_qprocessor(name: str, cfg: GenericQDeviceConfig) -> QuantumProcessor:
    phys_instructions = []

    single_qubit_gate_noise = DepolarNoiseModel(
        depolar_rate=cfg.single_qubit_gate_depolar_prob, time_independent=True
    )

    two_qubit_gate_noise = DepolarNoiseModel(
        depolar_rate=cfg.two_qubit_gate_depolar_prob, time_independent=True
    )

    phys_instructions.append(
        PhysicalInstruction(
            INSTR_INIT,
            duration=cfg.init_time,
        )
    )

    for instr in [
        INSTR_ROT_X,
        INSTR_ROT_Y,
        INSTR_ROT_Z,
        INSTR_X,
        INSTR_Y,
        INSTR_Z,
        INSTR_H,
    ]:
        phys_instructions.append(
            PhysicalInstruction(
                instr,
                quantum_noise_model=single_qubit_gate_noise,
                duration=cfg.single_qubit_gate_time,
            )
        )

    for instr in [INSTR_CNOT, INSTR_CZ]:
        phys_instructions.append(
            PhysicalInstruction(
                instr,
                quantum_noise_model=two_qubit_gate_noise,
                duration=cfg.two_qubit_gate_time,
            )
        )

    phys_instr_measure = PhysicalInstruction(
        INSTR_MEASURE,
        duration=cfg.measure_time,
    )
    phys_instructions.append(phys_instr_measure)

    electron_qubit_noise = T1T2NoiseModel(T1=cfg.T1, T2=cfg.T2)
    mem_noise_models = [electron_qubit_noise] * cfg.num_qubits
    qmem = QuantumProcessor(
        name=name,
        num_positions=cfg.num_qubits,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    return qmem


def build_nv_qprocessor(name: str, cfg: NVQDeviceConfig) -> QuantumProcessor:
    # noise models for single- and multi-qubit operations
    electron_init_noise = DepolarNoiseModel(
        depolar_rate=cfg.electron_init_depolar_prob, time_independent=True
    )

    electron_single_qubit_noise = DepolarNoiseModel(
        depolar_rate=cfg.electron_single_qubit_depolar_prob, time_independent=True
    )

    carbon_init_noise = DepolarNoiseModel(
        depolar_rate=cfg.carbon_init_depolar_prob, time_independent=True
    )

    carbon_z_rot_noise = DepolarNoiseModel(
        depolar_rate=cfg.carbon_z_rot_depolar_prob, time_independent=True
    )

    ec_noise = DepolarNoiseModel(
        depolar_rate=cfg.ec_gate_depolar_prob, time_independent=True
    )

    electron_qubit_noise = T1T2NoiseModel(T1=cfg.electron_T1, T2=cfg.electron_T2)

    carbon_qubit_noise = T1T2NoiseModel(T1=cfg.carbon_T1, T2=cfg.carbon_T2)

    # defining gates and their gate times

    phys_instructions = []

    electron_position = 0
    carbon_positions = [pos + 1 for pos in range(cfg.num_qubits - 1)]

    phys_instructions.append(
        PhysicalInstruction(
            INSTR_INIT,
            topology=carbon_positions,
            quantum_noise_model=carbon_init_noise,
            duration=cfg.carbon_init,
        )
    )

    for instr, dur in zip(
        [INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z],
        [cfg.carbon_rot_x, cfg.carbon_rot_y, cfg.carbon_rot_z],
    ):
        phys_instructions.append(
            PhysicalInstruction(
                instr,
                topology=carbon_positions,
                quantum_noise_model=carbon_z_rot_noise,
                duration=dur,
            )
        )

    phys_instructions.append(
        PhysicalInstruction(
            INSTR_INIT,
            topology=[electron_position],
            quantum_noise_model=electron_init_noise,
            duration=cfg.electron_init,
        )
    )

    for instr, dur in zip(
        [INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z],
        [cfg.electron_rot_x, cfg.electron_rot_y, cfg.electron_rot_z],
    ):
        phys_instructions.append(
            PhysicalInstruction(
                instr,
                topology=[electron_position],
                quantum_noise_model=electron_single_qubit_noise,
                duration=dur,
            )
        )

    electron_carbon_topologies = [
        (electron_position, carbon_pos) for carbon_pos in carbon_positions
    ]
    phys_instructions.append(
        PhysicalInstruction(
            INSTR_CXDIR,
            topology=electron_carbon_topologies,
            quantum_noise_model=ec_noise,
            duration=cfg.ec_controlled_dir_x,
        )
    )

    phys_instructions.append(
        PhysicalInstruction(
            INSTR_CYDIR,
            topology=electron_carbon_topologies,
            quantum_noise_model=ec_noise,
            duration=cfg.ec_controlled_dir_y,
        )
    )

    M0 = Operator(
        "M0", np.diag([np.sqrt(1 - cfg.prob_error_0), np.sqrt(cfg.prob_error_1)])
    )
    M1 = Operator(
        "M1", np.diag([np.sqrt(cfg.prob_error_0), np.sqrt(1 - cfg.prob_error_1)])
    )

    # hack to set imperfect measurements
    INSTR_MEASURE._meas_operators = [M0, M1]

    phys_instr_measure = PhysicalInstruction(
        INSTR_MEASURE,
        topology=[electron_position],
        quantum_noise_model=None,
        duration=cfg.measure,
    )

    phys_instructions.append(phys_instr_measure)

    # add qubits
    mem_noise_models = [electron_qubit_noise] + [carbon_qubit_noise] * len(
        carbon_positions
    )
    qmem = QuantumProcessor(
        name=name,
        num_positions=cfg.num_qubits,
        mem_noise_models=mem_noise_models,
        phys_instructions=phys_instructions,
    )
    return qmem


def build_procnode(cfg: ProcNodeConfig, global_env: GlobalEnvironment) -> ProcNode:
    topology = LhiTopologyBuilder.from_config(cfg.topology)
    qprocessor = build_qprocessor_from_topology(name=cfg.node_name, topology=topology)
    latencies = LhiLatencies.from_config(cfg.latencies)
    procnode = ProcNode(
        cfg.node_name,
        global_env=global_env,
        qprocessor=qprocessor,
        qdevice_topology=topology,
        latencies=latencies,
        ntf_interface=GenericToVanillaInterface(),  # TODO: make configurable
        node_id=cfg.node_id,
    )

    # TODO: refactor this hack
    procnode.qnos.processor._latencies.qnos_instr_time = cfg.latencies.qnos_instr_time
    procnode.host.processor._latencies.host_instr_time = cfg.latencies.host_instr_time
    procnode.host.processor._latencies.host_peer_latency = (
        cfg.latencies.host_peer_latency
    )
    return procnode


def build_ll_protocol(
    config: LinkConfig, proc_node1: ProcNode, proc_node2: ProcNode
) -> MagicLinkLayerProtocolWithSignaling:
    if config.typ == "perfect":
        link_dist = PerfectStateMagicDistributor(
            nodes=[proc_node1.node, proc_node2.node], state_delay=0
        )
    elif config.typ == "depolarise":
        link_cfg = config.cfg
        if not isinstance(link_cfg, DepolariseLinkConfig):
            link_cfg = DepolariseLinkConfig(**config.cfg)
        prob_max_mixed = fidelity_to_prob_max_mixed(link_cfg.fidelity)
        link_dist = DepolariseWithFailureMagicDistributor(
            nodes=[proc_node1.node, proc_node2.node],
            prob_max_mixed=prob_max_mixed,
            prob_success=link_cfg.prob_success,
            t_cycle=link_cfg.t_cycle,
        )
    # TODO: decide whether this is still wanted/needed
    # elif config.typ == "nv":
    #     link_cfg = config.cfg
    #     if not isinstance(link_cfg, NVLinkConfig):
    #         link_cfg = NVLinkConfig(**config.cfg)
    #     link_dist = NVSingleClickMagicDistributor(
    #         nodes=[proc_node1.node, proc_node2.node],
    #         length_A=link_cfg.length_A,
    #         length_B=link_cfg.length_B,
    #         full_cycle=link_cfg.full_cycle,
    #         cycle_time=link_cfg.cycle_time,
    #         alpha=link_cfg.alpha,
    #     )
    elif config.typ == "heralded":
        link_cfg = config.cfg
        if not isinstance(link_cfg, HeraldedLinkConfig):
            link_cfg = HeraldedLinkConfig(**config.cfg)
        connection = MiddleHeraldedConnection(name="heralded_conn", **link_cfg.dict())
        link_dist = DoubleClickMagicDistributor(
            [proc_node1.node, proc_node2.node], connection
        )
    else:
        raise ValueError

    return MagicLinkLayerProtocolWithSignaling(
        nodes=[proc_node1.node, proc_node2.node],
        magic_distributor=link_dist,
        translation_unit=SingleClickTranslationUnit(),
    )


def build_network(
    config: ProcNodeNetworkConfig, global_env: GlobalEnvironment
) -> ProcNodeNetwork:
    procnodes: Dict[str, ProcNode] = {}

    for cfg in config.nodes:
        procnodes[cfg.node_name] = build_procnode(cfg, global_env)

    ns_nodes = [procnode.node for procnode in procnodes.values()]
    gedcomp = EntDistComponent(global_env)
    ged = EntDist(nodes=ns_nodes, global_env=global_env, comp=gedcomp)

    for (_, s1), (_, s2) in itertools.combinations(procnodes.items(), 2):
        s1.connect_to(s2)

        factory = PerfectStateSamplerFactory()
        kwargs = {"cycle_time": 1000}
        ged.add_sampler(s1.node.ID, s2.node.ID, factory, kwargs=kwargs)

    for name, procnode in procnodes.items():
        procnode.node.entdist_out_port.connect(gedcomp.node_in_port(name))
        procnode.node.entdist_in_port.connect(gedcomp.node_out_port(name))

    return ProcNodeNetwork(procnodes, ged)
