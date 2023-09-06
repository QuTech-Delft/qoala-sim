import itertools
from typing import Dict, FrozenSet, List

from netsquid.components import ClassicalChannel
from netsquid.components.models import FibreDelayModel
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.components.qprocessor import PhysicalInstruction, QuantumProcessor
from netsquid.nodes.connections import Connection

from qoala.lang.ehi import EhiLinkInfo, EhiNetworkInfo

# Ignore type since whole 'config' module is ignored by mypy
from qoala.runtime.config import ProcNodeConfig, ProcNodeNetworkConfig  # type: ignore
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiLatencies,
    LhiLinkInfo,
    LhiNetworkInfo,
    LhiNetworkSchedule,
    LhiProcNodeInfo,
    LhiTopology,
    LhiTopologyBuilder,
)
from qoala.runtime.lhi_to_ehi import LhiConverter
from qoala.runtime.ntf import NtfInterface
from qoala.sim.entdist.entdist import EntDist
from qoala.sim.entdist.entdistcomp import EntDistComponent
from qoala.sim.network import ProcNodeNetwork
from qoala.sim.procnode import ProcNode


class ClassicalConnection(Connection):
    def __init__(self, name: str, length: float):
        super().__init__(name=name)
        self.add_subcomponent(
            ClassicalChannel(
                "Channel_A2B", length=length, models={"delay_model": FibreDelayModel()}
            )
        )
        self.ports["A"].forward_input(self.subcomponents["Channel_A2B"].ports["send"])
        self.subcomponents["Channel_A2B"].ports["recv"].forward_output(self.ports["B"])


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
            # TODO: refactor this hack
            if gate_info.instruction == INSTR_MEASURE_INSTANT:
                duration = 0.0
            else:
                duration = gate_info.duration

            phys_instr = PhysicalInstruction(
                instruction=gate_info.instruction,
                duration=duration,
                topology=[qubit_id],
                quantum_noise_model=gate_info.error_model(
                    **gate_info.error_model_kwargs
                ),
                parallel=True,
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

    if topology.all_qubit_gate_infos is not None:
        for gate_info in topology.all_qubit_gate_infos:
            phys_instr = PhysicalInstruction(
                instruction=gate_info.instruction,
                duration=gate_info.duration,
                topology=[tuple(range(num_qubits))],
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


def build_procnode_from_config(
    cfg: ProcNodeConfig, network_ehi: EhiNetworkInfo
) -> ProcNode:
    topology = LhiTopologyBuilder.from_config(cfg.topology)

    ntf_interface_cls = cfg.ntf.to_ntf_interface()
    ntf_interface = ntf_interface_cls()

    qprocessor = build_qprocessor_from_topology(name=cfg.node_name, topology=topology)
    latencies = LhiLatencies.from_config(cfg.latencies)
    procnode = ProcNode(
        cfg.node_name,
        qprocessor=qprocessor,
        qdevice_topology=topology,
        latencies=latencies,
        ntf_interface=ntf_interface,
        node_id=cfg.node_id,
        network_ehi=network_ehi,
        deterministic_scheduler=cfg.determ_sched,
        use_deadlines=cfg.use_deadlines,
        is_predictable=cfg.is_predictable,
    )

    # TODO: refactor this hack
    procnode.qnos.processor._latencies.qnos_instr_time = cfg.latencies.qnos_instr_time
    procnode.host.processor._latencies.host_instr_time = cfg.latencies.host_instr_time
    procnode.host.processor._latencies.host_peer_latency = (
        cfg.latencies.host_peer_latency
    )
    return procnode


def build_network_from_config(config: ProcNodeNetworkConfig) -> ProcNodeNetwork:
    procnodes: Dict[str, ProcNode] = {}

    ehi_links: Dict[FrozenSet[int], EhiLinkInfo] = {}
    for link_between_nodes in config.links:
        lhi_link = LhiLinkInfo.from_config(link_between_nodes.link_config)
        ehi_link = LhiConverter.link_info_to_ehi(lhi_link)
        ids = (link_between_nodes.node_id1, link_between_nodes.node_id2)
        node_link = frozenset(ids)
        ehi_links[node_link] = ehi_link
    nodes = {cfg.node_id: cfg.node_name for cfg in config.nodes}
    if config.netschedule is not None:
        lhi_netschedule = LhiNetworkSchedule.from_config(config.netschedule)
        ehi_netschedule = LhiConverter.netschedule_to_ehi(lhi_netschedule)
        network_ehi = EhiNetworkInfo(nodes, ehi_links, ehi_netschedule)
    else:
        network_ehi = EhiNetworkInfo(nodes, ehi_links)

    for cfg in config.nodes:
        procnodes[cfg.node_name] = build_procnode_from_config(cfg, network_ehi)

    ns_nodes = [procnode.node for procnode in procnodes.values()]
    entdistcomp = EntDistComponent(network_ehi)
    entdist = EntDist(nodes=ns_nodes, ehi_network=network_ehi, comp=entdistcomp)

    for link_between_nodes in config.links:
        link = LhiLinkInfo.from_config(link_between_nodes.link_config)
        n1 = link_between_nodes.node_id1
        n2 = link_between_nodes.node_id2
        entdist.add_sampler(n1, n2, link)

    def get_latency(node1: int, node2: int) -> float:
        if config.cconns is None:
            return 0.0
        for cconn in config.cconns:
            if {cconn.node_id1, cconn.node_id2} == {node1, node2}:
                return cconn.latency  # type: ignore
        return 0.0

    for s1, s2 in itertools.combinations(procnodes.values(), 2):
        latency = get_latency(s1.node.node_id, s2.node.node_id)
        s1.connect_to(s2, latency)

    # TODO: make this configurable
    node_entdist_latency = 0
    for name, procnode in procnodes.items():
        chan_ne = ClassicalChannel(f"chan_{name}_entdist", delay=node_entdist_latency)
        procnode.node.entdist_out_port.connect(chan_ne.ports["send"])
        chan_ne.ports["recv"].connect(entdistcomp.node_in_port(name))

        chan_en = ClassicalChannel(f"chan_entdist_{name}", delay=node_entdist_latency)
        entdistcomp.node_out_port(name).connect(chan_en.ports["send"])
        chan_en.ports["recv"].connect(procnode.node.entdist_in_port)

    return ProcNodeNetwork(procnodes, entdist)


def build_procnode_from_lhi(
    id: int,
    name: str,
    topology: LhiTopology,
    latencies: LhiLatencies,
    network_lhi: LhiNetworkInfo,
    ntf: NtfInterface,
) -> ProcNode:
    qprocessor = build_qprocessor_from_topology(f"{name}_processor", topology)
    network_ehi = LhiConverter.network_to_ehi(network_lhi)
    return ProcNode(
        name=name,
        node_id=id,
        qprocessor=qprocessor,
        qdevice_topology=topology,
        latencies=latencies,
        ntf_interface=ntf,
        network_ehi=network_ehi,
    )


def build_network_from_lhi(
    procnode_infos: List[LhiProcNodeInfo],
    ntfs: List[NtfInterface],
    network_lhi: LhiNetworkInfo,
) -> ProcNodeNetwork:
    procnodes: Dict[str, ProcNode] = {}

    # TODO: refactor two separate lists (infos and ntfs)
    for info, ntf in zip(procnode_infos, ntfs):
        procnode = build_procnode_from_lhi(
            info.id, info.name, info.topology, info.latencies, network_lhi, ntf
        )
        procnodes[info.name] = procnode

    ns_nodes = [procnode.node for procnode in procnodes.values()]
    network_ehi = LhiConverter.network_to_ehi(network_lhi)
    entdistcomp = EntDistComponent(network_ehi)
    entdist = EntDist(nodes=ns_nodes, ehi_network=network_ehi, comp=entdistcomp)

    for ([n1, n2], link_info) in network_lhi.links.items():
        entdist.add_sampler(n1, n2, link_info)

    for s1, s2 in itertools.combinations(procnodes.values(), 2):
        s1.connect_to(s2)

    for name, procnode in procnodes.items():
        procnode.node.entdist_out_port.connect(entdistcomp.node_in_port(name))
        procnode.node.entdist_in_port.connect(entdistcomp.node_out_port(name))

    return ProcNodeNetwork(procnodes, entdist)
