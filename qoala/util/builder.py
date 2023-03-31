from typing import Optional

from qoala.lang.ehi import EhiNetworkInfo, UnitModule
from qoala.lang.program import QoalaProgram
from qoala.runtime.environment import NetworkInfo
from qoala.runtime.lhi import LhiLatencies, LhiTopologyBuilder
from qoala.runtime.lhi_to_ehi import GenericToVanillaInterface, LhiConverter
from qoala.runtime.program import ProgramInput, ProgramInstance
from qoala.sim.build import build_qprocessor_from_topology
from qoala.sim.procnode import ProcNode


class ObjectBuilder:
    @classmethod
    def simple_procnode(cls, name: str, num_qubits: int) -> ProcNode:
        env = NetworkInfo.with_nodes({0: name})
        network_ehi = EhiNetworkInfo(links={})
        topology = LhiTopologyBuilder.perfect_uniform_default_gates(num_qubits)
        qprocessor = build_qprocessor_from_topology(f"{name}_processor", topology)
        return ProcNode(
            name=name,
            network_info=env,
            qprocessor=qprocessor,
            qdevice_topology=topology,
            latencies=LhiLatencies.all_zero(),
            ntf_interface=GenericToVanillaInterface(),
            network_ehi=network_ehi,
        )

    @classmethod
    def simple_program_instance(
        cls, program: QoalaProgram, pid: int = 0, inputs: Optional[ProgramInput] = None
    ) -> ProgramInstance:
        topology = LhiTopologyBuilder.perfect_uniform_default_gates(1)
        ehi = LhiConverter.to_ehi(topology, GenericToVanillaInterface())
        unit_module = UnitModule.from_full_ehi(ehi)

        if inputs is None:
            inputs = ProgramInput.empty()

        return ProgramInstance(
            pid,
            program,
            inputs,
            unit_module=unit_module,
            block_tasks=[],
        )
