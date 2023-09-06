# Low-level Hardware Info. Expressed using NetSquid concepts and objects.
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Type

from netsquid.components.instructions import (
    INSTR_CNOT,
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
    IMeasure,
)
from netsquid.components.instructions import Instruction as NetSquidInstruction
from netsquid.components.models.qerrormodels import (
    DepolarNoiseModel,
    QuantumErrorModel,
    T1T2NoiseModel,
)
from netsquid_magic.state_delivery_sampler import (
    DepolariseWithFailureStateSamplerFactory,
    IStateDeliverySamplerFactory,
    PerfectStateSamplerFactory,
)

from qoala.lang.common import MultiQubit
from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)

# A measurement that is meant to take 0 time.
# Used in e.g. Measure Directly EPR generation.
INSTR_MEASURE_INSTANT = IMeasure("measurement_instant_op")


# Config Interface


class LhiQubitConfigInterface(ABC):
    @abstractmethod
    def to_is_communication(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class LhiGateConfigInterface(ABC):
    @abstractmethod
    def to_instruction(self) -> Type[NetSquidInstruction]:
        raise NotImplementedError

    @abstractmethod
    def to_duration(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class LhiTopologyConfigInterface(ABC):
    @abstractmethod
    def get_qubit_configs(self) -> Dict[int, LhiQubitConfigInterface]:
        raise NotImplementedError

    @abstractmethod
    def get_single_gate_configs(self) -> Dict[int, List[LhiGateConfigInterface]]:
        raise NotImplementedError

    @abstractmethod
    def get_multi_gate_configs(
        self,
    ) -> Dict[MultiQubit, List[LhiGateConfigInterface]]:
        raise NotImplementedError

    def get_all_qubit_gate_configs(self) -> List[LhiGateConfigInterface]:
        raise NotImplementedError


# Data classes


@dataclass
class LhiQubitInfo:
    is_communication: bool
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


@dataclass
class LhiGateInfo:
    instruction: Type[NetSquidInstruction]
    duration: float  # ns
    error_model: Type[QuantumErrorModel]
    error_model_kwargs: Dict[str, Any]


@dataclass
class LhiTopology:
    qubit_infos: Dict[int, LhiQubitInfo]  # qubit ID -> info
    single_gate_infos: Dict[int, List[LhiGateInfo]]  # qubit ID -> gates
    multi_gate_infos: Dict[
        MultiQubit, List[LhiGateInfo]
    ]  # ordered qubit ID list -> gates
    all_qubit_gate_infos: Optional[List[LhiGateInfo]] = None

    def find_single_gate(
        self, qubit_id: int, instr: Type[NetSquidInstruction]
    ) -> Optional[LhiGateInfo]:
        if qubit_id not in self.single_gate_infos:
            return None
        for info in self.single_gate_infos[qubit_id]:
            if info.instruction == instr:
                return info
        return None

    def find_multi_gate(
        self, qubit_ids: List[int], instr: Type[NetSquidInstruction]
    ) -> Optional[LhiGateInfo]:
        multi = MultiQubit(qubit_ids)
        if multi not in self.multi_gate_infos:
            return None
        for info in self.multi_gate_infos[multi]:
            if info.instruction == instr:
                return info
        return None

    def find_all_qubit_gate(
        self, instr: Type[NetSquidInstruction]
    ) -> Optional[LhiGateInfo]:
        if self.all_qubit_gate_infos is None:
            return None

        for info in self.all_qubit_gate_infos:
            if info.instruction == instr:
                return info
        return None


# Convenience methods.


class LhiTopologyBuilder:
    """Convenience methods for creating a Topology object."""

    @classmethod
    def from_config(cls, cfg: LhiTopologyConfigInterface) -> LhiTopology:
        qubit_infos: Dict[int, LhiQubitInfo] = {}
        for i, cfg_info in cfg.get_qubit_configs().items():
            qubit_infos[i] = LhiQubitInfo(
                is_communication=cfg_info.to_is_communication(),
                error_model=cfg_info.to_error_model(),
                error_model_kwargs=cfg_info.to_error_model_kwargs(),
            )

        single_gate_infos: Dict[int, List[LhiGateInfo]] = {}
        for i, cfg_infos in cfg.get_single_gate_configs().items():
            single_gate_infos[i] = [
                LhiGateInfo(
                    instruction=info.to_instruction(),
                    duration=info.to_duration(),
                    error_model=info.to_error_model(),
                    error_model_kwargs=info.to_error_model_kwargs(),
                )
                for info in cfg_infos
            ]

        multi_gate_infos: Dict[MultiQubit, List[LhiGateInfo]] = {}
        for ids, cfg_infos in cfg.get_multi_gate_configs().items():
            multi_gate_infos[ids] = [
                LhiGateInfo(
                    instruction=info.to_instruction(),
                    duration=info.to_duration(),
                    error_model=info.to_error_model(),
                    error_model_kwargs=info.to_error_model_kwargs(),
                )
                for info in cfg_infos
            ]

        all_qubit_gate_infos: Optional[List[LhiGateInfo]] = None
        if cfg.get_all_qubit_gate_configs() is not None:
            all_qubit_gate_infos = []
            for info in cfg.get_all_qubit_gate_configs():
                all_qubit_gate_infos.append(
                    LhiGateInfo(
                        instruction=info.to_instruction(),
                        duration=info.to_duration(),
                        error_model=info.to_error_model(),
                        error_model_kwargs=info.to_error_model_kwargs(),
                    )
                )

        return LhiTopology(
            qubit_infos=qubit_infos,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
            all_qubit_gate_infos=all_qubit_gate_infos,
        )

    @classmethod
    def t1t2_qubit(cls, is_communication: bool, t1: float, t2: float) -> LhiQubitInfo:
        return LhiQubitInfo(
            is_communication=is_communication,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": t1, "T2": t2},
        )

    @classmethod
    def perfect_qubit(cls, is_communication: bool) -> LhiQubitInfo:
        return cls.t1t2_qubit(is_communication=is_communication, t1=0, t2=0)

    @classmethod
    def depolar_gates(
        cls,
        duration: float,
        instructions: List[NetSquidInstruction],
        depolar_rate: float,
    ) -> List[LhiGateInfo]:
        return [
            LhiGateInfo(
                instruction=instr,
                duration=duration,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": depolar_rate},
            )
            for instr in instructions
        ]

    @classmethod
    def perfect_gates(
        cls, duration: float, instructions: List[NetSquidInstruction]
    ) -> List[LhiGateInfo]:
        return cls.depolar_gates(
            duration=duration, instructions=instructions, depolar_rate=0
        )

    @classmethod
    def perfect_uniform_default_gates(cls, num_qubits) -> LhiTopology:
        # TODO: test this and update default values
        return cls.perfect_uniform(
            num_qubits=num_qubits,
            single_instructions=[
                INSTR_INIT,
                INSTR_X,
                INSTR_Y,
                INSTR_Z,
                INSTR_H,
                INSTR_ROT_X,
                INSTR_ROT_Y,
                INSTR_ROT_Z,
                INSTR_MEASURE,
                INSTR_MEASURE_INSTANT,
            ],
            single_duration=5e3,
            two_instructions=[INSTR_CNOT, INSTR_CZ],
            two_duration=100e3,
        )

    @classmethod
    def perfect_uniform(
        cls,
        num_qubits,
        single_instructions: List[NetSquidInstruction],
        single_duration: float,
        two_instructions: List[NetSquidInstruction],
        two_duration: float,
        all_qubit_instructions: List[NetSquidInstruction] = None,
        all_qubit_duration: float = 0,
    ) -> LhiTopology:
        if all_qubit_instructions is None:
            return cls.fully_uniform(
                num_qubits=num_qubits,
                qubit_info=cls.perfect_qubit(is_communication=True),
                single_gate_infos=cls.perfect_gates(
                    single_duration, single_instructions
                ),
                two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
            )
        else:
            all_qubit_gate_infos = cls.perfect_gates(
                all_qubit_duration, all_qubit_instructions
            )
            return cls.fully_uniform(
                num_qubits=num_qubits,
                qubit_info=cls.perfect_qubit(is_communication=True),
                single_gate_infos=cls.perfect_gates(
                    single_duration, single_instructions
                ),
                two_gate_infos=cls.perfect_gates(two_duration, two_instructions),
                all_qubit_gate_infos=all_qubit_gate_infos,
            )

    @classmethod
    def fully_uniform(
        cls,
        num_qubits,
        qubit_info: LhiQubitInfo,
        single_gate_infos: List[LhiGateInfo],
        two_gate_infos: List[LhiGateInfo],
        all_qubit_gate_infos: List[LhiGateInfo] = None,
    ) -> LhiTopology:
        q_infos = {i: qubit_info for i in range(num_qubits)}
        sg_infos = {i: single_gate_infos for i in range(num_qubits)}
        mg_infos = {}
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    multi = MultiQubit([i, j])
                    mg_infos[multi] = two_gate_infos

        return LhiTopology(q_infos, sg_infos, mg_infos, all_qubit_gate_infos)

    @classmethod
    def perfect_star(
        cls,
        num_qubits: int,
        comm_instructions: List[NetSquidInstruction],
        comm_duration: float,
        mem_instructions: List[NetSquidInstruction],
        mem_duration: float,
        two_instructions: List[NetSquidInstruction],
        two_duration: float,
    ) -> LhiTopology:
        comm_qubit_info = cls.perfect_qubit(is_communication=True)
        mem_qubit_info = cls.perfect_qubit(is_communication=False)
        comm_gate_infos = cls.perfect_gates(comm_duration, comm_instructions)
        mem_gate_infos = cls.perfect_gates(mem_duration, mem_instructions)
        two_gate_infos = cls.perfect_gates(two_duration, two_instructions)

        q_infos = {0: comm_qubit_info}
        for i in range(1, num_qubits):
            q_infos[i] = mem_qubit_info

        sg_infos = {0: comm_gate_infos}
        for i in range(1, num_qubits):
            sg_infos[i] = mem_gate_infos

        mg_infos = {}
        for i in range(1, num_qubits):
            mg_infos[MultiQubit([0, i])] = two_gate_infos

        return LhiTopology(q_infos, sg_infos, mg_infos)

    @classmethod
    def generic_t1t2_star(
        cls,
        num_qubits: int,
        comm_t1: float,
        comm_t2: float,
        mem_t1: float,
        mem_t2: float,
        comm_instructions: List[NetSquidInstruction],
        comm_duration: float,
        comm_instr_depolar_rate: float,
        mem_instructions: List[NetSquidInstruction],
        mem_duration: float,
        mem_instr_depolar_rate: float,
        two_instructions: List[NetSquidInstruction],
        two_duration: float,
        two_instr_depolar_rate: float,
    ) -> LhiTopology:
        comm_qubit_info = cls.t1t2_qubit(is_communication=True, t1=comm_t1, t2=comm_t2)
        mem_qubit_info = cls.t1t2_qubit(is_communication=False, t1=mem_t1, t2=mem_t2)

        comm_gate_infos = cls.depolar_gates(
            comm_duration, comm_instructions, comm_instr_depolar_rate
        )
        mem_gate_infos = cls.depolar_gates(
            mem_duration, mem_instructions, mem_instr_depolar_rate
        )
        two_gate_infos = cls.depolar_gates(
            two_duration, two_instructions, two_instr_depolar_rate
        )

        q_infos = {0: comm_qubit_info}
        for i in range(1, num_qubits):
            q_infos[i] = mem_qubit_info

        sg_infos = {0: comm_gate_infos}
        for i in range(1, num_qubits):
            sg_infos[i] = mem_gate_infos

        mg_infos = {}
        for i in range(1, num_qubits):
            mg_infos[MultiQubit([0, i])] = two_gate_infos

        return LhiTopology(q_infos, sg_infos, mg_infos)

    @classmethod
    def trapped_ion_default_perfect_gates(cls, num_qubits: int) -> LhiTopology:
        # TODO check default values
        return cls.perfect_uniform(
            num_qubits=num_qubits,
            single_instructions=[
                INSTR_INIT,
                INSTR_ROT_Z,
                INSTR_MEASURE,
                INSTR_MEASURE_INSTANT,
            ],
            single_duration=5e3,
            two_instructions=[],
            two_duration=0,
            all_qubit_instructions=[
                INSTR_INIT,
                INSTR_MEASURE_ALL,
                INSTR_ROT_X_ALL,
                INSTR_ROT_Y_ALL,
                INSTR_ROT_Z_ALL,
                INSTR_BICHROMATIC,
            ],
            all_qubit_duration=5e3,
        )


class LhiLatenciesConfigInterface(ABC):
    @abstractmethod
    def get_host_instr_time(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_qnos_instr_time(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_host_peer_latency(self) -> float:
        raise NotImplementedError

    def get_internal_sched_latency(self):
        raise NotImplementedError


@dataclass
class LhiLatencies:
    host_instr_time: float = 0  # duration of classical Host instr execution
    qnos_instr_time: float = 0  # duration of classical Qnos instr execution
    host_peer_latency: float = 0  # processing time for Host messages from remote node
    internal_sched_latency: float = 0  # processing time for messaging between node scheduler and processor schedulers

    @classmethod
    def from_config(cls, cfg: LhiLatenciesConfigInterface) -> LhiLatencies:
        return LhiLatencies(
            host_instr_time=cfg.get_host_instr_time(),
            qnos_instr_time=cfg.get_qnos_instr_time(),
            host_peer_latency=cfg.get_host_peer_latency(),
            internal_sched_latency=cfg.get_internal_sched_latency(),
        )

    @classmethod
    def all_zero(cls) -> LhiLatencies:
        # NOTE: can also just use LhiLatencies() which will default all values to 0
        # However, using this classmethod makes this behavior more explicit and clear.
        return LhiLatencies(0, 0, 0, 0)


class LhiLinkConfigInterface(ABC):
    @abstractmethod
    def to_sampler_factory(self) -> Type[IStateDeliverySamplerFactory]:
        raise NotImplementedError

    @abstractmethod
    def to_sampler_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_state_delay(self) -> float:
        raise NotImplementedError


class LhiNetworkScheduleConfigInterface(ABC):
    @abstractmethod
    def to_bin_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to_first_bin(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to_bin_pattern(self) -> List[LhiNetworkTimebin]:
        raise NotImplementedError

    @abstractmethod
    def to_repeat_period(self) -> int:
        raise NotImplementedError


@dataclass
class LhiLinkInfo:
    sampler_factory: Type[IStateDeliverySamplerFactory]
    sampler_kwargs: Dict[str, Any]
    state_delay: float  # time between EPR generation and putting the state into memory

    @classmethod
    def from_config(cls, cfg: LhiLinkConfigInterface) -> LhiLinkInfo:
        return LhiLinkInfo(
            sampler_factory=cfg.to_sampler_factory(),
            sampler_kwargs=cfg.to_sampler_kwargs(),
            state_delay=cfg.to_state_delay(),
        )

    @classmethod
    def perfect(cls, duration: float) -> LhiLinkInfo:
        return LhiLinkInfo(
            sampler_factory=PerfectStateSamplerFactory,
            sampler_kwargs={"cycle_time": 0},
            state_delay=duration,
        )

    @classmethod
    def depolarise(
        cls,
        cycle_time: float,
        prob_max_mixed: float,
        prob_success: float,
        state_delay: float,
    ) -> LhiLinkInfo:
        return LhiLinkInfo(
            sampler_factory=DepolariseWithFailureStateSamplerFactory,
            sampler_kwargs={
                "cycle_time": cycle_time,
                "prob_max_mixed": prob_max_mixed,
                "prob_success": prob_success,
            },
            state_delay=state_delay,
        )


@dataclass
class LhiNetworkInfo:
    nodes: Dict[int, str]  # node ID -> node name

    # (node A ID, node B ID) -> link info
    # for a pair (a, b) there exists no separate (b, a) info (it is the same)
    links: Dict[FrozenSet[int], LhiLinkInfo]

    @classmethod
    def fully_connected(
        cls, nodes: Dict[int, str], info: LhiLinkInfo
    ) -> LhiNetworkInfo:
        links: Dict[FrozenSet[int], LhiLinkInfo] = {}
        for n1, n2 in itertools.combinations(nodes.keys(), 2):
            node_link = frozenset([n1, n2])
            links[node_link] = info
        return LhiNetworkInfo(nodes, links)

    @classmethod
    def perfect_fully_connected(
        cls, nodes: Dict[int, str], duration: float
    ) -> LhiNetworkInfo:
        link = LhiLinkInfo.perfect(duration)
        return cls.fully_connected(nodes, link)

    def add_link(self, node1_id, node2_id, link_info: LhiLinkInfo):
        if node1_id not in self.nodes:
            raise ValueError(f"Node with ID {node1_id} not found")
        if node2_id not in self.nodes:
            raise ValueError(f"Node with ID {node2_id} not found")
        if node1_id == node2_id:
            raise ValueError("Cannot add link between same node")
        node_link = frozenset([node1_id, node2_id])
        if node_link in self.links:
            raise ValueError(
                f"Link between nodes {node1_id} and {node2_id} already exists"
            )
        self.links[node_link] = link_info

    def get_link(self, node_id1: int, node_id2: int) -> LhiLinkInfo:
        node_link = frozenset([node_id1, node_id2])
        try:
            return self.links[node_link]
        except KeyError:
            raise ValueError(
                f"There is no link between nodes {node_id1} and {node_id2} in the network"
            )


@dataclass
class LhiNetworkTimebin:
    nodes: FrozenSet[int]
    pids: Dict[int, int]  # node ID -> PID


@dataclass
class LhiNetworkSchedule:
    bin_length: int
    first_bin: int
    bin_pattern: List[LhiNetworkTimebin]
    repeat_period: int

    @classmethod
    def from_config(
        cls, config: LhiNetworkScheduleConfigInterface
    ) -> LhiNetworkSchedule:
        return LhiNetworkSchedule(
            config.to_bin_length(),
            config.to_first_bin(),
            config.to_bin_pattern(),
            config.to_repeat_period(),
        )


@dataclass
class LhiProcNodeInfo:
    id: int
    name: str
    topology: LhiTopology
    latencies: LhiLatencies
