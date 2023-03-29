# type: ignore
from __future__ import annotations

import itertools
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CROT_X,
    INSTR_CROT_Y,
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
from pydantic import BaseModel as PydanticBaseModel

from qoala.lang.common import MultiQubit
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiGateConfigInterface,
    LhiLatenciesConfigInterface,
    LhiLinkConfigInterface,
    LhiQubitConfigInterface,
    LhiTopologyConfigInterface,
)
from qoala.util.constants import fidelity_to_prob_max_mixed


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class InstrConfigRegistry(ABC):
    @abstractclassmethod
    def map(cls) -> Dict[str, NetSquidInstruction]:
        raise NotImplementedError


class QubitConfigRegistry(ABC):
    @abstractclassmethod
    def map(cls) -> Dict[str, LhiQubitConfigInterface]:
        raise NotImplementedError


class GateConfigRegistry(ABC):
    @abstractclassmethod
    def map(cls) -> Dict[str, GateNoiseConfigInterface]:
        raise NotImplementedError


class SamplerFactoryRegistry(ABC):
    @abstractclassmethod
    def map(cls) -> Dict[str, IStateDeliverySamplerFactory]:
        raise NotImplementedError


def _from_dict(dict: Any, typ: Any) -> Any:
    return typ(**dict)


def _from_file(path: str, typ: Any) -> Any:
    with open(path, "r") as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        return _from_dict(raw_config, typ)


def _read_dict(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


class QubitNoiseConfigInterface:
    @abstractclassmethod
    def from_dict(cls, dict: Any) -> QubitNoiseConfigInterface:
        raise NotImplementedError

    @abstractmethod
    def to_error_model(self) -> Type[QuantumErrorModel]:
        raise NotImplementedError

    @abstractmethod
    def to_error_model_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class QubitT1T2Config(QubitNoiseConfigInterface, BaseModel):
    T1: int
    T2: int

    @classmethod
    def from_file(cls, path: str) -> QubitT1T2Config:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> QubitT1T2Config:
        return QubitT1T2Config(**dict)

    def to_error_model(self) -> Type[QuantumErrorModel]:
        return T1T2NoiseModel

    def to_error_model_kwargs(self) -> Dict[str, Any]:
        return {"T1": self.T1, "T2": self.T2}


class QubitConfig(LhiQubitConfigInterface, BaseModel):
    is_communication: bool
    noise_config_cls: str
    noise_config: QubitNoiseConfigInterface

    @classmethod
    def from_file(
        cls, path: str, registry: Optional[List[Type[QubitConfigRegistry]]] = None
    ) -> QubitConfig:
        return cls.from_dict(_read_dict(path), registry)

    @classmethod
    def perfect_config(cls, is_communication: bool) -> QubitConfig:
        return QubitConfig(
            is_communication=is_communication,
            noise_config_cls="T1T2NoiseModel",
            noise_config=QubitT1T2Config(T1=0, T2=0),
        )

    @classmethod
    def from_dict(
        cls, dict: Any, registry: Optional[List[Type[QubitConfigRegistry]]] = None
    ) -> QubitConfig:
        is_communication = dict["is_communication"]
        raw_typ = dict["noise_config_cls"]

        # Try to get the type of the noise config class.
        typ: Optional[QubitNoiseConfigInterface] = None
        # First try custom registries.
        if registry is not None:
            try:
                for reg in registry:
                    if raw_typ in reg.map():
                        typ = reg.map()[raw_typ]
                        break
            except KeyError:
                pass
        # If not found in custom registries, try default registry.
        if typ is None:
            try:
                typ = DefaultQubitConfigRegistry.map()[raw_typ]
            except KeyError:
                raise RuntimeError("invalid qubit noise class type")

        raw_noise_config = dict["noise_config"]
        noise_config = typ.from_dict(raw_noise_config)
        return QubitConfig(
            is_communication=is_communication,
            noise_config_cls=raw_typ,
            noise_config=noise_config,
        )

    def to_is_communication(self) -> bool:
        return self.is_communication

    def to_error_model(self) -> Type[QuantumErrorModel]:
        return self.noise_config.to_error_model()

    def to_error_model_kwargs(self) -> Dict[str, Any]:
        return self.noise_config.to_error_model_kwargs()


class GateNoiseConfigInterface:
    @abstractclassmethod
    def from_dict(cls, dict: Any) -> GateNoiseConfigInterface:
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


class GateDepolariseConfig(GateNoiseConfigInterface, BaseModel):
    duration: int
    depolarise_prob: float

    @classmethod
    def from_file(cls, path: str) -> GateDepolariseConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> GateDepolariseConfig:
        return GateDepolariseConfig(**dict)

    def to_duration(self) -> int:
        return self.duration

    def to_error_model(self) -> Type[QuantumErrorModel]:
        return DepolarNoiseModel

    def to_error_model_kwargs(self) -> Dict[str, Any]:
        return {"depolar_rate": self.depolarise_prob, "time_independent": True}


class GateConfig(LhiGateConfigInterface, BaseModel):
    name: str
    noise_config_cls: str
    noise_config: GateNoiseConfigInterface

    @classmethod
    def from_file(
        cls, path: str, registry: Optional[List[Type[GateConfigRegistry]]] = None
    ) -> GateConfig:
        return cls.from_dict(_read_dict(path), registry)

    @classmethod
    def perfect_config(cls, name: str, duration: int) -> GateConfig:
        return GateConfig(
            name=name,
            noise_config_cls="GateDepolariseConfig",
            noise_config=GateDepolariseConfig(duration=duration, depolarise_prob=0),
        )

    @classmethod
    def from_dict(
        cls,
        dict: Any,
        registry: Optional[List[Type[GateConfigRegistry]]] = None,
    ) -> GateConfig:
        name = dict["name"]

        raw_typ = dict["noise_config_cls"]

        # Try to get the type of the noise config class.
        typ: Optional[GateNoiseConfigInterface] = None
        # First try custom registries.
        if registry is not None:
            try:
                for reg in registry:
                    if raw_typ in reg.map():
                        typ = reg.map()[raw_typ]
                        break
            except KeyError:
                pass
        # If not found in custom registries, try default registry.
        if typ is None:
            try:
                typ = DefaultGateConfigRegistry.map()[raw_typ]
            except KeyError:
                raise RuntimeError("invalid instruction type")

        raw_noise_config = dict["noise_config"]
        noise_config = typ.from_dict(raw_noise_config)
        return GateConfig(
            name=name,
            noise_config_cls=raw_typ,
            noise_config=noise_config,
        )

    def to_instruction(
        self,
        registry: Optional[List[Type[InstrConfigRegistry]]] = None,
    ) -> Type[NetSquidInstruction]:
        # Try to get the NetSquid Instruction class.
        instr: Optional[Type[NetSquidInstruction]] = None
        # First try custom registries.
        if registry is not None:
            try:
                for reg in registry:
                    if self.name in reg.map():
                        instr = reg.map()[self.name]
                        break
            except KeyError:
                pass
        # If not found in custom registries, try default registry.
        if instr is None:
            try:
                instr = DefaultInstrConfigRegistry.map()[self.name]
            except KeyError:
                raise RuntimeError("invalid instruction type")
        return instr

    def to_duration(self) -> int:
        return self.noise_config.to_duration()

    def to_error_model(self) -> Type[QuantumErrorModel]:
        return self.noise_config.to_error_model()

    def to_error_model_kwargs(self) -> Dict[str, Any]:
        return self.noise_config.to_error_model_kwargs()


class DefaultInstrConfigRegistry(InstrConfigRegistry):
    _MAP = {
        "INSTR_INIT": INSTR_INIT,
        "INSTR_X": INSTR_X,
        "INSTR_Y": INSTR_Y,
        "INSTR_Z": INSTR_Z,
        "INSTR_H": INSTR_H,
        "INSTR_ROT_X": INSTR_ROT_X,
        "INSTR_ROT_Y": INSTR_ROT_Y,
        "INSTR_ROT_Z": INSTR_ROT_Z,
        "INSTR_CNOT": INSTR_CNOT,
        "INSTR_CZ": INSTR_CZ,
        "INSTR_CROT_X": INSTR_CROT_X,
        "INSTR_CROT_Y": INSTR_CROT_Y,
        "INSTR_MEASURE": INSTR_MEASURE,
        "INSTR_MEASURE_INSTANT": INSTR_MEASURE_INSTANT,
    }

    @classmethod
    def map(cls) -> Dict[str, NetSquidInstruction]:
        return cls._MAP


class DefaultQubitConfigRegistry(QubitConfigRegistry):
    _MAP = {
        "QubitT1T2Config": QubitT1T2Config,
    }

    @classmethod
    def map(cls) -> Dict[str, QubitNoiseConfigInterface]:
        return cls._MAP


class DefaultGateConfigRegistry(GateConfigRegistry):
    _MAP = {
        "GateDepolariseConfig": GateDepolariseConfig,
    }

    @classmethod
    def map(cls) -> Dict[str, GateNoiseConfigInterface]:
        return cls._MAP


# Config classes directly used by Topology config.


class QubitIdConfig(BaseModel):
    qubit_id: int
    qubit_config: QubitConfig

    @classmethod
    def from_dict(cls, dict: Any) -> QubitIdConfig:
        return QubitIdConfig(
            qubit_id=dict["qubit_id"],
            qubit_config=QubitConfig.from_dict(dict["qubit_config"]),
        )


class SingleGateConfig(BaseModel):
    qubit_id: int
    gate_configs: List[GateConfig]

    @classmethod
    def from_dict(cls, dict: Any) -> MultiGateConfig:
        return SingleGateConfig(
            qubit_id=dict["qubit_id"],
            gate_configs=[GateConfig.from_dict(d) for d in dict["gate_configs"]],
        )


class MultiGateConfig(BaseModel):
    qubit_ids: List[int]
    gate_configs: List[GateConfig]

    @classmethod
    def from_dict(cls, dict: Any) -> MultiGateConfig:
        return MultiGateConfig(
            qubit_ids=dict["qubit_ids"],
            gate_configs=[GateConfig.from_dict(d) for d in dict["gate_configs"]],
        )


# Topology config.


class TopologyConfig(BaseModel, LhiTopologyConfigInterface):
    qubits: List[QubitIdConfig]
    single_gates: List[SingleGateConfig]
    multi_gates: List[MultiGateConfig]

    @classmethod
    def from_file(cls, path: str) -> TopologyConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def perfect_config_uniform(
        cls,
        num_qubits: int,
        single_instructions: List[str],
        single_duration: int,
        two_instructions: List[str],
        two_duration: int,
    ) -> TopologyConfig:
        qubits = [
            QubitIdConfig(
                qubit_id=i,
                qubit_config=QubitConfig.perfect_config(is_communication=True),
            )
            for i in range(num_qubits)
        ]

        single_gates = [
            SingleGateConfig(
                qubit_id=i,
                gate_configs=[
                    GateConfig.perfect_config(name=name, duration=single_duration)
                    for name in single_instructions
                ],
            )
            for i in range(num_qubits)
        ]

        multi_gates = []
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i == j:
                    continue
                cfg = MultiGateConfig(
                    qubit_ids=[i, j],
                    gate_configs=[
                        GateConfig.perfect_config(name=name, duration=two_duration)
                        for name in two_instructions
                    ],
                )
                multi_gates.append(cfg)

        return TopologyConfig(
            qubits=qubits, single_gates=single_gates, multi_gates=multi_gates
        )

    @classmethod
    def perfect_config_uniform_default_params(cls, num_qubits: int) -> TopologyConfig:
        return cls.perfect_config_uniform(
            num_qubits=num_qubits,
            single_instructions=[
                "INSTR_INIT",
                "INSTR_ROT_X",
                "INSTR_ROT_Y",
                "INSTR_ROT_Z",
                "INSTR_X",
                "INSTR_Y",
                "INSTR_Z",
                "INSTR_H",
                "INSTR_MEASURE",
                "INSTR_MEASURE_INSTANT",
            ],
            single_duration=5e3,
            two_instructions=["INSTR_CNOT", "INSTR_CZ"],
            two_duration=200e3,
        )

    @classmethod
    def perfect_config_star(
        cls,
        num_qubits: int,
        comm_instructions: List[str],
        comm_duration: int,
        mem_instructions: List[str],
        mem_duration: int,
        two_instructions: List[str],
        two_duration: int,
    ) -> TopologyConfig:
        # comm qubit
        qubits = [
            QubitIdConfig(
                qubit_id=0,
                qubit_config=QubitConfig.perfect_config(is_communication=True),
            )
        ]
        # mem qubits
        qubits += [
            QubitIdConfig(
                qubit_id=i,
                qubit_config=QubitConfig.perfect_config(is_communication=False),
            )
            for i in range(1, num_qubits)
        ]

        # comm gate
        single_gates = [
            SingleGateConfig(
                qubit_id=0,
                gate_configs=[
                    GateConfig.perfect_config(name=name, duration=comm_duration)
                    for name in comm_instructions
                ],
            )
        ]
        # mem gates
        single_gates += [
            SingleGateConfig(
                qubit_id=i,
                gate_configs=[
                    GateConfig.perfect_config(name=name, duration=mem_duration)
                    for name in mem_instructions
                ],
            )
            for i in range(1, num_qubits)
        ]

        multi_gates = [
            MultiGateConfig(
                qubit_ids=[0, i],
                gate_configs=[
                    GateConfig.perfect_config(name=name, duration=two_duration)
                    for name in two_instructions
                ],
            )
            for i in range(1, num_qubits)
        ]

        return TopologyConfig(
            qubits=qubits, single_gates=single_gates, multi_gates=multi_gates
        )

    @classmethod
    def from_dict(cls, dict: Any) -> TopologyConfig:
        raw_qubits = dict["qubits"]
        qubits = [QubitIdConfig.from_dict(d) for d in raw_qubits]
        raw_single_gates = dict["single_gates"]
        single_gates = [SingleGateConfig.from_dict(d) for d in raw_single_gates]
        raw_multi_gates = dict["multi_gates"]
        multi_gates = [MultiGateConfig.from_dict(d) for d in raw_multi_gates]
        return TopologyConfig(
            qubits=qubits, single_gates=single_gates, multi_gates=multi_gates
        )

    def get_qubit_configs(self) -> Dict[int, LhiQubitConfigInterface]:
        infos: Dict[int, LhiQubitConfigInterface] = {}
        for cfg in self.qubits:
            infos[cfg.qubit_id] = cfg.qubit_config
        return infos

    def get_single_gate_configs(self) -> Dict[int, List[LhiGateConfigInterface]]:
        infos: Dict[int, LhiGateConfigInterface] = {}
        for cfg in self.single_gates:
            infos[cfg.qubit_id] = cfg.gate_configs
        return infos

    def get_multi_gate_configs(
        self,
    ) -> Dict[MultiQubit, List[LhiGateConfigInterface]]:
        infos: Dict[Tuple[int, ...], LhiGateConfigInterface] = {}
        for cfg in self.multi_gates:
            infos[MultiQubit(cfg.qubit_ids)] = cfg.gate_configs
        return infos


class GenericQDeviceConfig(BaseModel):
    # total number of qubits
    num_qubits: int = 2
    # number of communication qubits
    num_comm_qubits: int = 2

    # coherence times (same for each qubit)
    T1: int = 10_000_000_000
    T2: int = 1_000_000_000

    # gate execution times
    init_time: int = 10_000
    single_qubit_gate_time: int = 1_000
    two_qubit_gate_time: int = 100_000
    measure_time: int = 10_000

    # noise model
    single_qubit_gate_depolar_prob: float = 0.0
    two_qubit_gate_depolar_prob: float = 0.01

    @classmethod
    def from_file(cls, path: str) -> GenericQDeviceConfig:
        return _from_file(path, GenericQDeviceConfig)  # type: ignore

    @classmethod
    def perfect_config(cls, num_qubits: int) -> GenericQDeviceConfig:
        cfg = GenericQDeviceConfig(num_qubits=num_qubits, num_comm_qubits=num_qubits)
        cfg.T1 = 0
        cfg.T2 = 0
        cfg.single_qubit_gate_depolar_prob = 0.0
        cfg.two_qubit_gate_depolar_prob = 0.0
        return cfg


class NVQDeviceConfig(BaseModel):
    # number of qubits per NV
    num_qubits: int = 2

    # initialization error of the electron spin
    electron_init_depolar_prob: float = 0.05

    # error of the single-qubit gate
    electron_single_qubit_depolar_prob: float = 0.0

    # measurement errors (prob_error_X is the probability that outcome X is flipped to 1 - X)
    prob_error_0: float = 0.05
    prob_error_1: float = 0.005

    # initialization error of the carbon nuclear spin
    carbon_init_depolar_prob: float = 0.05

    # error of the Z-rotation gate on the carbon nuclear spin
    carbon_z_rot_depolar_prob: float = 0.001

    # error of the native NV two-qubit gate
    ec_gate_depolar_prob: float = 0.008

    # coherence times
    electron_T1: int = 1_000_000_000
    electron_T2: int = 300_000_000
    carbon_T1: int = 150_000_000_000
    carbon_T2: int = 1_500_000_000

    # gate execution times
    carbon_init: int = 310_000
    carbon_rot_x: int = 500_000
    carbon_rot_y: int = 500_000
    carbon_rot_z: int = 500_000
    electron_init: int = 2_000
    electron_rot_x: int = 5_000
    electron_rot_y: int = 5_000
    electron_rot_z: int = 5_000
    ec_controlled_dir_x: int = 500_000
    ec_controlled_dir_y: int = 500_000
    measure: int = 3_700

    @classmethod
    def from_file(cls, path: str) -> NVQDeviceConfig:
        return _from_file(path, NVQDeviceConfig)  # type: ignore

    @classmethod
    def perfect_config(cls, num_qubits: int) -> NVQDeviceConfig:
        # get default config
        cfg = NVQDeviceConfig(num_qubits=num_qubits)

        # set all error params to 0
        cfg.electron_init_depolar_prob = 0
        cfg.electron_single_qubit_depolar_prob = 0
        cfg.prob_error_0 = 0
        cfg.prob_error_1 = 0
        cfg.carbon_init_depolar_prob = 0
        cfg.carbon_z_rot_depolar_prob = 0
        cfg.ec_gate_depolar_prob = 0
        return cfg


class LatenciesConfig(BaseModel, LhiLatenciesConfigInterface):
    host_instr_time: float = 0.0  # duration of classical Host instr execution
    qnos_instr_time: float = 0.0  # duration of classical Qnos instr execution
    host_peer_latency: float = 0.0  # processing time for Host messages from remote node

    @classmethod
    def from_file(cls, path: str) -> LatenciesConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> LatenciesConfig:
        host_instr_time = 0.0
        if "host_instr_time" in dict:
            host_instr_time = dict["host_instr_time"]
        qnos_instr_time = 0.0
        if "qnos_instr_time" in dict:
            qnos_instr_time = dict["qnos_instr_time"]
        host_peer_latency = 0.0
        if "host_peer_latency" in dict:
            host_peer_latency = dict["host_peer_latency"]
        return LatenciesConfig(
            host_instr_time=host_instr_time,
            qnos_instr_time=qnos_instr_time,
            host_peer_latency=host_peer_latency,
        )

    def get_host_instr_time(self) -> float:
        return self.host_instr_time

    def get_qnos_instr_time(self) -> float:
        return self.qnos_instr_time

    def get_host_peer_latency(self) -> float:
        return self.host_peer_latency


class ProcNodeConfig(BaseModel):
    node_name: str
    node_id: int
    topology: TopologyConfig
    latencies: LatenciesConfig

    @classmethod
    def from_file(cls, path: str) -> ProcNodeConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> ProcNodeConfig:
        node_name = dict["node_name"]
        node_id = dict["node_id"]
        topology = TopologyConfig.from_dict(dict["topology"])
        latencies = LatenciesConfig.from_dict(dict["latencies"])
        return ProcNodeConfig(
            node_name=node_name, node_id=node_id, topology=topology, latencies=latencies
        )


class DepolariseOldLinkConfig(BaseModel):
    fidelity: float
    prob_success: float
    t_cycle: float

    @classmethod
    def from_file(cls, path: str) -> DepolariseOldLinkConfig:
        return _from_file(path, DepolariseOldLinkConfig)  # type: ignore


class NVOldLinkConfig(BaseModel):
    length_A: float
    length_B: float
    full_cycle: float
    cycle_time: float
    alpha: float

    @classmethod
    def from_file(cls, path: str) -> NVOldLinkConfig:
        return _from_file(path, NVOldLinkConfig)  # type: ignore


class HeraldedOldLinkConfig(BaseModel):
    length: float
    p_loss_init: float = 0
    p_loss_length: float = 0.25
    speed_of_light: float = 200_000
    dark_count_probability: float = 0
    detector_efficiency: float = 1.0
    visibility: float = 1.0
    num_resolving: bool = False

    @classmethod
    def from_file(cls, path: str) -> HeraldedOldLinkConfig:
        return _from_file(path, HeraldedOldLinkConfig)  # type: ignore


class OldLinkConfig(BaseModel):
    node1: str
    node2: str
    typ: str
    cfg: Any
    host_host_latency: float = 0.0
    qnos_qnos_latency: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> OldLinkConfig:
        return _from_file(path, OldLinkConfig)  # type: ignore

    @classmethod
    def perfect_config(cls, node1: str, node2: str) -> OldLinkConfig:
        return OldLinkConfig(
            node1=node1,
            node2=node2,
            typ="perfect",
            cfg=None,
            host_host_latency=0.0,
            qnos_qnos_latency=0.0,
        )


class LinkSamplerConfigInterface:
    @abstractclassmethod
    def from_dict(cls, dict: Any) -> LinkSamplerConfigInterface:
        raise NotImplementedError

    @abstractmethod
    def to_sampler_factory(self) -> Type[IStateDeliverySamplerFactory]:
        raise NotImplementedError

    @abstractmethod
    def to_sampler_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class PerfectSamplerConfig(LinkSamplerConfigInterface, BaseModel):
    cycle_time: float

    @classmethod
    def from_dict(cls, dict: Any) -> PerfectSamplerConfig:
        return PerfectSamplerConfig(**dict)

    def to_sampler_factory(self) -> Type[IStateDeliverySamplerFactory]:
        return PerfectStateSamplerFactory

    def to_sampler_kwargs(self) -> Dict[str, Any]:
        return {"cycle_time": self.cycle_time}


class DepolariseSamplerConfig(LinkSamplerConfigInterface, BaseModel):
    cycle_time: float
    prob_max_mixed: float
    prob_success: float

    @classmethod
    def from_dict(cls, dict: Any) -> DepolariseSamplerConfig:
        return DepolariseSamplerConfig(**dict)

    def to_sampler_factory(self) -> Type[IStateDeliverySamplerFactory]:
        return DepolariseWithFailureStateSamplerFactory

    def to_sampler_kwargs(self) -> Dict[str, Any]:
        return {
            "cycle_time": self.cycle_time,
            "prob_max_mixed": self.prob_max_mixed,
            "prob_success": self.prob_success,
        }


class DefaultSamplerConfigRegistry(SamplerFactoryRegistry):
    _MAP = {
        "PerfectSamplerConfig": PerfectSamplerConfig,
        "DepolariseSamplerConfig": DepolariseSamplerConfig,
    }

    @classmethod
    def map(cls) -> Dict[str, LinkSamplerConfigInterface]:
        return cls._MAP


class LinkConfig(LhiLinkConfigInterface, BaseModel):
    state_delay: float
    sampler_config_cls: str
    sampler_config: LinkSamplerConfigInterface

    @classmethod
    def from_file(cls, path: str) -> LinkConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def perfect_config(cls, state_delay: float) -> LinkConfig:
        return LinkConfig(
            state_delay=state_delay,
            sampler_config_cls="PerfectSamplerConfig",
            sampler_config=PerfectSamplerConfig(cycle_time=0),
        )

    @classmethod
    def depolarise_config(cls, fidelity: float, state_delay: float) -> LinkConfig:
        prob_max_mixed = fidelity_to_prob_max_mixed(fidelity)
        sampler_config = DepolariseSamplerConfig(
            cycle_time=0, prob_max_mixed=prob_max_mixed, prob_success=1
        )
        return LinkConfig(
            state_delay=state_delay,
            sampler_config_cls="DepolariseSamplerConfig",
            sampler_config=sampler_config,
        )

    @classmethod
    def from_dict(
        cls, dict: Any, registry: Optional[List[Type[SamplerFactoryRegistry]]] = None
    ) -> LinkConfig:
        state_delay = dict["state_delay"]
        raw_typ = dict["sampler_config_cls"]

        # Try to get the type of the sampler config class.
        typ: Optional[LinkSamplerConfigInterface] = None
        # First try custom registries.
        if registry is not None:
            try:
                for reg in registry:
                    if raw_typ in reg.map():
                        typ = reg.map()[raw_typ]
                        break
            except KeyError:
                pass
        # If not found in custom registries, try default registry.
        if typ is None:
            try:
                typ = DefaultSamplerConfigRegistry.map()[raw_typ]
            except KeyError:
                raise RuntimeError("invalid sampler config type")

        raw_sampler_config = dict["sampler_config"]
        sampler_config = typ.from_dict(raw_sampler_config)

        return LinkConfig(
            state_delay=state_delay,
            sampler_config_cls=raw_typ,
            sampler_config=sampler_config,
        )

    def to_sampler_factory(self) -> Type[IStateDeliverySamplerFactory]:
        return self.sampler_config.to_sampler_factory()

    def to_sampler_kwargs(self) -> Dict[str, Any]:
        return self.sampler_config.to_sampler_kwargs()

    def to_state_delay(self) -> float:
        return self.state_delay


class LinkBetweenNodesConfig(BaseModel):
    node_id1: int
    node_id2: int
    link_config: LinkConfig

    @classmethod
    def from_file(cls, path: str) -> LinkBetweenNodesConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> LinkBetweenNodesConfig:
        return LinkBetweenNodesConfig(
            node_id1=dict["node_id1"],
            node_id2=dict["node_id2"],
            link_config=LinkConfig.from_dict(dict["link_config"]),
        )


class ProcNodeNetworkConfig(BaseModel):
    nodes: List[ProcNodeConfig]
    links: List[LinkBetweenNodesConfig]

    @classmethod
    def from_file(cls, path: str) -> ProcNodeNetworkConfig:
        return _from_file(path, ProcNodeNetworkConfig)  # type: ignore

    @classmethod
    def from_nodes_perfect_links(
        cls, nodes: List[ProcNodeConfig], link_duration: float
    ) -> ProcNodeNetworkConfig:
        links: List[LinkBetweenNodesConfig] = []
        for node1, node2 in itertools.combinations(nodes, 2):
            links.append(
                LinkBetweenNodesConfig(
                    node_id1=node1.node_id,
                    node_id2=node2.node_id,
                    link_config=LinkConfig.perfect_config(link_duration),
                )
            )
        return ProcNodeNetworkConfig(nodes=nodes, links=links)
