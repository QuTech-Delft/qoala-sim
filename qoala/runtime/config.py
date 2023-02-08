from __future__ import annotations

from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import yaml
from netsquid.components.instructions import (
    INSTR_CNOT,
    INSTR_CZ,
    INSTR_INIT,
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
from pydantic import BaseModel as PydanticBaseModel

from qoala.runtime.lhi import (
    LhiGateConfigInterface,
    LhiQubitConfigInterface,
    LhiTopologyConfigInterface,
    MultiQubit,
)


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


_NS_INSTR_MAP = {
    "INSTR_CNOT": INSTR_CNOT,
    "INSTR_CZ": INSTR_CZ,
    "INSTR_INIT": INSTR_INIT,
    "INSTR_X": INSTR_X,
    "INSTR_Y": INSTR_Y,
    "INSTR_Z": INSTR_Z,
}


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


_DEFAULT_QUBIT_REGISTRY: Dict[str, LhiQubitConfigInterface] = {
    "QubitT1T2Config": QubitT1T2Config,
}


class QubitConfig(LhiQubitConfigInterface, BaseModel):
    is_communication: bool
    noise_config_cls: str
    noise_config: QubitNoiseConfigInterface

    @classmethod
    def from_file(cls, path: str) -> QubitConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> QubitConfig:
        is_communication = dict["is_communication"]
        raw_typ = dict["noise_config_cls"]
        typ: QubitNoiseConfigInterface = _DEFAULT_QUBIT_REGISTRY[raw_typ]
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


_DEFAULT_GATE_REGISTRY: Dict[str, LhiGateConfigInterface] = {
    "GateDepolariseConfig": GateDepolariseConfig,
}


class GateConfig(LhiGateConfigInterface, BaseModel):
    name: str
    noise_config_cls: str
    noise_config: GateNoiseConfigInterface

    @classmethod
    def from_file(cls, path: str) -> GateConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> GateConfig:
        name = dict["name"]
        raw_typ = dict["noise_config_cls"]
        typ: GateNoiseConfigInterface = _DEFAULT_GATE_REGISTRY[raw_typ]
        raw_noise_config = dict["noise_config"]
        noise_config = typ.from_dict(raw_noise_config)
        return GateConfig(
            name=name, noise_config_cls=raw_typ, noise_config=noise_config
        )

    def to_instruction(self) -> Type[NetSquidInstruction]:
        return _NS_INSTR_MAP[self.name]

    def to_duration(self) -> int:
        return self.noise_config.to_duration()

    def to_error_model(self) -> Type[QuantumErrorModel]:
        return self.noise_config.to_error_model()

    def to_error_model_kwargs(self) -> Dict[str, Any]:
        return self.noise_config.to_error_model_kwargs()


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


class ProcNodeConfig(BaseModel):
    name: str
    node_id: int
    qdevice_typ: str
    qdevice_cfg: Any
    host_qnos_latency: float = 0.0
    instr_latency: float = 0.0
    receive_latency: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> ProcNodeConfig:
        return _from_file(path, ProcNodeConfig)  # type: ignore

    @classmethod
    def perfect_generic_config(
        cls, name: str, node_id: int, num_qubits: int
    ) -> ProcNodeConfig:
        return ProcNodeConfig(
            name=name,
            node_id=node_id,
            qdevice_typ="generic",
            qdevice_cfg=GenericQDeviceConfig.perfect_config(num_qubits),
            host_qnos_latency=0.0,
            instr_latency=0.0,
            receive_latency=0.0,
        )


class DepolariseLinkConfig(BaseModel):
    fidelity: float
    prob_success: float
    t_cycle: float

    @classmethod
    def from_file(cls, path: str) -> DepolariseLinkConfig:
        return _from_file(path, DepolariseLinkConfig)  # type: ignore


class NVLinkConfig(BaseModel):
    length_A: float
    length_B: float
    full_cycle: float
    cycle_time: float
    alpha: float

    @classmethod
    def from_file(cls, path: str) -> NVLinkConfig:
        return _from_file(path, NVLinkConfig)  # type: ignore


class HeraldedLinkConfig(BaseModel):
    length: float
    p_loss_init: float = 0
    p_loss_length: float = 0.25
    speed_of_light: float = 200_000
    dark_count_probability: float = 0
    detector_efficiency: float = 1.0
    visibility: float = 1.0
    num_resolving: bool = False

    @classmethod
    def from_file(cls, path: str) -> HeraldedLinkConfig:
        return _from_file(path, HeraldedLinkConfig)  # type: ignore


class LinkConfig(BaseModel):
    node1: str
    node2: str
    typ: str
    cfg: Any
    host_host_latency: float = 0.0
    qnos_qnos_latency: float = 0.0

    @classmethod
    def from_file(cls, path: str) -> LinkConfig:
        return _from_file(path, LinkConfig)  # type: ignore

    @classmethod
    def perfect_config(cls, node1: str, node2: str) -> LinkConfig:
        return LinkConfig(
            node1=node1,
            node2=node2,
            typ="perfect",
            cfg=None,
            host_host_latency=0.0,
            qnos_qnos_latency=0.0,
        )


class ProcNodeNetworkConfig(BaseModel):
    nodes: List[ProcNodeConfig]
    links: List[LinkConfig]

    @classmethod
    def from_file(cls, path: str) -> ProcNodeNetworkConfig:
        return _from_file(path, ProcNodeNetworkConfig)  # type: ignore
