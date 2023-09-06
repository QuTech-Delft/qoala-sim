# type: ignore
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
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
from qoala.runtime.instructions import (
    INSTR_BICHROMATIC,
    INSTR_MEASURE_ALL,
    INSTR_ROT_X_ALL,
    INSTR_ROT_Y_ALL,
    INSTR_ROT_Z_ALL,
)
from qoala.runtime.lhi import (
    INSTR_MEASURE_INSTANT,
    LhiGateConfigInterface,
    LhiLatenciesConfigInterface,
    LhiLinkConfigInterface,
    LhiNetworkScheduleConfigInterface,
    LhiNetworkTimebin,
    LhiQubitConfigInterface,
    LhiTopologyConfigInterface,
)
from qoala.runtime.ntf import (
    GenericNtf,
    NtfInterface,
    NtfInterfaceConfigInterface,
    NvNtf,
    TrappedIonNtf,
)
from qoala.util.math import fidelity_to_prob_max_mixed

GENERIC_GATES = [
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
]

GENERIC_TWO_GATES = ["INSTR_CNOT", "INSTR_CZ"]

NV_COM_GATES = [
    "INSTR_INIT",
    "INSTR_ROT_X",
    "INSTR_ROT_Y",
    "INSTR_MEASURE",
    "INSTR_MEASURE_INSTANT",
]

NV_MEM_GATES = [
    "INSTR_INIT",
    "INSTR_ROT_X",
    "INSTR_ROT_Y",
    "INSTR_ROT_Z",
    "INSTR_MEASURE",
    "INSTR_MEASURE_INSTANT",
]

NV_TWO_GATES = ["INSTR_CXDIR", "INSTR_CYDIR"]

NV_DEFAULT_COM_GATE_DURATION = int(1e5)
NV_DEFAULT_MEM_GATE_DURATION = int(1e6)
NV_DEFAULT_TWO_GATE_DURATION = int(2e6)

GENERIC_DEFAULT_ONE_GATE_DURATION = int(5e3)
GENERIC_DEFAULT_TWO_GATE_DURATION = int(200e3)

# NV_LAB_COMM_T1 = 1e6
# NV_LAB_COMM_T2 = 1e6
# NV_LAB_MEM_T1 = 1e6
# NV_LAB_MEM_T2 = 1e6

# NV_LAB_COMM_INIT_DURATION = 1e3


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class InstrConfigRegistry(ABC):
    @classmethod
    @abstractmethod
    def map(cls) -> Dict[str, NetSquidInstruction]:
        raise NotImplementedError


class QubitConfigRegistry(ABC):
    @classmethod
    @abstractmethod
    def map(cls) -> Dict[str, LhiQubitConfigInterface]:
        raise NotImplementedError


class GateConfigRegistry(ABC):
    @classmethod
    @abstractmethod
    def map(cls) -> Dict[str, GateNoiseConfigInterface]:
        raise NotImplementedError


class NtfInterfaceRegistry(ABC):
    @classmethod
    @abstractmethod
    def map(cls) -> Dict[str, NtfInterface]:
        raise NotImplementedError


class SamplerFactoryRegistry(ABC):
    @classmethod
    @abstractmethod
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
    @classmethod
    @abstractmethod
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
    def t1t2_config(cls, is_communication: bool, T1: int, T2: int) -> QubitConfig:
        return QubitConfig(
            is_communication=is_communication,
            noise_config_cls="T1T2NoiseModel",
            noise_config=QubitT1T2Config(T1=T1, T2=T2),
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
    @classmethod
    @abstractmethod
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
    def with_depolar_prob(
        cls, name: str, duration: int, depolar_prob: float
    ) -> GateConfig:
        return GateConfig(
            name=name,
            noise_config_cls="GateDepolariseConfig",
            noise_config=GateDepolariseConfig(
                duration=duration, depolarise_prob=depolar_prob
            ),
        )

    @classmethod
    def with_single_gate_fidelity(
        cls, name: str, duration: int, fidelity: float
    ) -> GateConfig:
        depolar_prob = fidelity_to_prob_max_mixed(num_qubits=1, fid=fidelity)
        return cls.with_depolar_prob(name, duration, depolar_prob)

    @classmethod
    def with_two_gate_fidelity(
        cls, name: str, duration: int, fidelity: float
    ) -> GateConfig:
        depolar_prob = fidelity_to_prob_max_mixed(num_qubits=2, fid=fidelity)
        return cls.with_depolar_prob(name, duration, depolar_prob)

    @classmethod
    def with_all_qubit_gate_fidelity(
        cls, name: str, duration: int, fidelity: float, num_qubits: int
    ) -> GateConfig:
        depolar_prob = fidelity_to_prob_max_mixed(num_qubits=num_qubits, fid=fidelity)
        return cls.with_depolar_prob(name, duration, depolar_prob)

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
        "INSTR_CXDIR": INSTR_CXDIR,
        "INSTR_CYDIR": INSTR_CYDIR,
        "INSTR_MEASURE": INSTR_MEASURE,
        "INSTR_MEASURE_INSTANT": INSTR_MEASURE_INSTANT,
        "INSTR_MEASURE_ALL": INSTR_MEASURE_ALL,
        "INSTR_ROT_X_ALL": INSTR_ROT_X_ALL,
        "INSTR_ROT_Y_ALL": INSTR_ROT_Y_ALL,
        "INSTR_ROT_Z_ALL": INSTR_ROT_Z_ALL,
        "INSTR_BICHROMATIC": INSTR_BICHROMATIC,
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


class DefaultNtfRegistry(NtfInterfaceRegistry):
    _MAP = {
        "GenericNtf": GenericNtf,
        "NvNtf": NvNtf,
        "TrappedIonNtf": TrappedIonNtf,
    }

    @classmethod
    def map(cls) -> Dict[str, NtfInterface]:
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
    def from_dict(cls, dict: Any) -> SingleGateConfig:
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


class AllQubitGateConfig(BaseModel):
    gate_configs: List[GateConfig]

    @classmethod
    def from_dict(cls, dict: Any) -> AllQubitGateConfig:
        return AllQubitGateConfig(
            gate_configs=[GateConfig.from_dict(d) for d in dict["gate_configs"]],
        )


# Topology config.


class TopologyConfig(BaseModel, LhiTopologyConfigInterface):
    qubits: List[QubitIdConfig]
    single_gates: List[SingleGateConfig]
    multi_gates: List[MultiGateConfig]
    all_qubit_gates: Optional[AllQubitGateConfig] = None

    @classmethod
    def from_file(cls, path: str) -> TopologyConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def uniform_t1t2_qubits_perfect_gates(
        cls,
        num_qubits: int,
        t1: int,
        t2: int,
        single_instructions: List[str],
        single_duration: int,
        two_instructions: List[str],
        two_duration: int,
        all_qubit_gate_instructions: List[str] = None,
        all_qubit_gate_duration: int = 0,
    ) -> TopologyConfig:
        qubits = [
            QubitIdConfig(
                qubit_id=i,
                qubit_config=QubitConfig.t1t2_config(
                    is_communication=True, T1=t1, T2=t2
                ),
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

        all_qubit_gates = None
        if all_qubit_gate_instructions is not None:
            all_qubit_gates = AllQubitGateConfig(
                gate_configs=[
                    GateConfig.perfect_config(
                        name=name, duration=all_qubit_gate_duration
                    )
                    for name in all_qubit_gate_instructions
                ]
            )

        return TopologyConfig(
            qubits=qubits,
            single_gates=single_gates,
            multi_gates=multi_gates,
            all_qubit_gates=all_qubit_gates,
        )

    @classmethod
    def perfect_config_uniform(
        cls,
        num_qubits: int,
        single_instructions: List[str],
        single_duration: int,
        two_instructions: List[str],
        two_duration: int,
        all_qubit_gate_instructions: List[str] = None,
        all_qubit_gate_duration: int = 0,
    ) -> TopologyConfig:
        return cls.uniform_t1t2_qubits_perfect_gates(
            num_qubits=num_qubits,
            t1=0,
            t2=0,
            single_instructions=single_instructions,
            single_duration=single_duration,
            two_instructions=two_instructions,
            two_duration=two_duration,
            all_qubit_gate_instructions=all_qubit_gate_instructions,
            all_qubit_gate_duration=all_qubit_gate_duration,
        )

    @classmethod
    def perfect_config_uniform_default_params(cls, num_qubits: int) -> TopologyConfig:
        return cls.perfect_config_uniform(
            num_qubits=num_qubits,
            single_instructions=GENERIC_GATES,
            single_duration=GENERIC_DEFAULT_ONE_GATE_DURATION,
            two_instructions=GENERIC_TWO_GATES,
            two_duration=GENERIC_DEFAULT_TWO_GATE_DURATION,
        )

    @classmethod
    def uniform_t1t2_qubits_perfect_gates_default_params(
        cls, num_qubits: int, t1: int, t2: int
    ) -> TopologyConfig:
        return cls.uniform_t1t2_qubits_perfect_gates(
            num_qubits=num_qubits,
            t1=t1,
            t2=t2,
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
            single_duration=GENERIC_DEFAULT_ONE_GATE_DURATION,
            two_instructions=["INSTR_CNOT", "INSTR_CZ"],
            two_duration=GENERIC_DEFAULT_TWO_GATE_DURATION,
        )

    @classmethod
    def config_star(
        cls,
        num_qubits: int,
        comm_t1: int,
        comm_t2: int,
        mem_t1: int,
        mem_t2: int,
        comm_init_fidelity: float,
        comm_meas_fidelity: float,
        comm_gate_fidelity: float,
        comm_instructions: List[str],
        comm_duration: int,
        mem_instructions: List[str],
        mem_duration: int,
        two_instructions: List[str],
        two_duration: int,
    ) -> TopologyConfig:
        pass

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
    def perfect_nv_default_params(cls, num_qubits: int) -> TopologyConfig:
        return cls.perfect_config_star(
            num_qubits=num_qubits,
            comm_instructions=NV_COM_GATES,
            comm_duration=NV_DEFAULT_COM_GATE_DURATION,
            mem_instructions=NV_MEM_GATES,
            mem_duration=NV_DEFAULT_MEM_GATE_DURATION,
            two_instructions=NV_TWO_GATES,
            two_duration=NV_DEFAULT_TWO_GATE_DURATION,
        )

    @classmethod
    def from_nv_params(cls, num_qubits: int, params: NvParams) -> TopologyConfig:
        return (
            TopologyConfigBuilder()
            .star_topology()
            .num_qubits(num_qubits)
            .default_nv_gates()
            .comm_t1(params.comm_t1)
            .comm_t2(params.comm_t2)
            .mem_t1(params.mem_t1)
            .mem_t2(params.mem_t2)
            .all_comm_gates_fidelity(params.comm_gate_fidelity)
            .all_comm_gates_duration(params.comm_gate_duration)
            .comm_gate_fidelity("INSTR_INIT", params.comm_init_fidelity)
            .comm_gate_duration("INSTR_INIT", params.comm_init_duration)
            .comm_gate_fidelity("INSTR_MEASURE", params.comm_meas_fidelity)
            .comm_gate_duration("INSTR_MEASURE", params.comm_meas_duration)
            .comm_gate_fidelity("INSTR_MEASURE_INSTANT", params.comm_meas_fidelity)
            .comm_gate_duration("INSTR_MEASURE_INSTANT", params.comm_meas_duration)
            .all_mem_gates_fidelity(params.mem_gate_fidelity)
            .all_mem_gates_duration(params.mem_gate_duration)
            .mem_gate_fidelity("INSTR_INIT", params.mem_init_fidelity)
            .mem_gate_duration("INSTR_INIT", params.mem_init_duration)
            .mem_gate_fidelity("INSTR_MEASURE", params.mem_meas_fidelity)
            .mem_gate_duration("INSTR_MEASURE", params.mem_meas_duration)
            .mem_gate_fidelity("INSTR_MEASURE_INSTANT", params.mem_meas_fidelity)
            .mem_gate_duration("INSTR_MEASURE_INSTANT", params.mem_meas_duration)
            .all_two_gates_fidelity(params.two_gate_fidelity)
            .all_two_gates_duration(params.two_gate_duration)
            .build()
        )

    # @classmethod
    # def nv_lab_setup(cls, num_qubits: int) -> TopologyConfig:
    #     # comm qubit
    #     qubits = [
    #         QubitIdConfig(
    #             qubit_id=0,
    #             qubit_config=QubitConfig.t1t2_config(
    #                 is_communication=True, T1=NV_LAB_COMM_T1, T2=NV_LAB_COMM_T2
    #             ),
    #         )
    #     ]
    #     # mem qubits
    #     qubits += [
    #         QubitIdConfig(
    #             qubit_id=i,
    #             qubit_config=QubitConfig.t1t2_config(
    #                 is_communication=False, T1=NV_LAB_MEM_T1, T2=NV_LAB_MEM_T2
    #             ),
    #         )
    #         for i in range(1, num_qubits)
    #     ]
    #     comm_gates: List[SingleGateConfig] = []
    #     comm_gates += [
    #         SingleGateConfig(
    #             qubit_id=0,
    #             gate_configs=[
    #                 GateConfig.perfect_config(
    #                     name=name, duration=NV_LAB_COMM_INIT_DURATION
    #                 )
    #                 for name in comm_instructions
    #             ],
    #         )
    #     ]

    @classmethod
    def from_dict(cls, dict: Any) -> TopologyConfig:
        raw_qubits = dict["qubits"]
        qubits = [QubitIdConfig.from_dict(d) for d in raw_qubits]
        raw_single_gates = dict["single_gates"]
        single_gates = [SingleGateConfig.from_dict(d) for d in raw_single_gates]
        raw_multi_gates = dict["multi_gates"]
        multi_gates = [MultiGateConfig.from_dict(d) for d in raw_multi_gates]
        all_qubit_gates = None
        if "all_qubit_gates" in dict:
            raw_all_qubit_gates = dict["all_qubit_gates"]
            all_qubit_gates = AllQubitGateConfig.from_dict(raw_all_qubit_gates)
        return TopologyConfig(
            qubits=qubits,
            single_gates=single_gates,
            multi_gates=multi_gates,
            all_qubit_gates=all_qubit_gates,
        )

    def get_qubit_configs(self) -> Dict[int, LhiQubitConfigInterface]:
        infos: Dict[int, LhiQubitConfigInterface] = {}
        for cfg in self.qubits:
            infos[cfg.qubit_id] = cfg.qubit_config
        return infos

    def get_single_gate_configs(self) -> Dict[int, List[LhiGateConfigInterface]]:
        infos: Dict[int, List[LhiGateConfigInterface]] = {}
        for cfg in self.single_gates:
            infos[cfg.qubit_id] = cfg.gate_configs
        return infos

    def get_multi_gate_configs(
        self,
    ) -> Dict[MultiQubit, List[LhiGateConfigInterface]]:
        infos: Dict[MultiQubit, List[LhiGateConfigInterface]] = {}
        for cfg in self.multi_gates:
            infos[MultiQubit(cfg.qubit_ids)] = cfg.gate_configs
        return infos

    def get_all_qubit_gate_configs(self) -> List[LhiGateConfigInterface]:
        if self.all_qubit_gates is None:
            return []
        return self.all_qubit_gates.gate_configs


class TopologyConfigBuilder:
    def __init__(self) -> None:
        self._num_qubits: Optional[int] = None
        self._uniform: Optional[bool] = None
        self._comm_t1: Optional[int] = None
        self._comm_t2: Optional[int] = None
        self._mem_t1: Optional[int] = None
        self._mem_t2: Optional[int] = None
        self._comm_single_gates: Optional[List[str]] = None
        self._comm_single_gate_durations: Optional[Dict[str, int]] = None
        self._comm_single_gate_fidelities: Optional[Dict[str, float]] = None
        self._mem_single_gates: Optional[List[str]] = None
        self._mem_single_gate_durations: Optional[Dict[str, int]] = None
        self._mem_single_gate_fidelities: Optional[Dict[str, float]] = None
        self._two_gates: Optional[List[str]] = None
        self._two_gate_durations: Optional[Dict[str, int]] = None
        self._two_gate_fidelities: Optional[Dict[str, float]] = None

    def _build_qubits(self) -> List[QubitIdConfig]:
        if self._uniform:
            qubits = [
                QubitIdConfig(
                    qubit_id=i,
                    qubit_config=QubitConfig.t1t2_config(
                        is_communication=True, T1=self._comm_t1, T2=self._comm_t2
                    ),
                )
                for i in range(self._num_qubits)
            ]
        else:
            # comm qubit
            qubits = [
                QubitIdConfig(
                    qubit_id=0,
                    qubit_config=QubitConfig.t1t2_config(
                        is_communication=True, T1=self._comm_t1, T2=self._comm_t2
                    ),
                )
            ]
            # mem qubits
            qubits += [
                QubitIdConfig(
                    qubit_id=i,
                    qubit_config=QubitConfig.t1t2_config(
                        is_communication=False, T1=self._mem_t1, T2=self._mem_t2
                    ),
                )
                for i in range(1, self._num_qubits)
            ]
        return qubits

    def _build_single_gates(self) -> List[SingleGateConfig]:
        single_gates: List[SingleGateConfig] = []

        if self._uniform:
            for i in range(self._num_qubits):
                gate_cfgs: List[GateConfig] = []
                for gate in self._comm_single_gates:
                    dur = self._comm_single_gate_durations[gate]
                    fid = self._comm_single_gate_fidelities[gate]
                    gate_cfgs.append(
                        GateConfig.with_single_gate_fidelity(gate, dur, fid)
                    )
                single_gates.append(
                    SingleGateConfig(qubit_id=i, gate_configs=gate_cfgs)
                )
        else:
            gate_cfgs: List[GateConfig] = []
            for gate in self._comm_single_gates:
                dur = self._comm_single_gate_durations[gate]
                fid = self._comm_single_gate_fidelities[gate]
                gate_cfgs.append(GateConfig.with_single_gate_fidelity(gate, dur, fid))
            single_gates.append(SingleGateConfig(qubit_id=0, gate_configs=gate_cfgs))
            for i in range(1, self._num_qubits):
                gate_cfgs: List[GateConfig] = []
                for gate in self._mem_single_gates:
                    dur = self._mem_single_gate_durations[gate]
                    fid = self._mem_single_gate_fidelities[gate]
                    gate_cfgs.append(
                        GateConfig.with_single_gate_fidelity(gate, dur, fid)
                    )
                single_gates.append(
                    SingleGateConfig(qubit_id=i, gate_configs=gate_cfgs)
                )

        return single_gates

    def _build_multi_gates(self) -> List[MultiGateConfig]:
        multi_gates: List[MultiGateConfig] = []

        if self._uniform:
            for i in range(self._num_qubits):
                for j in range(self._num_qubits):
                    if i == j:
                        continue
                    gate_cfgs: List[GateConfig] = []
                    for gate in self._two_gates:
                        dur = self._two_gate_durations[gate]
                        fid = self._two_gate_fidelities[gate]
                        gate_cfgs.append(
                            GateConfig.with_two_gate_fidelity(gate, dur, fid)
                        )
                    cfg = MultiGateConfig(qubit_ids=[i, j], gate_configs=gate_cfgs)
                    multi_gates.append(cfg)
        else:
            for i in range(1, self._num_qubits):
                gate_cfgs: List[GateConfig] = []
                for gate in self._two_gates:
                    dur = self._two_gate_durations[gate]
                    fid = self._two_gate_fidelities[gate]
                    gate_cfgs.append(GateConfig.with_two_gate_fidelity(gate, dur, fid))
                cfg = MultiGateConfig(qubit_ids=[0, i], gate_configs=gate_cfgs)
                multi_gates.append(cfg)

        return multi_gates

    def build(self) -> TopologyConfig:
        assert self._num_qubits is not None
        assert self._uniform is not None
        assert self._comm_t1 is not None
        assert self._comm_t2 is not None
        assert self._mem_t1 is not None
        assert self._mem_t2 is not None
        assert self._comm_single_gates is not None
        assert self._comm_single_gate_durations is not None
        assert self._comm_single_gate_fidelities is not None
        assert self._mem_single_gates is not None
        assert self._mem_single_gate_durations is not None
        assert self._mem_single_gate_fidelities is not None
        assert self._two_gates is not None
        assert self._two_gate_durations is not None
        assert self._two_gate_fidelities is not None

        qubits = self._build_qubits()
        single_gates = self._build_single_gates()
        multi_gates = self._build_multi_gates()
        return TopologyConfig(
            qubits=qubits,
            single_gates=single_gates,
            multi_gates=multi_gates,
        )

    def num_qubits(self, num: int) -> TopologyConfigBuilder:
        self._num_qubits = num
        return self

    def uniform_topology(self) -> TopologyConfigBuilder:
        self._uniform = True
        return self

    def star_topology(self) -> TopologyConfigBuilder:
        self._uniform = False
        return self

    def qubit_t1(self, t1: int) -> TopologyConfigBuilder:
        self._comm_t1 = t1
        self._mem_t1 = t1
        return self

    def qubit_t2(self, t2: int) -> TopologyConfigBuilder:
        self._comm_t2 = t2
        self._mem_t2 = t2
        return self

    def comm_t1(self, t1: int) -> TopologyConfigBuilder:
        self._comm_t1 = t1
        return self

    def comm_t2(self, t2: int) -> TopologyConfigBuilder:
        self._comm_t2 = t2
        return self

    def mem_t1(self, t1: int) -> TopologyConfigBuilder:
        self._mem_t1 = t1
        return self

    def mem_t2(self, t2: int) -> TopologyConfigBuilder:
        self._mem_t2 = t2
        return self

    def no_decoherence(self) -> TopologyConfigBuilder:
        self._comm_t1 = 0
        self._comm_t2 = 0
        self._mem_t1 = 0
        self._mem_t2 = 0
        return self

    def default_generic_gates(self) -> TopologyConfigBuilder:
        self._comm_single_gates = GENERIC_GATES
        self._mem_single_gates = GENERIC_GATES
        self._two_gates = GENERIC_TWO_GATES
        return self

    def default_nv_gates(self) -> TopologyConfigBuilder:
        self._comm_single_gates = NV_COM_GATES
        self._mem_single_gates = NV_MEM_GATES
        self._two_gates = NV_TWO_GATES
        return self

    def zero_gate_durations(self) -> TopologyConfigBuilder:
        if self._comm_single_gate_durations is None:
            self._comm_single_gate_durations = {}
        if self._mem_single_gate_durations is None:
            self._mem_single_gate_durations = {}
        if self._two_gate_durations is None:
            self._two_gate_durations = {}

        for gate in self._comm_single_gates:
            self._comm_single_gate_durations[gate] = 0
        for gate in self._mem_single_gates:
            self._mem_single_gate_durations[gate] = 0
        for gate in self._two_gates:
            self._two_gate_durations[gate] = 0
        return self

    def perfect_gate_fidelities(self) -> TopologyConfigBuilder:
        if self._comm_single_gate_fidelities is None:
            self._comm_single_gate_fidelities = {}
        if self._mem_single_gate_fidelities is None:
            self._mem_single_gate_fidelities = {}
        if self._two_gate_fidelities is None:
            self._two_gate_fidelities = {}

        for gate in self._comm_single_gates:
            self._comm_single_gate_fidelities[gate] = 1
        for gate in self._mem_single_gates:
            self._mem_single_gate_fidelities[gate] = 1
        for gate in self._two_gates:
            self._two_gate_fidelities[gate] = 1
        return self

    def all_comm_gates_fidelity(self, fidelity: float) -> TopologyConfigBuilder:
        if self._comm_single_gate_fidelities is None:
            self._comm_single_gate_fidelities = {}
        for gate in self._comm_single_gates:
            self._comm_single_gate_fidelities[gate] = fidelity
        return self

    def comm_gate_fidelity(self, gate: str, fidelity: float) -> TopologyConfigBuilder:
        if self._comm_single_gate_fidelities is None:
            self._comm_single_gate_fidelities = {}
        self._comm_single_gate_fidelities[gate] = fidelity
        return self

    def all_comm_gates_duration(self, duration: int) -> TopologyConfigBuilder:
        if self._comm_single_gate_durations is None:
            self._comm_single_gate_durations = {}
        for gate in self._comm_single_gates:
            self._comm_single_gate_durations[gate] = duration
        return self

    def comm_gate_duration(self, gate: str, duration: int) -> TopologyConfigBuilder:
        if self._comm_single_gate_durations is None:
            self._comm_single_gate_durations = {}
        self._comm_single_gate_durations[gate] = duration
        return self

    def all_mem_gates_fidelity(self, fidelity: float) -> TopologyConfigBuilder:
        if self._mem_single_gate_fidelities is None:
            self._mem_single_gate_fidelities = {}
        for gate in self._mem_single_gates:
            self._mem_single_gate_fidelities[gate] = fidelity
        return self

    def mem_gate_fidelity(self, gate: str, fidelity: float) -> TopologyConfigBuilder:
        if self._mem_single_gate_fidelities is None:
            self._mem_single_gate_fidelities = {}
        self._mem_single_gate_fidelities[gate] = fidelity
        return self

    def all_mem_gates_duration(self, duration: int) -> TopologyConfigBuilder:
        if self._mem_single_gate_durations is None:
            self._mem_single_gate_durations = {}
        for gate in self._mem_single_gates:
            self._mem_single_gate_durations[gate] = duration
        return self

    def mem_gate_duration(self, gate: str, duration: int) -> TopologyConfigBuilder:
        if self._mem_single_gate_durations is None:
            self._mem_single_gate_durations = {}
        self._mem_single_gate_durations[gate] = duration
        return self

    def all_two_gates_fidelity(self, fidelity: float) -> TopologyConfigBuilder:
        if self._two_gate_fidelities is None:
            self._two_gate_fidelities = {}
        for gate in self._two_gates:
            self._two_gate_fidelities[gate] = fidelity
        return self

    def two_gate_fidelity(self, gate: str, fidelity: float) -> TopologyConfigBuilder:
        if self._two_gate_fidelities is None:
            self._two_gate_fidelities = {}
        self._two_gate_fidelities[gate] = fidelity
        return self

    def all_two_gates_duration(self, duration: int) -> TopologyConfigBuilder:
        if self._two_gate_durations is None:
            self._two_gate_durations = {}
        for gate in self._two_gates:
            self._two_gate_durations[gate] = duration
        return self

    def two_gate_duration(self, gate: str, duration: int) -> TopologyConfigBuilder:
        if self._two_gate_durations is None:
            self._two_gate_durations = {}
        self._two_gate_durations[gate] = duration
        return self


# TODO: actually use this class (currently ProcNodeConfig just uses a string for ntf)
class NtfConfig(NtfInterfaceConfigInterface, BaseModel):
    ntf_interface_cls: str
    ntf_interface: Type[NtfInterface]

    @classmethod
    def from_file(
        cls, path: str, registry: Optional[List[Type[NtfInterfaceRegistry]]] = None
    ) -> NtfConfig:
        return cls.from_dict(_read_dict(path), registry)

    @classmethod
    def from_cls_name(
        cls, name: str, registry: Optional[List[Type[NtfInterfaceRegistry]]] = None
    ) -> NtfConfig:
        # Try to get the type of the ntf interface class.
        typ: Optional[NtfInterface] = None
        if registry is not None:
            try:
                for reg in registry:
                    if name in reg.map():
                        typ = reg.map()[name]
                        break
            except KeyError:
                pass
        # If not found in custom registries, try default registry.
        if typ is None:
            try:
                typ = DefaultNtfRegistry.map()[name]
            except KeyError:
                raise RuntimeError("invalid NTF interface type")

        return NtfConfig(ntf_interface_cls=name, ntf_interface=typ)

    @classmethod
    def from_dict(
        cls, dict: Any, registry: Optional[List[Type[NtfInterfaceRegistry]]] = None
    ) -> NtfConfig:
        raw_typ = dict["ntf_interface_cls"]
        return cls.from_cls_name(raw_typ, registry)

    def to_ntf_interface(self) -> Type[NtfInterface]:
        return self.ntf_interface


@dataclass
class NvParams:
    # Communication qubit coherence
    comm_t1: float = 0.0
    comm_t2: float = 0.0

    # Communication qubit gate noise
    comm_init_fidelity: float = 1.0
    comm_meas_fidelity: float = 1.0
    comm_gate_fidelity: float = 1.0

    # Communication qubit gate duration
    comm_init_duration: float = 0.0
    comm_meas_duration: float = 0.0
    comm_gate_duration: float = 0.0

    # Memory qubit coherence
    mem_t1: float = 0.0
    mem_t2: float = 0.0

    # Memory qubit gate noise
    mem_init_fidelity: float = 1.0
    mem_meas_fidelity: float = 1.0
    mem_gate_fidelity: float = 1.0

    # Memory qubit gate duration
    mem_init_duration: float = 0.0
    mem_meas_duration: float = 0.0
    mem_gate_duration: float = 0.0

    # Two-qubit gate noise
    two_gate_fidelity: float = 1.0
    # Two qubit gate duration
    two_gate_duration: float = 0.0


class LatenciesConfig(BaseModel, LhiLatenciesConfigInterface):
    host_instr_time: float = 0.0  # duration of classical Host instr execution
    qnos_instr_time: float = 0.0  # duration of classical Qnos instr execution
    host_peer_latency: float = 0.0  # processing time for Host messages from remote node
    internal_sched_latency: float = 0  # processing time for messaging between node scheduler and processor schedulers

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
        internal_sched_latency = 0.0
        if "internal_sched_latency" in dict:
            host_peer_latency = dict["internal_sched_latency"]
        return LatenciesConfig(
            host_instr_time=host_instr_time,
            qnos_instr_time=qnos_instr_time,
            host_peer_latency=host_peer_latency,
            internal_sched_latency=internal_sched_latency,
        )

    def get_host_instr_time(self) -> float:
        return self.host_instr_time

    def get_qnos_instr_time(self) -> float:
        return self.qnos_instr_time

    def get_host_peer_latency(self) -> float:
        return self.host_peer_latency

    def get_internal_sched_latency(self) -> float:
        return self.internal_sched_latency


class ProcNodeConfig(BaseModel):
    node_name: str
    node_id: int
    topology: TopologyConfig
    latencies: LatenciesConfig
    ntf: NtfConfig
    determ_sched: bool = True
    use_deadlines: bool = True
    is_predictable: bool = False

    @classmethod
    def from_file(cls, path: str) -> ProcNodeConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> ProcNodeConfig:
        node_name = dict["node_name"]
        node_id = dict["node_id"]
        topology = TopologyConfig.from_dict(dict["topology"])
        latencies = LatenciesConfig.from_dict(dict["latencies"])
        ntf = NtfConfig.from_dict(dict["ntf"])
        return ProcNodeConfig(
            node_name=node_name,
            node_id=node_id,
            topology=topology,
            latencies=latencies,
            ntf=ntf,
        )


class LinkSamplerConfigInterface:
    @classmethod
    @abstractmethod
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
    def simple_depolarise_config(
        cls, fidelity: float, state_delay: float
    ) -> LinkConfig:
        prob_max_mixed = fidelity_to_prob_max_mixed(2, fidelity)
        sampler_config = DepolariseSamplerConfig(
            cycle_time=0, prob_max_mixed=prob_max_mixed, prob_success=1
        )
        return LinkConfig(
            state_delay=state_delay,
            sampler_config_cls="DepolariseSamplerConfig",
            sampler_config=sampler_config,
        )

    @classmethod
    def depolarise_config(
        cls,
        prob_max_mixed: float,
        attempt_success_prob: float,
        attempt_duration: float,
        state_delay: float,
    ) -> LinkConfig:
        sampler_config = DepolariseSamplerConfig(
            cycle_time=attempt_duration,
            prob_max_mixed=prob_max_mixed,
            prob_success=attempt_success_prob,
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


class ClassicalConnectionConfig(BaseModel):
    node_id1: int
    node_id2: int
    latency: float

    @classmethod
    def from_nodes(
        cls, node_id1: int, node_id2: int, latency: float
    ) -> ClassicalConnectionConfig:
        return ClassicalConnectionConfig(
            node_id1=node_id1, node_id2=node_id2, latency=latency
        )

    @classmethod
    def from_file(cls, path: str) -> ClassicalConnectionConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> ClassicalConnectionConfig:
        return ClassicalConnectionConfig(
            node_id1=dict["node_id1"],
            node_id2=dict["node_id2"],
            latency=dict["latency"],
        )


class NetworkScheduleConfig(BaseModel, LhiNetworkScheduleConfigInterface):
    bin_length: int
    first_bin: int
    bin_pattern: List[Tuple[int, int, int, int]]
    repeat_period: int

    @classmethod
    def from_file(cls, path: str) -> NetworkScheduleConfig:
        return cls.from_dict(_read_dict(path))

    @classmethod
    def from_dict(cls, dict: Any) -> NetworkScheduleConfig:
        return NetworkScheduleConfig(
            bin_length=dict["bin_length"],
            first_bin=dict["first_bin"],
            bin_pattern=dict["bin_pattern"],
            repeat_period=dict["repeat_period"],
        )

    def to_bin_length(self) -> int:
        return self.bin_length

    def to_first_bin(self) -> int:
        return self.first_bin

    def to_bin_pattern(self) -> List[LhiNetworkTimebin]:
        pattern = [
            LhiNetworkTimebin(frozenset({node1, node2}), {node1: pid1, node2: pid2})
            for (node1, pid1, node2, pid2) in self.bin_pattern
        ]
        return pattern

    def to_repeat_period(self) -> int:
        return self.repeat_period


class ProcNodeNetworkConfig(BaseModel):
    nodes: List[ProcNodeConfig]
    links: List[LinkBetweenNodesConfig]
    netschedule: Optional[NetworkScheduleConfig] = None
    cconns: List[ClassicalConnectionConfig] = None

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
