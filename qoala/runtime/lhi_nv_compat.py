from __future__ import annotations

from typing import Dict, List

from netsquid.components.instructions import (
    INSTR_CXDIR,
    INSTR_CYDIR,
    INSTR_INIT,
    INSTR_MEASURE,
    INSTR_ROT_X,
    INSTR_ROT_Y,
    INSTR_ROT_Z,
)
from netsquid.components.models.qerrormodels import DepolarNoiseModel, T1T2NoiseModel

from qoala.lang.common import MultiQubit

# Ignore type since whole 'config' module is ignored by mypy
from qoala.runtime.config import NVQDeviceConfig  # type: ignore
from qoala.runtime.lhi import LhiGateInfo, LhiQubitInfo, LhiTopology


class LhiTopologyBuilderForOldNV:
    @classmethod
    def from_nv_config(cls, cfg: NVQDeviceConfig) -> LhiTopology:
        qubit_infos: Dict[int, LhiQubitInfo] = {}

        comm_info = LhiQubitInfo(
            is_communication=True,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": cfg.electron_T1, "T2": cfg.electron_T2},
        )
        mem_info = LhiQubitInfo(
            is_communication=False,
            error_model=T1T2NoiseModel,
            error_model_kwargs={"T1": cfg.carbon_T1, "T2": cfg.carbon_T2},
        )
        qubit_infos[0] = comm_info
        for i in range(1, cfg.num_qubits):
            qubit_infos[i] = mem_info

        single_gate_infos: Dict[int, List[LhiGateInfo]] = {}
        comm_gates: List[LhiGateInfo] = []
        comm_gates += [
            LhiGateInfo(
                instruction=INSTR_INIT,
                duration=cfg.electron_init,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": cfg.electron_init_depolar_prob},
            )
        ]
        comm_gates += [
            # TODO: how to implement 'hack for imperfect measurements'?
            # (see bulid_nv_qprocessor in build.py)
            LhiGateInfo(
                instruction=INSTR_MEASURE,
                duration=cfg.measure,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": 0},
            )
        ]
        comm_gates += [
            LhiGateInfo(
                instruction=instr,
                duration=dur,
                error_model=DepolarNoiseModel,
                error_model_kwargs={
                    "depolar_rate": cfg.electron_single_qubit_depolar_prob
                },
            )
            for instr, dur in zip(
                [INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z],
                [cfg.electron_rot_x, cfg.electron_rot_y, cfg.electron_rot_z],
            )
        ]
        mem_gates: List[LhiGateInfo] = []
        mem_gates += [
            LhiGateInfo(
                instruction=INSTR_INIT,
                duration=cfg.carbon_init,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": cfg.carbon_init_depolar_prob},
            )
        ]
        mem_gates += [
            LhiGateInfo(
                instruction=instr,
                duration=dur,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": cfg.carbon_z_rot_depolar_prob},
            )
            for instr, dur in zip(
                [INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z],
                [cfg.carbon_rot_x, cfg.carbon_rot_y, cfg.carbon_rot_z],
            )
        ]

        single_gate_infos[0] = comm_gates
        for i in range(1, cfg.num_qubits):
            single_gate_infos[i] = mem_gates

        multi_gate_infos: Dict[MultiQubit, List[LhiGateInfo]] = {}
        gate_infos: List[LhiGateInfo] = [
            LhiGateInfo(
                instruction=instr,
                duration=dur,
                error_model=DepolarNoiseModel,
                error_model_kwargs={"depolar_rate": cfg.ec_gate_depolar_prob},
            )
            for instr, dur in zip(
                [INSTR_CXDIR, INSTR_CYDIR],
                [cfg.ec_controlled_dir_x, cfg.ec_controlled_dir_y],
            )
        ]

        for i in range(1, cfg.num_qubits):
            multi = MultiQubit([0, i])
            multi_gate_infos[multi] = gate_infos

        return LhiTopology(
            qubit_infos=qubit_infos,
            single_gate_infos=single_gate_infos,
            multi_gate_infos=multi_gate_infos,
        )
