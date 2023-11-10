import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt


@dataclass
class DataPoint:
    cc_latency: float
    num_qubits_bob: int
    tel_succ_prob: float
    tel_succ_prob_lower: float
    tel_succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    latency_factor: float
    net_bin_factors: List[float]
    prio_epr: bool
    num_qubits: int
    num_steps: int
    t1: float
    t2: float


@dataclass
class Data:
    meta: DataMeta
    data_points: List[DataPoint]


def relative_to_cwd(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)


def create_png(filename: str):
    output_dir = relative_to_cwd("plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_path)
    print(f"plot written to {output_path}")


def create_meta(filename: str, datas: Dict[str, Data]):
    output_dir = relative_to_cwd("plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.json")
    meta = {}
    meta["datafiles"] = [
        {
            "filename": datafile,
            "timestamp": data.meta.timestamp,
            "num_qubits": data.meta.num_qubits,
            "num_steps": data.meta.num_steps,
        }
        for datafile, data in datas.items()
    ]
    with open(output_path, "w") as metafile:
        json.dump(meta, metafile)


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    return dacite.from_dict(Data, all_data)


FORMATS = {
    1: "^-",
    2: "s-",
    5: "D-",
}

COLORS = {
    1: "#1F77B4",
    2: "#2CA02C",
    5: "#FF7F0E",
}


def plot_sweep_net_bin_period(timestamp: str, datas: Dict[str, Data]) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Time bin length as fraction of node-node communication latency")
    ax.set_ylabel("Makespan (ms)")

    fmts = ["o-b", "o-r", "o-k", "o-g", "s-y"]
    labels = [f"{data.meta.num_qubits} qubits" for data in datas.values()]

    nbf = [nbf for nbf in list(datas.values())[0].meta.net_bin_factors]
    for data, fmt, label in zip(datas.values(), fmts, labels):
        makespans = [p.makespan / 1e6 for p in data.data_points]
        ax.errorbar(x=nbf, y=makespans, fmt=FORMATS[data.meta.num_qubits], label=label)

    ax.set_title(
        "Teleportation makespan vs time bin length in network schedule",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    ax.legend(loc="upper left")

    create_png("LAST")
    create_png(timestamp)


def sweep_net_bin_factor_per_qubit_num(num_qubits: List[int], num_steps: int):
    datas: Dict[str, Data] = {}

    for nq in num_qubits:
        dir = f"data/sweep_bin_length_{nq}_{num_steps}"
        data = load_data(f"{dir}/LAST.json")
        data_timestamp = data.meta.timestamp
        datas[f"{dir}/{data_timestamp}.json"] = data

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", datas)
    create_meta(f"{timestamp}_meta", datas)
    plot_sweep_net_bin_period(timestamp, datas)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_qubits", "-q", type=int, nargs="+", required=True)
    parser.add_argument("--num_steps", "-s", type=int, required=True)

    args = parser.parse_args()

    num_qubits = args.num_qubits
    num_steps = args.num_steps

    sweep_net_bin_factor_per_qubit_num(num_qubits, num_steps)
