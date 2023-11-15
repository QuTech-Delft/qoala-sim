import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import dacite
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DataPoint:
    num_teleport: int
    num_local: int
    tel_succ_prob: float
    tel_succ_prob_lower: float
    tel_succ_prob_upper: float
    loc_succ_prob: float
    loc_succ_prob_lower: float
    loc_succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    num_teleport: int
    num_local: int
    num_runs: int
    t1: float
    t2: float
    latency_factor: float
    cc_latency: float
    local_busy_duration: float
    network_bin_len: int
    network_period: int
    network_first_bin: int
    num_qubits_bob: int


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


def create_meta(filename: str, datas: List[Data], plot_tel: bool):
    output_dir = relative_to_cwd("plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.json")
    meta = {}
    meta["datafiles"] = [
        {
            "num_teleport": data.meta.num_teleport,
            "num_local": data.meta.num_local,
            "timestamp": data.meta.timestamp,
        }
        for data in datas
    ]
    if plot_tel:
        meta["plotted_succ_prob"] = "teleport"
    else:
        meta["plotted_succ_prob"] = "local"
    with open(output_path, "w") as metafile:
        json.dump(meta, metafile)


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    # assert isinstance(all_data, list)
    # return [dacite.from_dict(DataPoint, entry) for entry in all_data]
    return dacite.from_dict(Data, all_data)


def plot_heatmap(
    timestamp: str, datas: List[Data], num_teleport: int, num_local: int, plot_tel: bool
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Number of teleportation iterations", fontsize=14)
    ax.set_ylabel("Number of local iterations", fontsize=14)

    plot_data = np.empty((num_local, num_teleport))

    for data in datas:
        tel = data.meta.num_teleport
        loc = data.meta.num_local
        tel_succ = data.data_points[0].tel_succ_prob
        loc_succ = data.data_points[0].loc_succ_prob
        if plot_tel:
            plot_data[loc - 1][tel - 1] = tel_succ
        else:
            plot_data[loc - 1][tel - 1] = loc_succ

    plt.pcolor(plot_data, cmap="viridis")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    if plot_tel:
        ax.set_title("Teleportation success probability", wrap=True, fontsize=14)
    else:
        ax.set_title("Local success probability", wrap=True, fontsize=14)

    ax.set_xticks(np.arange(0.5, num_teleport + 0.5), range(1, num_teleport + 1))
    ax.set_yticks(np.arange(0.5, num_local + 0.5), range(1, num_local + 1))

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    tel_or_loc = "tel" if plot_tel else "loc"
    create_png(f"LAST_{tel_or_loc}")
    create_png(f"{timestamp}_{tel_or_loc}")


def heat_map(num_teleport: int, num_local: int):
    datas: List[Data] = []
    for tel in range(1, num_teleport + 1):
        for loc in range(1, num_local + 1):
            data = load_data(f"data/sweep_teleport_local_{tel}_{loc}/LAST.json")
            datas.append(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot Teleportation success probability
    create_meta("LAST_tel_meta", datas, plot_tel=True)
    create_meta(f"{timestamp}_tel_meta", datas, plot_tel=True)
    plot_heatmap(timestamp, datas, num_teleport, num_local, plot_tel=True)

    # Plot Local success probability
    create_meta("LAST_loc_meta", datas, plot_tel=False)
    create_meta(f"{timestamp}_loc_meta", datas, plot_tel=False)
    plot_heatmap(timestamp, datas, num_teleport, num_local, plot_tel=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_teleport", "-t", type=int, required=True)
    parser.add_argument("--num_local", "-l", type=int, required=True)

    args = parser.parse_args()

    num_teleport = args.num_teleport
    num_local = args.num_local

    heat_map(num_teleport, num_local)
