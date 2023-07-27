import json
import numpy as np
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@dataclass
class DataPoint:
    t2: float
    cc_latency: float
    num_qubits_bob: int
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
    num_iterations: int
    num_local_iterations: int
    num_runs: int
    latency_factor: float


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
            "num_teleport": data.meta.num_iterations,
            "num_local": data.meta.num_local_iterations,
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
    timestamp: str, datas: List[Data], num_range: int, plot_tel: bool
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Number of teleportation iterations")
    ax.set_ylabel("Number of local iterations")

    plot_data = np.empty((num_range, num_range))

    for data in datas:
        tel = data.meta.num_iterations
        loc = data.meta.num_local_iterations
        tel_succ = data.data_points[0].tel_succ_prob
        loc_succ = data.data_points[0].loc_succ_prob
        if plot_tel:
            plot_data[loc - 1][tel - 1] = tel_succ
        else:
            plot_data[loc - 1][tel - 1] = loc_succ

    plt.pcolor(plot_data, cmap="viridis")
    plt.colorbar()

    if plot_tel:
        ax.set_title("Teleportation success probability", wrap=True)
    else:
        ax.set_title("Local success probability", wrap=True)

    ax.set_xticks(np.arange(0.5, num_range + 0.5), range(1, num_range + 1))
    ax.set_yticks(np.arange(0.5, num_range + 0.5), range(1, num_range + 1))

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("LAST")
    create_png(timestamp)


def heat_map(num_range: int, plot_tel: bool):
    datas: List[Data] = []
    for tel in range(1, num_range + 1):
        for loc in range(1, num_range + 1):
            data = load_data(f"data/sweep_teleport_local_{tel}_{loc}/LAST.json")
            datas.append(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", datas, plot_tel)
    create_meta(f"{timestamp}_meta", datas, plot_tel)
    plot_heatmap(timestamp, datas, num_range, plot_tel)


if __name__ == "__main__":
    heat_map(10, plot_tel=True)
    # heat_map(10, plot_tel=False)
