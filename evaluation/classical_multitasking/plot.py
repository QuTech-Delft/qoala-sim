import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt

from qoala.util.runner import SchedulerType


@dataclass
class DataPoint:
    t2: float
    sched_typ: str
    latency_factor: float
    num_const_tasks: int
    const_rate_factor: float
    busy_factor: float
    succ_prob: float
    succ_prob_lower: float
    succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    const_rate_factors: List[float]
    num_iterations: int


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


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    # assert isinstance(all_data, list)
    # return [dacite.from_dict(DataPoint, entry) for entry in all_data]
    return dacite.from_dict(Data, all_data)


def plot_sweep_const_rate(
    no_sched_data: Data, fcfs_data: Data, qoala_data: Data
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Success probability")

    for data in [no_sched_data, fcfs_data, qoala_data]:
        crf = [p.const_rate_factor for p in data.data_points]
        succ_probs = [p.succ_prob for p in data.data_points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in data.data_points]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in data.data_points]
        errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.errorbar(
            x=crf,
            y=succ_probs,
            yerr=errors,
            # fmt=FORMATS[version],
            # label=VERSION_LABELS_1[version],
        )

    ax.set_title(
        "Success probability vs CPU heaviness",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("theplot")


def sweep_const_period():
    no_sched_data = load_data("data/no_sched/LAST.json")
    fcfs_data = load_data("data/fcfs/LAST.json")
    qoala_data = load_data("data/qoala/LAST.json")
    plot_sweep_const_rate(no_sched_data, fcfs_data, qoala_data)


if __name__ == "__main__":
    sweep_const_period()
