import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    const_rate_factor: float
    num_iterations: int
    t1: float
    t2: float
    latency_factor: float
    num_const_tasks: int
    busy_factors: List[float]


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


def create_meta(filename: str, no_sched: Data, fcfs: Data, qoala: Data):
    output_dir = relative_to_cwd("plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.json")
    meta = {}
    meta["no_sched"] = no_sched.meta.timestamp
    meta["fcfs"] = fcfs.meta.timestamp
    meta["qoala"] = qoala.meta.timestamp
    with open(output_path, "w") as metafile:
        json.dump(meta, metafile)


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    # assert isinstance(all_data, list)
    # return [dacite.from_dict(DataPoint, entry) for entry in all_data]
    return dacite.from_dict(Data, all_data)


def plot_sweep_const_rate(
    timestamp: str, no_sched_data: Data, fcfs_data: Data, qoala_data: Data
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Fraction of CC latency")
    ax.set_ylabel("Success probability")

    ax2 = ax.twinx()

    for data in [no_sched_data, fcfs_data, qoala_data]:
        crf = [p.const_rate_factor for p in data.data_points]
        succ_probs = [p.succ_prob for p in data.data_points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in data.data_points]
        error_plus = [max(0, e) for e in error_plus]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in data.data_points]
        error_minus = [max(0, e) for e in error_minus]
        errors = [error_minus, error_plus]
        ax.set_xscale("log")
        ax.errorbar(
            x=crf,
            y=succ_probs,
            yerr=errors,
            # fmt=FORMATS[version],
            label=data.data_points[0].sched_typ,
        )

        makespans = [p.makespan for p in data.data_points]
        ax2.errorbar(
            x=crf,
            y=makespans,
            # yerr=errors,
            # fmt=FORMATS[version],
            label=f"makespan {data.data_points[0].sched_typ}",
        )
        # ax2.set_yscale("log")

    ax.set_title(
        "Success probability vs Fraction of CC latency",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    ax.legend(loc="upper left")

    create_png("LAST")
    create_png(timestamp)


def plot_sweep_busy_factor(
    timestamp: str, no_sched_data: Data, fcfs_data: Data, qoala_data: Data
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Fraction of CC latency")
    ax.set_ylabel("Success probability")
    ax.set_xscale("log")

    ax2 = ax.twinx()
    ax2.set_ylabel("Makespan improvement factor")

    lines = []

    for data in [no_sched_data, fcfs_data, qoala_data]:
        bf = [p.busy_factor for p in data.data_points]
        succ_probs = [p.succ_prob for p in data.data_points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in data.data_points]
        error_plus = [max(0, e) for e in error_plus]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in data.data_points]
        error_minus = [max(0, e) for e in error_minus]
        errors = [error_minus, error_plus]
        line = ax.errorbar(
            x=bf,
            y=succ_probs,
            yerr=errors,
            # fmt=FORMATS[version],
            label=data.data_points[0].sched_typ,
        )

        lines.append(line)

        # makespans = [p.makespan for p in data.data_points]
        # ax2.errorbar(
        #     x=bf,
        #     y=makespans,
        #     # yerr=errors,
        #     # fmt=FORMATS[version],
        #     label=f"makespan {data.data_points[0].sched_typ}",
        # )
        # ax2.set_yscale("log")

    makespan_improvements = [
        p2.makespan / p1.makespan
        for p1, p2 in zip(no_sched_data.data_points, qoala_data.data_points)
    ]
    bf = [p.busy_factor for p in no_sched_data.data_points]
    print(bf)
    lines.append(
        ax2.errorbar(
            x=bf,
            y=makespan_improvements,
            # yerr=errors,
            fmt="o-r",
            label=f"Makespan improvement",
        )
    )

    def format_func(value, tick_number):
        # return f"{10.0**value:.1f}"
        return f"{value}"

    # Create a FuncFormatter object using the format function
    formatter = ticker.FuncFormatter(format_func)

    # Set the x-axis formatter
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([0.1, 1, 10]))

    # ax2.set_yscale("log")
    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels, loc="lower center")

    ax.set_title(
        "Success probability vs busy task duration",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)

    create_png("LAST")
    create_png(timestamp)


def sweep_const_period():
    no_sched_data = load_data("data/no_sched/LAST.json")
    fcfs_data = load_data("data/fcfs/LAST.json")
    qoala_data = load_data("data/qoala/LAST.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta.json", no_sched_data, fcfs_data, qoala_data)
    create_meta(f"{timestamp}_meta.json", no_sched_data, fcfs_data, qoala_data)
    plot_sweep_const_rate(timestamp, no_sched_data, fcfs_data, qoala_data)


def sweep_busy_factor():
    no_sched_data = load_data("data/no_sched/LAST.json")
    fcfs_data = load_data("data/fcfs/LAST.json")
    qoala_data = load_data("data/qoala/LAST.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta.json", no_sched_data, fcfs_data, qoala_data)
    create_meta(f"{timestamp}_meta.json", no_sched_data, fcfs_data, qoala_data)
    plot_sweep_busy_factor(timestamp, no_sched_data, fcfs_data, qoala_data)


if __name__ == "__main__":
    # sweep_const_period()
    sweep_busy_factor()
