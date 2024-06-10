import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import dacite
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@dataclass
class DataPoint:
    sched_typ: str
    busy_factor: float
    busy_duration: float
    succ_prob: float
    succ_prob_lower: float
    succ_prob_upper: float
    makespan: float


@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    num_iterations: int
    t1: float
    t2: float
    latency_factor: float
    num_const_tasks: int
    const_rate_factor: float
    cc_latency: float
    const_period: float
    const_start: float
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
    plt.savefig(output_path, transparent=True, dpi=1000)
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

    return dacite.from_dict(Data, all_data)


def sched_type_to_label(sched_type: str) -> str:
    return {"NO_SCHED": "Baseline", "FCFS": "FCFS", "QOALA": "EDF"}[sched_type]


FORMATS = {
    "NO_SCHED": "^-",
    "FCFS": "s--",
    "QOALA": "D-.",
}

COLORS = {
    "NO_SCHED": "#1F77B4",
    "FCFS": "#2CA02C",
    "QOALA": "#FF7F0E",
}


def plot_sweep_busy_factor(
    timestamp: str, no_sched_data: Data, fcfs_data: Data, qoala_data: Data
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel(
        "Classical task duration as fraction of classical communication latency",
        fontsize=12,
    )
    ax.set_ylabel("Success prob. of interactive quantum program", fontsize=12)
    ax.set_xscale("log")
    ax.tick_params(labelsize=11)

    ax2 = ax.twinx()
    ax2.set_ylabel("Makespan improvement factor (EDF)", fontsize=12)
    ax2.spines["right"].set_color("red")
    ax2.yaxis.label.set_color("red")
    ax2.tick_params(axis="y", colors="red", labelsize=11)

    lines = []

    for data in [no_sched_data, fcfs_data, qoala_data]:
        bf = [p.busy_factor for p in data.data_points]
        succ_probs = [p.succ_prob for p in data.data_points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in data.data_points]
        error_plus = [max(0, e) for e in error_plus]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in data.data_points]
        error_minus = [max(0, e) for e in error_minus]
        errors = [error_minus, error_plus]
        sched_typ = data.data_points[0].sched_typ
        line = ax.errorbar(
            x=bf,
            y=succ_probs,
            yerr=errors,
            fmt=FORMATS[sched_typ],
            color=COLORS[sched_typ],
            label=sched_type_to_label(sched_typ),
        )

        lines.append(line)

    makespan_improvements = [
        p1.makespan / p2.makespan
        for p1, p2 in zip(no_sched_data.data_points, qoala_data.data_points)
    ]
    bf = [p.busy_factor for p in no_sched_data.data_points]
    print(bf)
    lines.append(
        ax2.errorbar(
            x=bf,
            y=makespan_improvements,
            # yerr=errors,
            fmt="o:r",
            label="Makespan improv.",
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
    labels = [line.get_label() for line in lines]

    ax.legend(lines, labels, loc="lower left", fontsize=11)

    # ax.set_title(
    #     "Success probability and makespan for quantum program in presence of busy classical program",
    #     wrap=True,
    # )

    # ax.set_ylim(0.5, 1.0)

    create_png("LAST")
    create_png(timestamp)


def sweep_busy_factor():
    no_sched_data = load_data("data/no_sched/LAST.json")
    fcfs_data = load_data("data/fcfs/LAST.json")
    qoala_data = load_data("data/qoala/LAST.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", no_sched_data, fcfs_data, qoala_data)
    create_meta(f"{timestamp}_meta", no_sched_data, fcfs_data, qoala_data)
    plot_sweep_busy_factor(timestamp, no_sched_data, fcfs_data, qoala_data)


if __name__ == "__main__":
    sweep_busy_factor()
