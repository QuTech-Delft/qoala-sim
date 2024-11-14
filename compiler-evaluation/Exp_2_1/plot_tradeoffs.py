import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
from argparse import ArgumentParser
import dacite
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@dataclass
class DataPoint:
    naive: bool
    succ_prob: float
    makespan: float
    param_name: str
    param_value: float
    succ_std_dev: float


@dataclass
class DataMeta:
    timestamp: str
    num_iterations: int
    theta0: float
    theta1: float
    theta2: float
    theta3: float
    t1: float
    t2: float
    cc: float
    single_gate_fid: float
    single_gate_duration: float
    qnos_instr_time: float
    sim_duration: float
    param_name: str


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


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    return dacite.from_dict(Data, all_data)


def create_plots(timestamp: str, data: Data):
    """

    :param data:
    :param y_axis: Either succ_prob or makespan
    """

    default_suboptimal_makespan = 2480580.0
    default_optimal_makespan = 1145580.0

    # Parse data from data object to get lists of x and y values
    data_points = data.data_points
    naive_y_vals_makespan = [
        data_point.makespan for data_point in data_points if data_point.naive
    ]
    opt_y_vals_makespan = [
        data_point.makespan for data_point in data_points if not data_point.naive
    ]

    naive_y_vals_succ_prob = [
        data_point.succ_prob for data_point in data_points if data_point.naive
    ]
    opt_y_vals_succ_prob = [
        data_point.succ_prob for data_point in data_points if not data_point.naive
    ]

    naive_y_vals_succ_prob_errors = [
        data_point.succ_std_dev for data_point in data_points if data_point.naive
    ]
    opt_y_vals_succ_prob_errors = [
        data_point.succ_std_dev for data_point in data_points if not data_point.naive
    ]

    naive_x_vals = [
        data_point.param_value / default_suboptimal_makespan for data_point in data_points if data_point.naive
    ]
    opt_x_vals = [
        data_point.param_value / default_suboptimal_makespan for data_point in data_points if not data_point.naive
    ]

    # Create labels for axes
    succ_label = "Success probability of quantum program"
    makespan_label = "Makespan of quantum programs (ns)"

    xlabel = ""
    x_axis = data.meta.param_name
    if x_axis == "g_fid":
        xlabel = "Quantum gate fidelity"
    elif x_axis == "g_dur":
        xlabel = "Quantum gate duration (ns)"
    elif x_axis == "q_mem":
        xlabel = "Quantum memory decoherence time (ns)"
    elif x_axis == "cc_dur":
        xlabel = "Classical communication latency (ns)"
    elif x_axis == "instr_time":
        xlabel = "Quantum instruction processing time (ns)"

    # Plot Makespan
    plt.xscale("log")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(makespan_label, fontsize=12)

    # Fix ticks
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.yscale("log")
    plt.ylim(1e4, 1e9)
    # if x_axis == "cc_dur":
    #     plt.ylim(0,9e8)
    # elif x_axis == "instr_time":
    #     plt.ylim(0, 3e7)
    # else:
    #     plt.ylim(0,8e6)

    plt.plot(
        naive_x_vals, naive_y_vals_makespan, label="Suboptimal program", marker="o"
    )
    plt.plot(opt_x_vals, opt_y_vals_makespan, label="Optimal program", marker="s")

    plt.legend(loc="upper left", fontsize=11)

    create_png("LAST_" + x_axis + "_makespan")
    create_png(timestamp + "_" + x_axis + "_makespan")
    plt.cla()

    # Plot Success Probability
    plt.xscale("log")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(succ_label, fontsize=12)

    plt.ylim(0, 1.1)

    plt.errorbar(
        x=naive_x_vals,
        y=naive_y_vals_succ_prob,
        yerr=naive_y_vals_succ_prob_errors,
        label="Suboptimal program",
        marker="o",
        capsize=6,
    )
    plt.errorbar(
        x=opt_x_vals,
        y=opt_y_vals_succ_prob,
        yerr=opt_y_vals_succ_prob_errors,
        label="Optimal program",
        marker="s",
        capsize=6,
    )

    plt.legend(loc="lower right", fontsize=11)

    create_png("LAST_" + x_axis + "_succ_prob")
    create_png(timestamp + "_" + x_axis + "_succ_prob")
    plt.cla()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = ArgumentParser()
    parser.add_argument("--filenames", "-f", type=str, nargs="+", required=True)

    args = parser.parse_args()
    filenames = args.filenames

    for file in filenames:
        data = load_data(file)
        create_plots(timestamp, data)
