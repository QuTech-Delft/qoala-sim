import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt


@dataclass
class DataPoint:
    t2: float
    use_deadlines: bool
    latency_factor: float
    busy_factor: float
    arrival_rate: float
    succ_prob: float
    succ_prob_lower: float
    succ_prob_upper: float
    makespan: float
    succ_per_s: float
    succ_per_s_lower: float
    succ_per_s_upper: float


@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    latency_factors: List[float]
    determ: bool
    use_deadlines: bool
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


def sweep_busy_factor(data: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Success probability")

    # Data per latency factor
    lf_data: Dict[float, List[DataPoint]] = {}

    for lf in data.meta.latency_factors:
        lf_data[lf] = [p for p in data.data_points if p.latency_factor == lf]

    for lf, points in lf_data.items():
        busy_factor = [p.busy_factor for p in points]
        succ_probs = [p.succ_prob for p in points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in points]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in points]
        errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.errorbar(
            x=busy_factor,
            y=succ_probs,
            yerr=errors,
        )

    ax.set_title(
        "Success probability vs CPU heaviness",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("theplot")


def sweep_busy_factor_plot_succ_per_sec(data: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Success probability / s")

    # Data per latency factor
    lf_data: Dict[float, List[DataPoint]] = {}

    for lf in data.meta.latency_factors:
        lf_data[lf] = [p for p in data.data_points if p.latency_factor == lf]

    for lf, points in lf_data.items():
        busy_factor = [p.busy_factor for p in points]
        succ_per_sec = [p.succ_per_s for p in points]
        error_plus = [p.succ_per_s_upper - p.succ_per_s for p in points]
        error_minus = [p.succ_per_s - p.succ_per_s_lower for p in points]
        error_plus = [max(e, 0) for e in error_plus]
        error_minus = [max(e, 0) for e in error_minus]
        errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(
            x=busy_factor,
            y=succ_per_sec,
            yerr=errors,
            # fmt=FORMATS[version],
            # label=VERSION_LABELS_1[version],
        )

    ax.set_title(
        "Success probability per second vs CPU heaviness",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("themakespan")


def sweep_hog_prob(data: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Success probability")

    # Data per latency factor
    lf_data: Dict[float, List[DataPoint]] = {}

    for lf in data.meta.latency_factors:
        lf_data[lf] = [p for p in data.data_points if p.latency_factor == lf]

    for lf, points in lf_data.items():
        hog_prob = [p.hog_prob for p in points]
        succ_probs = [p.succ_prob for p in points]
        error_plus = [p.succ_prob_upper - p.succ_prob for p in points]
        error_minus = [p.succ_prob - p.succ_prob_lower for p in points]
        errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.errorbar(
            x=hog_prob,
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


def plot_improvement_factor(with_deadlines: Data, without_deadlines: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Success probability")

    # Data per latency factor
    lf_data_with_deadlines: Dict[float, List[DataPoint]] = {}
    lf_data_without_deadlines: Dict[float, List[DataPoint]] = {}

    assert with_deadlines.meta.latency_factors == without_deadlines.meta.latency_factors

    for lf in with_deadlines.meta.latency_factors:
        lf_data_with_deadlines[lf] = [
            p for p in with_deadlines.data_points if p.latency_factor == lf
        ]
        lf_data_without_deadlines[lf] = [
            p for p in without_deadlines.data_points if p.latency_factor == lf
        ]

    for lf in lf_data_with_deadlines.keys():
        points_with = lf_data_with_deadlines[lf]
        points_without = lf_data_without_deadlines[lf]
        busy_factor_with = [p.busy_factor for p in points_with]
        busy_factor_without = [p.busy_factor for p in points_without]
        assert busy_factor_with == busy_factor_without
        improv_factor = [
            p_with.succ_prob / p_without.succ_prob
            for p_with, p_without in zip(points_with, points_without)
        ]
        # succ_probs = [p.succ_prob for p in points]
        # error_plus = [p.succ_prob_upper - p.succ_prob for p in points]
        # error_minus = [p.succ_prob - p.succ_prob_lower for p in points]
        # errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.errorbar(
            x=busy_factor_with,
            y=improv_factor,
            # yerr=errors,
            # fmt=FORMATS[version],
            # label=VERSION_LABELS_1[version],
        )

    ax.set_title(
        "Success probability vs CPU heaviness",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("theimprovement")


def plot_improvement_factor_per_sec(
    with_deadlines: Data, without_deadlines: Data
) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("CPU heaviness")
    ax.set_ylabel("Improvement factor")

    # Data per latency factor
    lf_data_with_deadlines: Dict[float, List[DataPoint]] = {}
    lf_data_without_deadlines: Dict[float, List[DataPoint]] = {}

    assert with_deadlines.meta.latency_factors == without_deadlines.meta.latency_factors

    for lf in with_deadlines.meta.latency_factors:
        lf_data_with_deadlines[lf] = [
            p for p in with_deadlines.data_points if p.latency_factor == lf
        ]
        lf_data_without_deadlines[lf] = [
            p for p in without_deadlines.data_points if p.latency_factor == lf
        ]

    for lf in lf_data_with_deadlines.keys():
        points_with = lf_data_with_deadlines[lf]
        points_without = lf_data_without_deadlines[lf]
        busy_factor_with = [p.busy_factor for p in points_with]
        busy_factor_without = [p.busy_factor for p in points_without]
        assert busy_factor_with == busy_factor_without
        improv_factor = [
            (p_with.succ_prob / p_with.makespan)
            / (p_without.succ_prob / p_without.makespan)
            for p_with, p_without in zip(points_with, points_without)
        ]
        # succ_probs = [p.succ_prob for p in points]
        # error_plus = [p.succ_prob_upper - p.succ_prob for p in points]
        # error_minus = [p.succ_prob - p.succ_prob_lower for p in points]
        # errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.errorbar(
            x=busy_factor_with,
            y=improv_factor,
            # yerr=errors,
            # fmt=FORMATS[version],
            # label=VERSION_LABELS_1[version],
        )

    ax.set_title(
        "Success/s improvement factor of using deadlines)",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    # ax.legend(loc="upper left")

    create_png("theimprovement")


def run_sweep_busy_factor():
    data = load_data("data/no_deadlines/LAST.json")
    sweep_busy_factor(data)


def run_succ_per_s():
    data = load_data("data.json")
    sweep_busy_factor_plot_succ_per_sec(data)


def run_hog_prob():
    data = load_data("data.json")
    sweep_hog_prob(data)


def run_improvement():
    with_deadlines = load_data("results/sweep_hog_duration_with_deadlines/data.json")
    without_deadlines = load_data(
        "results/sweep_hog_duration_without_deadlines/data.json"
    )
    plot_improvement_factor(with_deadlines, without_deadlines)


def run_improvement_per_sec():
    # with_deadlines = load_data("results/succ_prob_per_sec_with_deadlines/data.json")
    # without_deadlines = load_data(
    #     "results/succ_prob_per_sec_without_deadlines/data.json"
    # )
    with_deadlines = load_data("with_deadlines.json")
    without_deadlines = load_data("no_deadlines.json")
    plot_improvement_factor_per_sec(with_deadlines, without_deadlines)


if __name__ == "__main__":
    run_sweep_busy_factor()
    # makespan()
    # run_improvement()
    # run_improvement_per_sec()
    # run_succ_per_s()
