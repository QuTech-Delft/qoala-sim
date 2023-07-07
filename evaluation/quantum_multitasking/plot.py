import json
import os
from argparse import ArgumentParser
from pathlib import Path
from platform import release
from typing import Dict, List

import dacite
import matplotlib.pyplot as plt

from evaluation.classical_multitasking.eval_classical_multitasking import (
    Data,
    DataPoint,
)

COMPILE_VERSIONS = ["meas_epr_first", "meas_epr_last"]
FORMATS = {
    "meas_epr_first": "-rs",
    "meas_epr_last": "-bo",
}
FORMATS_2 = {
    "meas_epr_first": "--gs",
    "meas_epr_last": "--yo",
}

VERSION_LABELS = {
    "meas_epr_first": "Unit modules",
    "meas_epr_last": "No unit modules",
}
VERSION_LABELS_1 = {
    "meas_epr_first": "Unit modules (fidelity)",
    "meas_epr_last": "No unit modules (fidelity)",
}
VERSION_LABELS_2 = {
    "meas_epr_first": "Unit modules (execution time)",
    "meas_epr_last": "No unit modules (execution time)",
}

X_LABELS = {
    "fidelity": "Fidelity",
    "rate": "Success probability per entanglement attempt",
    "t2": "T2 (ns)",
    "gate_noise": "2-qubit gate depolarising probability",
    "gate_time": "2-qubit gate duration (ms)",
    "latency": "Host <-> QNodeOS latency (ms)",
}


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


def sweep_busy_factor_plot_succ_prob_per_sec(data: Data) -> None:
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
        succ_prob_per_sec = [1e9 * p.succ_prob / p.makespan for p in points]
        makespan = [p.makespan for p in points]
        # succ_probs = [p.succ_prob for p in points]
        # error_plus = [p.succ_prob_upper - p.succ_prob for p in points]
        # error_minus = [p.succ_prob - p.succ_prob_lower for p in points]
        error_plus = [
            1e9 * (p.succ_prob_upper - p.succ_prob) / p.makespan for p in points
        ]
        error_minus = [
            1e9 * (p.succ_prob - p.succ_prob_lower) / p.makespan for p in points
        ]
        error_plus = [max(e, 0) for e in error_plus]
        # error_minus = [max(e, 0) for e in error_minus]
        errors = [error_plus, error_minus]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(
            x=busy_factor,
            y=succ_prob_per_sec,
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


def run():
    data = load_data("data.json")
    sweep_busy_factor(data)


def makespan():
    data = load_data("data.json")
    sweep_busy_factor_plot_succ_prob_per_sec(data)


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
    with_deadlines = load_data("results/succ_prob_per_sec_with_deadlines/data.json")
    without_deadlines = load_data(
        "results/succ_prob_per_sec_without_deadlines/data.json"
    )
    plot_improvement_factor_per_sec(with_deadlines, without_deadlines)


if __name__ == "__main__":
    # makespan()
    # run_improvement()
    run_improvement_per_sec()
