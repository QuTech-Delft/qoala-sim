import json
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
    latency_factor: float
    net_bin_factors: List[float]


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


def create_meta(filename: str, data: Data):
    output_dir = relative_to_cwd("plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.json")
    meta = {}
    meta["datafile"] = data.meta.timestamp
    with open(output_path, "w") as metafile:
        json.dump(meta, metafile)


def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    # assert isinstance(all_data, list)
    # return [dacite.from_dict(DataPoint, entry) for entry in all_data]
    return dacite.from_dict(Data, all_data)


def plot_sweep_network_period(timestamp: str, data: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Network bin factor")
    ax.set_ylabel("Makespan")

    npf = [npf for npf in data.meta.net_period_factors]
    succ_probs = [p.succ_prob for p in data.data_points]
    error_plus = [p.succ_prob_upper - p.succ_prob for p in data.data_points]
    error_plus = [max(0, e) for e in error_plus]
    error_minus = [p.succ_prob - p.succ_prob_lower for p in data.data_points]
    error_minus = [max(0, e) for e in error_minus]
    errors = [error_minus, error_plus]
    print(errors)
    ax.set_xscale("log")
    ax.errorbar(
        x=npf,
        y=succ_probs,
        yerr=errors,
    )

    ax.set_title(
        "Success probability vs Network period factor",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    ax.legend(loc="upper left")

    create_png("LAST")
    create_png(timestamp)


def plot_sweep_net_bin_period(timestamp: str, data: Data) -> None:
    fig, ax = plt.subplots()

    ax.grid()
    ax.set_xlabel("Network period factor")
    ax.set_ylabel("Makespan")

    ax2 = ax.twinx()

    nbf = [npf for npf in data.meta.net_bin_factors]
    makespans = [p.makespan for p in data.data_points]
    succ_probs = [p.tel_succ_prob for p in data.data_points]
    error_plus = [p.tel_succ_prob_upper - p.tel_succ_prob for p in data.data_points]
    error_plus = [max(0, e) for e in error_plus]
    error_minus = [p.tel_succ_prob - p.tel_succ_prob_lower for p in data.data_points]
    error_minus = [max(0, e) for e in error_minus]
    errors = [error_minus, error_plus]
    # ax.set_xscale("log")
    ax.errorbar(
        x=nbf,
        y=makespans,
        fmt="o-r",
    )
    # ax2.errorbar(
    #     x=nbf,
    #     y=succ_probs,
    #     yerr=errors,
    #     fmt="o-b",
    # )

    ax.set_title(
        "Makespan vs Network bin factor",
        wrap=True,
    )

    # ax.set_ylim(0.75, 0.9)
    ax.legend(loc="upper left")

    create_png("LAST")
    create_png(timestamp)


def sweep_network_period():
    data = load_data("data/net_period/LAST.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", data)
    create_meta(f"{timestamp}_meta", data)
    plot_sweep_network_period(timestamp, data)


def sweep_net_bin_factor():
    data = load_data("data/net_bin_factor/LAST.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", data)
    create_meta(f"{timestamp}_meta", data)
    plot_sweep_net_bin_period(timestamp, data)


def sweep_net_bin_factor_prio_epr():
    data = load_data("data/net_bin_factor_prio_epr/LAST.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_meta("LAST_meta", data)
    create_meta(f"{timestamp}_meta", data)
    plot_sweep_net_bin_period(timestamp, data)


if __name__ == "__main__":
    # sweep_network_period()
    # sweep_net_bin_factor()
    sweep_net_bin_factor_prio_epr()
