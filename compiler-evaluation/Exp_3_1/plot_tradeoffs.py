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
    num_qubits_server: int
    makespan_selfish: float
    makespan_cooperative: float
    succ_prob_selfish: float
    succ_prob_cooperative: float
    param_name: str # Name of param being varied
    param_value: float # Value of the varied param 

@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    client_progs: List[str]
    server_progs: List[str]
    client_prog_args: List[dict]
    server_prog_args: List[dict]
    client_num_iterations: List[int]
    server_num_iterations: List[int]
    num_clients: int
    linear: bool
    cc: float 
    t1: float 
    t2: float 
    single_gate_dur: float 
    two_gate_dur: float 
    all_gate_dur: float
    single_gate_fid: float 
    two_gate_fid: float
    all_gate_fid: float
    qnos_instr_proc_time: float 
    host_instr_time: float 
    host_peer_latency: float 
    client_num_qubits: float 
    use_netschedule: bool
    bin_length: float 
    param_name: str # The parameter being varied

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


def create_plots(timestamp: str, data: List[Data], num_qubits: List[int]):
    """

    :param data:
    :param y_axis: Either succ_prob or makespan
    """

    # Parse data from data object to get lists of x and y values
    data_points = data[0].data_points

    num_clients = data[0].meta.num_clients

    x_vals = [
        data_point.param_value for data_point in data_points if data_point.num_qubits_server == num_qubits[0]
    ]
    
    selfish_makespans = [[[data_point.makespan_selfish for data_point in d.data_points if data_point.num_qubits_server == q] for q in num_qubits] for d in data]
    cooperative_makespans = [[[data_point.makespan_cooperative for data_point in d.data_points if data_point.num_qubits_server == q] for q in num_qubits] for d in data] 
    
    selfish_succ_probs = [[[data_point.succ_prob_selfish for data_point in d.data_points if data_point.num_qubits_server == q] for q in num_qubits] for d in data] 
    cooperative_succ_probs = [[[data_point.succ_prob_cooperative for data_point in d.data_points if data_point.num_qubits_server == q] for q in num_qubits] for d in data] 

    

    # Create labels for axes
    succ_label = "Success probability of quantum program"
    makespan_label = "Makespan of quantum programs (ns)"

    xlabel = data[0].meta.param_name
    num_qubits_str = "_num_qubits"+ "".join([f'_{str(q)}' for q in num_qubits])
    num_clients_str = "_num_clients" + "".join([f'_{d.meta.num_clients}' for d in data])

    # Plot Makespan
    plt.xscale("log")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(makespan_label, fontsize=12)

    # Fix ticks
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.yscale("log")
    # plt.ylim(1e4, 1e9)
    # if x_axis == "cc_dur":
    #     plt.ylim(0,9e8)
    # elif x_axis == "instr_time":
    #     plt.ylim(0, 3e7)
    # else:
    #     plt.ylim(0,8e6)
    for d in range(0, len(data)):
        for i in range(0,len(num_qubits)):
            q=num_qubits[i]
            plt.plot(x_vals, cooperative_makespans[d][i], label=f"{data[d].meta.num_clients} clients, Coop, {q} server qbts", marker="s")
            plt.plot(
                x_vals, selfish_makespans[d][i], label=f"{data[d].meta.num_clients} clients, Self, {q} server qbts", marker="o"
            )

    plt.legend(loc="upper left", fontsize=11)

    create_png("LAST_" + xlabel + num_clients_str + num_qubits_str + "_makespan")
    create_png(timestamp + "_" + xlabel + num_clients_str + num_qubits_str +"_makespan")
    plt.cla()

    # Plot Success Probability
    plt.xscale("log")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(succ_label, fontsize=12)

    plt.ylim(0, 1.1)

    for d in range(0,len(data)):
        for i in range(0,len(num_qubits)):
            q=num_qubits[i]
            plt.plot(x_vals, cooperative_succ_probs[d][i], label=f"{data[d].meta.num_clients} clients, Coop, {q} server qbts", marker="s")
            plt.plot(
                x_vals, selfish_succ_probs[d][i], label=f"{data[d].meta.num_clients} clients, Self, {q} server qbts", marker="o"
            )

    plt.legend(loc="lower right", fontsize=11)

    create_png("LAST_" + xlabel + num_clients_str + num_qubits_str + "_succ_prob")
    create_png(timestamp + "_" + xlabel + num_clients_str + num_qubits_str +"_succ_prob")
    plt.cla()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = ArgumentParser()
    parser.add_argument("--filename", "-f", type=str, required=True)
    parser.add_argument("--client_nums", "-c" , nargs="+", type=int, required=True)
    parser.add_argument("--num_qubits", "-q" , nargs="+", type=int, required=True)

    args = parser.parse_args()
    filename = args.filename
    client_nums = args.client_nums
    num_qubits = args.num_qubits
    # "host_peer_latency", "host_instr_time" ,"cc",
    param_strings = ["bin_length",   "host_instr_time", "host_peer_latency", "qnos_instr_proc_time", "t2"] # "single_gate_dur", "single_gate_fid", 
    filenames = [f"{filename}_{param_name}" for param_name in param_strings]

    for file in filenames:
        data = [load_data(f"{file}_nclients_{i}.json") for i in client_nums]
        create_plots(timestamp, data, num_qubits)
