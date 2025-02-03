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
    naive_makespan: float
    opt_makespan: float
    naive_succ_prob: float
    opt_succ_prob: float
    prog_size: int
    param_name: str  # Name of param being varied
    param_value: float  # Value of the varied param

@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    hardware: str
    qia_sga: int
    prog_name: str 
    prog_sizes: List[int]
    num_iterations: List[int]
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
    server_num_qubits: float
    # use_netschedule: bool
    # bin_length: int
    param_name: str  # The parameter being varied
    link_duration: int
    link_fid: float

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

# Returns 5 dictionaries that map program size to a list of values
def get_vals(data: Data):
    naive_makespan_size_dp_map = dict()
    naive_succprob_size_dp_map = dict()
    opt_makespan_size_dp_map = dict()
    opt_succprob_size_dp_map = dict()
    x_val_size_dp_map = dict()

    meta = data.meta
    datapoints = data.data_points

    sizes = meta.prog_sizes
    for size in sizes:
        dps = [dp for dp in datapoints if dp.prog_size == size]
        naive_makespan_size_dp_map[size] = [dp.naive_makespan for dp in dps]
        naive_succprob_size_dp_map[size] = [dp.naive_succ_prob for dp in dps]

        opt_makespan_size_dp_map[size] = [dp.opt_makespan for dp in dps]
        opt_succprob_size_dp_map[size] = [dp.opt_succ_prob for dp in dps]
        
        x_val_size_dp_map[size] = [dp.param_value for dp in dps]

    return x_val_size_dp_map, naive_makespan_size_dp_map, naive_succprob_size_dp_map, opt_makespan_size_dp_map, opt_succprob_size_dp_map

# Scans all .json files in a folder and finds the 'worst' results in terms of makespan and success probability
def find_worst(path:str, param:str, hardware:str, program:str):
    # Get all .json files for the correct parameter and hardware
    files = [f for f in os.listdir(relative_to_cwd(path)) if f[-5:] == ".json" and param in f and hardware in f and program in f]

    # Load all of the data objects
    datas = [load_data(path+"/"+f) for f in files]
    
    worst_makespan = 0
    worst_makespan_file = ""
    worst_succprob = 0
    worst_succprob_file = ""
    # For each data object
    for i in range(0, len(datas)):
        data = datas[i]

        avg_makespan_diff = 0
        avg_succprob_diff = 0
        # Compute the average difference for makespan and success probability
        for dp in data.data_points:
            avg_makespan_diff += dp.naive_makespan-dp.opt_makespan
            avg_succprob_diff += dp.opt_succ_prob-dp.naive_succ_prob
        avg_makespan_diff = avg_makespan_diff / len(data.data_points) 
        avg_succprob_diff = avg_succprob_diff / len(data.data_points)

        if avg_makespan_diff > worst_makespan:
            worst_makespan = avg_makespan_diff
            worst_makespan_file = files[i]

        if avg_succprob_diff > worst_succprob:
            worst_succprob = avg_succprob_diff
            worst_succprob_file = files[i]
    
    print(worst_makespan, worst_makespan_file)
    print(worst_succprob,worst_succprob_file)
    create_plots(None,load_data(path+"/"+worst_makespan_file))
    create_plots(None,load_data(path+"/"+worst_succprob_file))

def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    return dacite.from_dict(Data, all_data)

def create_plots(timestamp, data: Data, save=True):
    meta = data.meta
    prog_sizes = meta.prog_sizes
    x_val_map, naive_makespan_map, naive_succprob_map, opt_makespan_map, opt_succprob_map = get_vals(data)
    label_fontsize = 14

    for key in x_val_map.keys():
        plt.plot(
            x_val_map[key], [val / len(x_val_map[key]) for val in naive_makespan_map[key]] , label=f"Subopt n={key}", marker="o"
        )
        plt.plot(x_val_map[key], [val / len(x_val_map[key]) for val in opt_makespan_map[key]], label=f"Opt n={key}", marker="s")

    plt.legend(loc="upper left", fontsize=11)
    plt.ylabel("Avg Makespan (ns)", fontsize=label_fontsize)
    plt.xlabel(meta.param_name, fontsize=label_fontsize)
    plt.show()

    # create_png("LAST_" + x_axis + "_makespan_n" + str(prog_size))
    # create_png(timestamp + "_" + x_axis + "_makespan_n" + str(prog_size))
    plt.cla()


    for key in x_val_map.keys():
        plt.plot(
            x_val_map[key], naive_succprob_map[key], label=f"Subopt n={key}", marker="o"
        )
        plt.plot(x_val_map[key], opt_succprob_map[key], label=f"Opt n={key}", marker="s")

    plt.legend(loc="lower right", fontsize=11)
    plt.ylabel("Success Probability",fontsize=label_fontsize)
    plt.xlabel(meta.param_name,fontsize=label_fontsize)   
    plt.show()


    for key in x_val_map.keys():
        plt.plot(
            x_val_map[key], [naive_succprob_map[key][i] / naive_makespan_map[key][i] for i in range(0,len(x_val_map[key]))], label=f"Subopt n={key}", marker="o"
        )
        plt.plot(x_val_map[key],  [opt_succprob_map[key][i] / opt_makespan_map[key][i] for i in range(0,len(x_val_map[key]))], label=f"Opt n={key}", marker="s")

    plt.legend(loc="upper right", fontsize=11)
    plt.ylabel("Successes / ns", fontsize=label_fontsize)
    plt.xlabel(meta.param_name, fontsize=label_fontsize)   
    plt.show()
    pass

if __name__ == "__main__":
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # parser = ArgumentParser()
    # parser.add_argument("--filenames", "-f", type=str, nargs="+", required=True)
    # parser.add_argument("--save", "-s", action="store_true", default=False)

    # args = parser.parse_args()
    # filenames = args.filenames
    # saveFile = args.save
    # for filename in filenames:
    #     data = load_data(filename)
    #     create_plots(timestamp, data, saveFile)
    find_worst("data","distance", "NV", "vbqc")
