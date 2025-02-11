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
import math

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

    def filter_data_points(self, value):
        # Remove data points with param_value > value 
        self.data_points = [dp for dp in self.data_points if dp.param_value <= value]

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
def find_worst(path:str, param:str, hardware:str, program:str, savefile:bool=False, timestamp=None):
    # Get all .json files for the correct parameter and hardware
    files = [f for f in os.listdir(relative_to_cwd(path)) if f[-5:] == ".json" and param in f and hardware in f and program in f]

    # Load all of the data objects
    datas = [load_data(path+"/"+f) for f in files] 

    # Extra filtering necessary
    if param == "cc":
        _data = [data.filter_data_points(1e7) for data in datas]


    worst_makespan = math.inf
    worst_makespan_file = ""
    worst_succprob = math.inf 
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

        if avg_makespan_diff < worst_makespan:
            worst_makespan = avg_makespan_diff
            worst_makespan_file = files[i]

        if avg_succprob_diff < worst_succprob:
            worst_succprob = avg_succprob_diff
            worst_succprob_file = files[i]
    
    print("Avg Makespan diff: ", worst_makespan, worst_makespan_file)
    print("Avg Succprob diff: ", worst_succprob,worst_succprob_file)
    worst_makespan_data = load_data(path+"/"+worst_makespan_file)
    worst_succprob_data = load_data(path+"/"+worst_makespan_file)
    # Extra filtering necessary
    if param == "cc":
        worst_makespan_data.filter_data_points(1e7)
        worst_succprob_data.filter_data_points(1e7)

    create_plots(timestamp,worst_makespan_data,"makespan",saveFile)
    create_plots(timestamp,worst_succprob_data,"succprob",saveFile)
    create_plots(timestamp,worst_succprob_data,"succsec",saveFile)

def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)

    return dacite.from_dict(Data, all_data)

def create_plots(timestamp, data: Data, plottype:str, save=True):
    meta = data.meta
    prog_sizes = meta.prog_sizes
    x_val_map, naive_makespan_map, naive_succprob_map, opt_makespan_map, opt_succprob_map = get_vals(data)
    label_fontsize = 14
    opt_markersize=10

    plt.xscale('log')
    if plottype=="makespan" or plottype=="":
        for key in x_val_map.keys():
            plt.plot(
                x_val_map[key], [(val / len(x_val_map[key])) / 1e9 for val in naive_makespan_map[key]] , label=f"Subopt $n$={key}", marker="o"
            )
            if meta.prog_name == "rotation":
                if key == 10:
                    plt.plot(x_val_map[key], [(val / len(x_val_map[key])) / 1e9 for val in opt_makespan_map[key]], label=f"Opt $n$={key}", marker="*", markersize=opt_markersize, color="red")
            else:
                plt.plot(x_val_map[key], [(val / len(x_val_map[key])) / 1e9 for val in opt_makespan_map[key]], label=f"Opt $n$={key}", marker="*", markersize=opt_markersize)

        plt.legend(loc="upper left", fontsize=11)
        if meta.param_name == "single_gate_fid":
            plt.xlabel("Single qubit gate fidelity",fontsize=label_fontsize)   
        elif meta.param_name == "distance":
            plt.xlabel("Distance (km)",fontsize=label_fontsize)   
        else:
            plt.xlabel(meta.param_name,fontsize=label_fontsize)   
        plt.ylabel("Avg Makespan (s)", fontsize=label_fontsize)
        
        if save:
            create_png(timestamp + "_" + meta.prog_name + "_"+ meta.param_name + "_makespan_n_"+ meta.hardware)
        else:
            plt.show()
        plt.cla()

    if plottype=="succprob" or plottype=="":
        plt.ylim(0.55, 1.01)
        for key in x_val_map.keys():
            plt.plot(
                x_val_map[key], naive_succprob_map[key], label=f"Subopt $n$={key}", marker="o"
            )
            if meta.prog_name == "rotation":
                if key == 10:
                    plt.plot(x_val_map[key], opt_succprob_map[key], label=f"Opt $n$={key}", marker="*", markersize=opt_markersize, color="red")
            else:
                plt.plot(x_val_map[key], opt_succprob_map[key], label=f"Opt $n$={key}", marker="*", markersize=opt_markersize)

        plt.legend(loc="lower right", fontsize=11)
        plt.ylabel("Success Probability",fontsize=label_fontsize)
        if meta.param_name == "single_gate_fid":
            plt.xlabel("Single qubit gate fidelity",fontsize=label_fontsize)   
        elif meta.param_name == "distance":
            plt.xlabel("Distance (km)",fontsize=label_fontsize)   
        else:
            plt.xlabel(meta.param_name,fontsize=label_fontsize)   
        
        if save:
            create_png(timestamp + "_" + meta.prog_name + "_"+ meta.param_name + "_succprob_n_" + meta.hardware)
        else:
            plt.show()
        plt.cla()


    if plottype=="succsec" or plottype=="": 
        for key in x_val_map.keys():
            plt.plot(
                x_val_map[key], [naive_succprob_map[key][i] / naive_makespan_map[key][i] * 1e9 for i in range(0,len(x_val_map[key]))], label=f"Subopt $n$={key}", marker="o"
            )
            plt.plot(x_val_map[key],  [opt_succprob_map[key][i] / opt_makespan_map[key][i] *1e9 for i in range(0,len(x_val_map[key]))], label=f"Opt $n$={key}", marker="*", markersize=opt_markersize)

        plt.legend(loc="upper right", fontsize=11)
        plt.ylabel("Successes / s", fontsize=label_fontsize)
        if meta.param_name == "single_gate_fid":
            plt.xlabel("Single qubit gate fidelity",fontsize=label_fontsize)   
        elif meta.param_name == "distance":
            plt.xlabel("Distance (km)",fontsize=label_fontsize)   
        else:
            plt.xlabel(meta.param_name,fontsize=label_fontsize)

        if save:
            create_png(timestamp + "_" + meta.prog_name + "_" + meta.param_name + "_succsec_n_"+ meta.hardware)
        else:
            plt.show()
        plt.cla()

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, required=True)
    parser.add_argument("--save", "-s", action="store_true", default=False)
    parser.add_argument("--params", type=str, nargs="+", required=True)
    parser.add_argument("--hardware", type=str, nargs="+", required=True)
    parser.add_argument("--programs", type=str, nargs="+",required=True)

    args = parser.parse_args()
    folder = args.folder
    saveFile = args.save
    params = args.params
    hardware = args.hardware
    programs = args.programs
    
    for param in params:
        for hw in hardware:
            for program in programs:
                find_worst(folder, param, hw, program, saveFile, timestamp)

        
