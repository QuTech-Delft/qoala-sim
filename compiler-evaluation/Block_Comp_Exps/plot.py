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
    selfish_bqc_makespan: float
    selfish_local_makespan: float
    cooperative_bqc_makespan: float
    cooperative_local_makespan: float
    selfish_bqc_succ_prob: float
    selfish_local_succ_prob: float
    cooperative_bqc_succ_prob: float
    cooperative_local_succ_prob: float   
    prog_size: float
    num_clients: float
    param_name: str  # Name of param being varied
    param_value: float  # Value of the varied param

@dataclass
class DataMeta:
    timestamp: str
    sim_duration: float
    hardware: str
    qia_sga: float
    scenario: float
    prog_sizes: List[int]
    num_iterations: List[int]
    num_trials: float
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
    internal_sched_latency: float
    client_num_qubits: float
    server_num_qubits: float
    use_netschedule: bool
    bin_length: float
    param_name: str  # The parameter being varied
    link_duration: float
    link_fid: float
    seed: float

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
    selfish_makespan_size_dp_map = dict()
    selfish_succprob_size_dp_map = dict()
    cooperative_makespan_size_dp_map = dict()
    cooperative_succprob_size_dp_map = dict()
    x_val_size_dp_map = dict()

    meta = data.meta
    datapoints = data.data_points

    sizes = meta.prog_sizes
    for size in sizes:
        dps = [dp for dp in datapoints if dp.prog_size == size]
        selfish_makespan_size_dp_map[size] = ([dp.selfish_bqc_makespan for dp in dps] ,[dp.selfish_local_makespan for dp in dps])
        selfish_succprob_size_dp_map[size] = ([dp.selfish_bqc_succ_prob for dp in dps],[dp.selfish_local_succ_prob for dp in dps])

        cooperative_makespan_size_dp_map[size] = ([dp.cooperative_bqc_makespan for dp in dps] , [dp.cooperative_local_makespan for dp in dps])
        cooperative_succprob_size_dp_map[size] = ([dp.cooperative_bqc_succ_prob for dp in dps], [dp.cooperative_local_succ_prob for dp in dps])
        
        x_val_size_dp_map[size] = [dp.param_value for dp in dps]

    return x_val_size_dp_map, selfish_makespan_size_dp_map, selfish_succprob_size_dp_map, cooperative_makespan_size_dp_map, cooperative_succprob_size_dp_map

# Scans all .json files in a folder and finds the 'worst' results in terms of makespan and success probability
def find_worst(path:str, param:str, hardware:str, scenario:str, savefile:bool=False, timestamp=None):
    # Get all .json files for the correct parameter and hardware
    files = [f for f in os.listdir(relative_to_cwd(path)) if f[-5:] == ".json" and param in f and hardware in f and scenario in f]

    # Load all of the data objects
    datas = [load_data(path+"/"+f) for f in files]
    
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
            avg_makespan_diff += dp.selfish_bqc_makespan-dp.cooperative_bqc_makespan
            avg_succprob_diff += dp.cooperative_bqc_succ_prob-dp.selfish_bqc_succ_prob
        avg_makespan_diff = avg_makespan_diff / len(data.data_points) 
        avg_succprob_diff = avg_succprob_diff / len(data.data_points)

        if avg_makespan_diff < worst_makespan:
            worst_makespan = avg_makespan_diff
            worst_makespan_file = files[i]

        if avg_succprob_diff < worst_succprob:
            worst_succprob = avg_succprob_diff
            worst_succprob_file = files[i]
    
    print(worst_makespan, worst_makespan_file)
    print(worst_succprob,worst_succprob_file)
    create_plots(timestamp,load_data(path+"/"+worst_makespan_file),"makespan",saveFile)
    create_plots(timestamp,load_data(path+"/"+worst_succprob_file),"succprob",saveFile)

def load_data(path: str) -> Data:
    with open(relative_to_cwd(path), "r") as f:
        all_data = json.load(f)
    
    return dacite.from_dict(Data, all_data)

def create_plots(timestamp, data: Data, plottype:str, save=True):
    meta = data.meta
    prog_sizes = meta.prog_sizes
    x_val_map, selfish_makespan_map, selfish_succprob_map, cooperative_makespan_map, cooperative_succprob_map = get_vals(data)
    label_fontsize = 14

    if plottype=="makespan" or plottype=="":
        for key in x_val_map.keys():
            for i in range(0,1):
                plt.plot(
                    x_val_map[key], [val for val in selfish_makespan_map[key][i]] , label=f"Self {'bqc' if i == 0 else 'local'}, n={key}", marker="o"
                )
                plt.plot(x_val_map[key], [val for val in cooperative_makespan_map[key][i]], label=f"Coop {'bqc' if i == 0 else 'local'}, n={key}", marker="*")

        plt.legend(loc="upper right", fontsize=11)
        plt.ylabel("Avg Makespan (ns)", fontsize=label_fontsize)
        plt.xlabel(meta.param_name, fontsize=label_fontsize)
        
        if save:
            create_png(timestamp + "_" + meta.param_name + "_makespan_n")
        else:
            plt.show()
        plt.cla()

    if plottype=="succprob" or plottype=="":
        for key in x_val_map.keys():
            for i in range(0,1):
                plt.plot(
                    x_val_map[key], [val for val in selfish_succprob_map[key][i]] , label=f"Self {'bqc' if i == 0 else 'local'}, n={key}", marker="o"
                )
                plt.plot(x_val_map[key], [val for val in cooperative_succprob_map[key][i]], label=f"Coop {'bqc' if i == 0 else 'local'}, n={key}", marker="*")


        plt.legend(loc="lower right", fontsize=11)
        plt.ylabel("Success Probability",fontsize=label_fontsize)
        plt.xlabel(meta.param_name,fontsize=label_fontsize)   
        
        if save:
            create_png(timestamp + "_" + meta.param_name + "_succprob_n")
        else:
            plt.show()
        plt.cla()


    if plottype=="succsec" or plottype=="": 
        for key in x_val_map.keys():
            for j in range(0,2):
                plt.plot(
                    x_val_map[key], [selfish_succprob_map[key][j][i] / selfish_makespan_map[key][j][i] for i in range(0,len(x_val_map[key]))], label=f"Self {'bqc' if j==0 else 'local'} n={key}", marker="o"
                )
                plt.plot(x_val_map[key],  [cooperative_succprob_map[key][j][i] / cooperative_makespan_map[key][j][i] for i in range(0,len(x_val_map[key]))], label=f"Coop {'bqc' if j==0 else 'local'} n={key}", marker="*")

        plt.legend(loc="upper right", fontsize=11)
        plt.ylabel("Successes / ns", fontsize=label_fontsize)
        plt.xlabel(meta.param_name, fontsize=label_fontsize)   
        if save:
            create_png(timestamp + "_" + meta.param_name + "_succsec_n")
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
    parser.add_argument("--scenario", type=str, nargs="+", required=True)
    
    args = parser.parse_args()
    folder = args.folder
    saveFile = args.save
    params = args.params
    hardware = args.hardware
    scenarios = args.scenario
    
    for param in params:
        for hw in hardware:
            for scenario in scenarios:
                find_worst(folder, param, hw, scenario, saveFile, timestamp)

        
