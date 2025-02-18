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
from collections import defaultdict

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

    filter_out = []

    meta = data.meta
    datapoints = data.data_points

    sizes = meta.prog_sizes
    for size in sizes:
        dps = [dp for dp in datapoints if dp.prog_size == size]
        selfish_makespan_size_dp_map[size] = ([dp.selfish_bqc_makespan for dp in dps if dp.param_value  not in filter_out] ,[dp.selfish_local_makespan for dp in dps if dp.param_value  not in filter_out])
        selfish_succprob_size_dp_map[size] = ([dp.selfish_bqc_succ_prob for dp in dps if dp.param_value  not in filter_out],[dp.selfish_local_succ_prob for dp in dps if dp.param_value  not in filter_out])

        cooperative_makespan_size_dp_map[size] = ([dp.cooperative_bqc_makespan for dp in dps if dp.param_value not in filter_out] , [dp.cooperative_local_makespan for dp in dps if dp.param_value  not in filter_out])
        cooperative_succprob_size_dp_map[size] = ([dp.cooperative_bqc_succ_prob for dp in dps if dp.param_value not in filter_out], [dp.cooperative_local_succ_prob for dp in dps if dp.param_value  not in filter_out])
        
        x_val_size_dp_map[size] = [dp.param_value for dp in dps if dp.param_value not in filter_out]

    return x_val_size_dp_map, selfish_makespan_size_dp_map, selfish_succprob_size_dp_map, cooperative_makespan_size_dp_map, cooperative_succprob_size_dp_map

# Scans all .json files in a folder and finds the 'worst' results in terms of makespan and success probability
def find_worst(path:str, param:str, hardware:str, scenario:str, savefile:bool=False, timestamp=None):
    # Get all .json files for the correct parameter and hardware
    files = [f for f in os.listdir(relative_to_cwd(path)) if f[-5:] == ".json" and param in f and hardware in f and scenario in f]

    # Load all of the data objects
    datas = [load_data(path+"/"+f) for f in files]
    # for f in files:
        # create_plots(timestamp, load_data(path+"/"+f), "makespan", False)
        # create_plots(timestamp, load_data(path+"/"+f), "succprob", False)
    #     create_plots(timestamp, load_data(path+"/"+f), "succsec", False)


    worst_makespan = math.inf 
    worst_makespan_file = ""
    worst_succprob = math.inf 
    worst_succprob_file = ""
    local_makespan_diff = 0
    bqc_makespan_diff = 0
    bqc_succsec_diff =0
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
            local_makespan_diff = [[sum([dp.cooperative_local_makespan for dp in data.data_points if dp.prog_size == n and dp.param_value !=1])/9,sum(dp.selfish_local_makespan for dp in data.data_points if dp.prog_size==n and dp.param_value!=1)/9] for n in [3,5]]
            bqc_makespan_diff = [[sum([dp.cooperative_bqc_makespan for dp in data.data_points if (dp.prog_size == n and dp.param_value != 1)])/9,sum(dp.selfish_bqc_makespan for dp in data.data_points if (dp.prog_size==n and dp.param_value != 1))/9] for n in [3,5]]

        if avg_succprob_diff < worst_succprob:
            worst_succprob = avg_succprob_diff
            worst_succprob_file = files[i]

            bqc_succsec_diff = [[sum([dp.cooperative_bqc_succ_prob / dp.cooperative_bqc_makespan for dp in data.data_points if (dp.prog_size == n and dp.param_value != 1)])/9,sum(dp.selfish_bqc_succ_prob / dp.selfish_bqc_makespan for dp in data.data_points if (dp.prog_size==n and dp.param_value != 1))/9] for n in [3,5]]
    
    print(worst_makespan, worst_makespan_file)
    print(worst_succprob,worst_succprob_file)
    print(local_makespan_diff)
    print([(ld[0] - ld[1])/ld[1] for ld in local_makespan_diff])
    # print(bqc_makespan_diff)
    # print([(ld[1] - ld[0])/ld[1] for ld in bqc_makespan_diff])
    print(bqc_succsec_diff)
    print([(ld[0]-ld[1])/ld[0] for ld in bqc_succsec_diff])
    create_plots(timestamp,load_data(path+"/"+worst_makespan_file),"makespan",saveFile)
    create_plots(timestamp,load_data(path+"/"+worst_succprob_file),"succprob",saveFile)
    create_plots(timestamp,load_data(path+"/"+worst_succprob_file),"succsec",saveFile)

def plot_avg(path:str, param:str, hardware:str, scenario:str, savefile:bool=False, timestamp=None):
    # Get all .json files for the correct parameter and hardware
    files = [f for f in os.listdir(relative_to_cwd(path)) if f[-5:] == ".json" and param in f and hardware in f and scenario in f]

    # Load all of the data objects
    datas = [load_data(path+"/"+f) for f in files]
    avg_data_points = average_data_points(datas)
    datas[0].data_points=avg_data_points
    create_plots(timestamp,datas[0],"makespan",saveFile)
    create_plots(timestamp,datas[0],"succprob",saveFile)
    create_plots(timestamp,datas[0],"succsec",saveFile)

def average_data_points(data_list: List[Data]) -> List[DataPoint]:
    # This dictionary will store the combined data points, keyed by (param_value, prog_size, num_clients)
    combined_data = defaultdict(lambda: {
        'selfish_bqc_makespan': 0.0,
        'selfish_local_makespan': 0.0,
        'cooperative_bqc_makespan': 0.0,
        'cooperative_local_makespan': 0.0,
        'selfish_bqc_succ_prob': 0.0,
        'selfish_local_succ_prob': 0.0,
        'cooperative_bqc_succ_prob': 0.0,
        'cooperative_local_succ_prob': 0.0,
        'count': 0
    })
    
    # Iterate through all data points and aggregate
    for data in data_list:
        for data_point in data.data_points:
            key = (data_point.param_value, data_point.prog_size, data_point.num_clients)
            
            # Aggregate the values for each matching key
            combined_data[key]['selfish_bqc_makespan'] += data_point.selfish_bqc_makespan
            combined_data[key]['selfish_local_makespan'] += data_point.selfish_local_makespan
            combined_data[key]['cooperative_bqc_makespan'] += data_point.cooperative_bqc_makespan
            combined_data[key]['cooperative_local_makespan'] += data_point.cooperative_local_makespan
            combined_data[key]['selfish_bqc_succ_prob'] += data_point.selfish_bqc_succ_prob
            combined_data[key]['selfish_local_succ_prob'] += data_point.selfish_local_succ_prob
            combined_data[key]['cooperative_bqc_succ_prob'] += data_point.cooperative_bqc_succ_prob
            combined_data[key]['cooperative_local_succ_prob'] += data_point.cooperative_local_succ_prob
            combined_data[key]['count'] += 1
    
    # Now, calculate the averages and return a list of DataPoint objects
    averaged_data_points = []
    
    for key, values in combined_data.items():
        # Avoid division by zero
        count = values['count']
        if count > 0:
            averaged_data_point = DataPoint(
                selfish_bqc_makespan=values['selfish_bqc_makespan'] / count,
                selfish_local_makespan=values['selfish_local_makespan'] / count,
                cooperative_bqc_makespan=values['cooperative_bqc_makespan'] / count,
                cooperative_local_makespan=values['cooperative_local_makespan'] / count,
                selfish_bqc_succ_prob=values['selfish_bqc_succ_prob'] / count,
                selfish_local_succ_prob=values['selfish_local_succ_prob'] / count,
                cooperative_bqc_succ_prob=values['cooperative_bqc_succ_prob'] / count,
                cooperative_local_succ_prob=values['cooperative_local_succ_prob'] / count,
                prog_size=key[1],  # prog_size from key
                num_clients=key[2],  # num_clients from key
                param_name=data_list[0].meta.param_name,  # assuming same param_name for all Data objects
                param_value=key[0]  # param_value from key
            )
            averaged_data_points.append(averaged_data_point)
    
    return averaged_data_points

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
        plt.ylim(0.4, 1.01)
        for key in x_val_map.keys():
            for i in range(0,1):
                plt.plot(
                    x_val_map[key], [val for val in selfish_succprob_map[key][i]] , label=f"Self {'bqc' if i == 0 else 'local'}, n={key}", marker="o"
                )
                plt.plot(x_val_map[key], [val for val in cooperative_succprob_map[key][i]], label=f"Coop {'bqc' if i == 0 else 'local'}, n={key}", marker="*")


        plt.legend(loc="upper right", fontsize=11)
        plt.ylabel("Success Probability",fontsize=label_fontsize)
        plt.xlabel(meta.param_name,fontsize=label_fontsize)   
        
        if save:
            create_png(timestamp + "_" + meta.param_name + "_succprob_n")
        else:
            plt.show()
        plt.cla()


    if plottype=="succsec" or plottype=="": 
        for key in x_val_map.keys():
            for j in range(0,1):
                plt.plot(
                    x_val_map[key], [selfish_succprob_map[key][j][i] / selfish_makespan_map[key][j][i] * 1e9 for i in range(0,len(x_val_map[key]))], label=f"Self {'bqc' if j==0 else 'local'} n={key}", marker="o"
                )
                plt.plot(x_val_map[key],  [cooperative_succprob_map[key][j][i] / cooperative_makespan_map[key][j][i] * 1e9 for i in range(0,len(x_val_map[key]))], label=f"Coop {'bqc' if j==0 else 'local'} n={key}", marker="*")

        plt.legend(loc="upper right", fontsize=11)
        plt.ylabel("Successes / s", fontsize=label_fontsize)
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
                # find_worst(folder, param, hw, scenario, saveFile, timestamp)
                plot_avg(folder, param, hw, scenario, saveFile, timestamp)

        
