#!/bin/bash

num_iterations=100
decoherence_times="q_mem 85e6 5e8 1e9"
fidelities="g_fid 0.95 0.96 0.97 0.98 0.99 0.995 0.999"
gate_durations="g_dur 5 20e3 25e3 100e3"
qnos_instr_time="instr_time 4e3 50e3 100e3"
cc_latency="cc_dur 1e6 5e6 10e6 20e6 40e6" 
param_list=("$decoherence_times" "$fidelities" "$gate_durations" "$cc_latency" "$qnos_instr_time")

for params in "${param_list[@]}"
do
    echo "python3 eval_rotation_opt.py -n $num_iterations --sweepList $params"
    python3 eval_rotation_opt.py -n $num_iterations --sweepList $params &
done