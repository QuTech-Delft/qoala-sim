#!/bin/bash

num_iterations=1000
decoherence_times="q_mem 85e6 200e6 5e8 1e9 3e9 5e9 7e9"
fidelities="g_fid 0.95 0.97 0.99 0.995 0.999"
gate_durations="g_dur 5 20e3 130e3 200e3"
cc_latency="cc_dur 1e8 1e7 1e6 1e5" #100ms 10ms 1ms 0.1ms

param_list=("$decoherence_times" "$fidelities" "$gate_durations" "$cc_latency")

for params in "${param_list[@]}"
do
    echo "python3 eval_rotation_opt.py -n $num_iterations --sweepList $params"
    python3 eval_rotation_opt.py -n $num_iterations --sweepList $params
done