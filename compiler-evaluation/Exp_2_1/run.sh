#!/bin/bash

num_iterations=1000
decoherence_times="q_mem 1e3 3e3 5e3 7e3 9e3 1e4 3e4 5e4 7e4 9e4 1e5 3e5 5e5 7e5 9e5 1e6 3e6 5e6 7e6 9e6 1e7 3e7 5e7 7e7 9e7 1e8"
fidelities="g_fid 0.95 0.955 0.96 0.965 0.97 0.975 0.98  0.985 0.99 0.995 0.999"
gate_durations="g_dur 1 10 1e2 3e2 5e2 7e2 9e2 1e3 3e3 5e3 7e3 9e3 1e4 3e4 5e4 7e4 9e4 1e5 3e5 5e5 7e5 9e5 1e6"
qnos_instr_time="instr_time 1e3 3e3 5e3 7e3 9e3 1e4 3e4 5e4 7e4 9e4 1e5 3e5 5e5 7e5 9e5 1e6"
cc_latency="cc_dur 1e3 3e3 5e3 7e3 9e3 1e4 3e4 5e4 7e4 9e4 1e5 3e5 5e5 7e5 9e5 1e6 3e6 5e6 7e6 9e6 1e7" # 1\mus 10\mus 0.1ms 1ms 10ms

param_list=("$decoherence_times" "$fidelities" "$gate_durations" "$cc_latency" "$qnos_instr_time")

for params in "${param_list[@]}"
do
    echo "python3 eval_rotation_opt.py -n $num_iterations --sweepList $params"
    python3 eval_rotation_opt.py -n $num_iterations --sweepList $params &
done