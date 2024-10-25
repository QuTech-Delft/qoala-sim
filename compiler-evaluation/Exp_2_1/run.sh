#!/bin/bash

num_iterations=1000
decoherence_times="q_mem 1e3 1e4 1e5 1e6 1e7 1e8"
fidelities="g_fid 0.95 0.97 0.99 0.995 0.999"
gate_durations="g_dur 1 10 1e2 1e3 1e4 1e5 1e6"
qnos_instr_time="instr_time 1e3 5e3 10e3 50e3 100e3 500e3 1e6"
cc_latency="cc_dur 1e3 1e4 1e5 1e6 1e7" # 1\mus 10\mus 0.1ms 1ms 10ms

param_list=("$decoherence_times" "$fidelities" "$gate_durations" "$cc_latency" "$qnos_instr_time")

for params in "${param_list[@]}"
do
    echo "python3 eval_rotation_opt.py -n $num_iterations --sweepList $params"
    python3 eval_rotation_opt.py -n $num_iterations --sweepList $params &
done