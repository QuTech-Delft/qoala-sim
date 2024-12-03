#!/bin/bash

num_iterations=5
num_clients=5

cc_latency="cc 1e3 3e3 5e3 7e3 1e4 3e4 5e4 7e4 1e5 3e5 5e5 7e5 1e6 3e6 5e6 7e6 1e7" # 1\mus 10\mus 0.1ms 1ms 10ms
decoherence_times="t2 1e3 3e3 5e3 7e3 1e4 3e4 5e4 7e4 1e5 3e5 5e5 7e5 1e6 3e6 5e6 7e6 1e7 3e7 5e7 7e7 1e8"
# single_gate_durations="single_gate_dur 1e0 1e1 1e2 1e3 1e4 1e5 1e6"
# fidelities="single_gate_fid 0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995 0.999"

qnos_instr_time="qnos_instr_proc_time 1e3 3e3 5e3 7e3 1e4 3e4 5e4 7e4 1e5 3e5 5e5 7e5 1e6"
host_instr_time="host_instr_time 1e2 3e2 5e2 7e2 1e3 3e3 5e3 7e3 1e4 3.0001e4 5e4 7e4 1e5"
host_peer_latency="host_peer_latency 1e2 3e2 5e2 7e2 9e2  1e3 3e3 5e3 7e3 9e3 1e4 3e4 5e4 1e5"

netsched_timebin="bin_length 2e3 3e3 5e3 7e3 1e4 3e4 5e4 7e4 1e5 3e5 5e5 7e5 1e6 3e6 5e6 7e6"

# param_list=("$netsched_timebin")
param_list=("$cc_latency" "$decoherence_times" "$single_gate_durations" "$fidelities" "$qnos_instr_time" "$host_instr_time" "$host_peer_latency" "$netsched_timebin")

for num_clients in 1 2 3 4 5
do
    for params in "${param_list[@]}"
    do
        # echo "python3 eval_selfish_cooperative_compilation -c $num_clients -n $num_iterations --param_sweep_list $params --save"
        python3 eval_selfish_cooperative_compilation.py -c $num_clients -n $num_iterations --param_sweep_list $params --save &
    done
done
