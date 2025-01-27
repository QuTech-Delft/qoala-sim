#!/bin/bash

num_iterations=100
distances="distance 0 2.2 16.8 19.8 26.3 30.6 33.1 40.2 47.9 55.2"
fidelities="single_gate_fid 0.95 0.96 0.97 0.98 0.99 0.995 0.999"
param_list=("$distances" "$fidelities")
configs=("./configs/bqc_eval_NV.json" "./configs/bqc_eval_TI.json" "./configs/rotation_eval_NV.json" "./configs/rotation_eval_TI.json")

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for config in "${configs[@]}"
    do
        for params in "${param_list[@]}"
        do
            echo "python3 eval.py -n $num_iterations -c 1 --param_sweep_list $params --save --config $config --seed $seed"
            python3 eval.py -n $num_iterations -c 1 --param_sweep_list $params --save --config $config --seed $seed &
        done
    done
done