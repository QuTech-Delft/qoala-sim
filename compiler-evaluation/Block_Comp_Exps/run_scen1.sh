#!/bin/bash

num_trials=100
distances="distance 0 2.2 16.8 19.8 26.3 30.6 33.1 40.2 47.9 55.2"
fidelities="single_gate_fid 0.95 0.96 0.97 0.98 0.99 0.995 0.999"
bin_length="bin_length 1 2 3 4 5 6 7 8 9 10"
param_list=("$bin_length")
# "./configs/rotation_eval_NV.json" "./configs/rotation_eval_TI.json"
configs=("./configs/scen1/NV_1.json" "./configs/scen1/TI_2.json")

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for config in "${configs[@]}"
    do
        for params in "${param_list[@]}"
        do
            echo "python3 eval.py --num_trials $num_trials --param_sweep_list $params --save --config $config --seed $seed"
            python3 eval.py --num_trials $num_trials --param_sweep_list $params --save --config $config --seed $seed &
        done
    done
done