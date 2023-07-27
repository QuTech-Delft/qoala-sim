#!/bin/bash

num_runs=100

# teleport_values=(1 2 3 4 5)
num_teleport=10
num_local=10

commands=()

for tel_val in $(seq 1 $num_teleport); do
    for loc_val in $(seq 1 $num_local); do
        command="python eval_quantum_multitasking.py -d -t $tel_val -l $loc_val -n $num_runs"
        echo "generated command: $command"
        commands+=("$command")
    done
done

for cmd in "${commands[@]}"; do
    $cmd &
done

wait

echo "done"
