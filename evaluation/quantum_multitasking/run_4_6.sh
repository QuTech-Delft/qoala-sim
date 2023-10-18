#!/bin/bash

START_TIME=$SECONDS

num_runs=200
# num_runs=1

# teleport_values=(1 2 3 4 5)
num_teleport=15

commands=()

for tel_val in $(seq 1 $num_teleport); do
    for loc_val in $(seq 4 6); do
        command="python eval_quantum_multitasking.py -d -t $tel_val -l $loc_val -n $num_runs"
        echo "generated command: $command"
        commands+=("$command")
    done
done

for cmd in "${commands[@]}"; do
    $cmd &
done

wait

END_TIME=$SECONDS

ELAPSED_TIME=$((END_TIME - START_TIME))

echo "done in $ELAPSED_TIME seconds"
