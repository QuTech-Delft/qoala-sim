#!/bin/bash

num_runs=10

# QKD
num_pairs=100

# BQC
num_clients=1

python ghz/eval_ghz.py -n $num_runs
python pingpong/eval_pingpong.py -n $num_runs
python qkd/eval_qkd.py -n $num_runs -p $num_pairs
python teleport/eval_teleport.py -n $num_runs
python vbqc/eval_vbqc.py -n $num_runs -c $num_clients
