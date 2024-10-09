#!/bin/bash

num_runs=1000

# QKD
num_pairs=1000

# BQC
num_clients=1

echo "python ghz/eval_ghz.py -n $num_runs"
python ghz/eval_ghz.py -n $num_runs

echo "python pingpong/eval_pingpong.py -n $num_runs"
python pingpong/eval_pingpong.py -n $num_runs

echo "python qkd/eval_qkd.py -n $num_runs -p $num_pairs"
python qkd/eval_qkd.py -n $num_runs -p $num_pairs

echo "python teleport/eval_teleport.py -n $num_runs"
python teleport/eval_teleport.py -n $num_runs

echo "python vbqc/eval_vbqc.py -n $num_runs -c $num_clients"
python vbqc/eval_vbqc.py -n $num_runs -c $num_clients
