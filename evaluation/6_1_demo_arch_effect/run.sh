#!/bin/bash

num_runs=1000

# QKD
num_pairs=100

# BQC
num_clients=1

echo "Running GHZ"
python ghz/eval_ghz.py -n $num_runs

echo "Running Ping-Pong"
python pingpong/eval_pingpong.py -n $num_runs

echo "Running QKD"
python qkd/eval_qkd.py -n $num_runs -p $num_pairs

echo "Running Teleport"
python teleport/eval_teleport.py -n $num_runs

echo "Running VBQC"
python vbqc/eval_vbqc.py -n $num_runs -c $num_clients
