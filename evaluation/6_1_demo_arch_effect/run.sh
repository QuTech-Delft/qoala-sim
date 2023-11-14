#!/bin/bash

num_iterations=10

# QKD
num_pairs=100

# BQC
num_clients=1

python ghz/eval_ghz.py -n $num_iterations
python pingpong/eval_pingpong.py -n $num_iterations
python qkd/eval_qkd.py -n $num_iterations -p $num_pairs
python teleport/eval_teleport.py -n $num_iterations
python vbqc/eval_vbqc.py -n $num_iterations -c $num_clients
