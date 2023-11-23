#!/bin/bash

# Multi-Teleport
MT_num_iterations=10

# Multi-BQC
MB_num_iterations=10
MB_num_clients=10

# BQC-QKD
BQ_bqc_num_iterations=50
BQ_qkd_num_iterations=50

echo "python multi_teleport/eval_multi_teleport.py -n $MT_num_iterations"
python multi_teleport/eval_multi_teleport.py -n $MT_num_iterations

echo "python multi_bqc/eval_multi_bqc.py -n $MB_num_iterations -c $MB_num_clients"
python multi_bqc/eval_multi_bqc.py -n $MB_num_iterations -c $MB_num_clients

echo "python bqc_qkd/eval_bqc_qkd.py -b $BQ_bqc_num_iterations -q $BQ_qkd_num_iterations"
python bqc_qkd/eval_bqc_qkd.py -b $BQ_bqc_num_iterations -q $BQ_qkd_num_iterations
