#!/bin/bash

num_runs=100

echo "python eval_vbqc_sweep.py -n $num_runs -s cc"
python eval_vbqc_sweep.py -n $num_runs -s cc

echo "python eval_vbqc_sweep.py -n $num_runs -s sched_latency"
python eval_vbqc_sweep.py -n $num_runs -s sched_latency

echo "python eval_vbqc_sweep.py -n $num_runs -s bin_length"
python eval_vbqc_sweep.py -n $num_runs -s bin_length
