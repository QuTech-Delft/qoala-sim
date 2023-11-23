#!/bin/bash

num_steps=40

num_runs=1

echo "python eval_netschedule_impact.py -d -q 1 -s $num_steps -n $num_runs"
python eval_netschedule_impact.py -d -q 1 -s $num_steps -n $num_runs

echo "python eval_netschedule_impact.py -d -q 2 -s $num_steps -n $num_runs"
python eval_netschedule_impact.py -d -q 2 -s $num_steps -n $num_runs

echo "python eval_netschedule_impact.py -d -q 5 -s $num_steps -n $num_runs"
python eval_netschedule_impact.py -d -q 5 -s $num_steps -n $num_runs
