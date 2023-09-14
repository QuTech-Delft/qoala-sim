#!/bin/bash

START_TIME=$SECONDS

num_runs=1000


file1=/tmp/status1.txt
file2=/tmp/status2.txt
file3=/tmp/status3.txt

python eval_classical_multitasking_2.py -d -n $num_runs -s qoala > $file1 &
PID1=$!
python eval_classical_multitasking_2.py -d -n $num_runs -s fcfs > $file2 &
PID2=$!
python eval_classical_multitasking_2.py -d -n $num_runs -s no_sched > $file3 &
PID3=$!

while :; do
    clear
    cat $file1; echo ""
    cat $file2; echo ""
    cat $file3; echo ""

    if ! kill -0 $PID1 2>/dev/null; then
        if ! kill -0 $PID2 2>/dev/null; then
            if ! kill -0 $PID3 2>/dev/null; then
                break
            fi
        fi
    fi

    sleep 0.1  # Update every second, adjust as needed
done

rm $file1 $file2 $file3

END_TIME=$SECONDS

ELAPSED_TIME=$((END_TIME - START_TIME))

echo "done in $ELAPSED_TIME seconds"
