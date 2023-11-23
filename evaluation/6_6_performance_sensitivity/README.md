Evaluation scripts and data used for section 6.6 of the paper.

The `eval_vbqc_sweep.py` script starts the simulations for each of the three sweeps (classical message-passing latency (`cc`), internal latencies (`sched_latency`), and network schedule slot length (`bin_length`)).
Upon completion, it produces the raw data in the `data/` folder in the scenario folder.
In the `paper/` folder the data is provided which was used for the paper.
