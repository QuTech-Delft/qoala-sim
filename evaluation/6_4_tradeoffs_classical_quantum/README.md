# Produce data

```
python eval_tradeoffs_cq.py -d -n {num_iterations}
```

where
- `-d` indicates that data should be written to a file
- `-n` is the number of iterations per produces data point

To produce the data for the plot, run:

```
python eval_tradeoffs_cq.py -d -n 100 -s qoala
python eval_tradeoffs_cq.py -d -n 100 -s fcfs
python eval_tradeoffs_cq.py -d -n 100 -s no_sched
```

This produces the folders:
```
data/qoala/ (qoala scheduler)
data/fcfs/ (fcfs scheduler)
data/no_sched/ (no scheduler)
```

In each folder, a single simulation run (single run of the python script) produces one `.json` file, with the timestamp of running it as the filename.
Also the `LAST.json` file is always a copy of the most recently created data file.


## Bash script
The `run.sh` script may be used to speed up the simulation, by running the simulations for the 3 scheduler types in parallel.


# Produce plot
Given that the folders `data/qoala`, `data/fcfs` and `data/no_sched` exist, a plot can be created with

```
python plot_tradeoffs_cq.py
```

This produces two files:

```
plots/{timestamp}.png
plots/{timetsamp}_meta.json
```

These are the plot in `.png` format and a `_meta.json` file with information about which data files were used to procude the plot. Both files have the timestamp of generating the plot in the filename. The `LAST_meta.json` and `LAST.png` files are copies of the most recent two plot files.