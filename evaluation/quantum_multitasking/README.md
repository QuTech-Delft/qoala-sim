# Produce data

```
python eval_qlassical_multitasking.py -d -t {tel} -l {loc} -n {num_runs}
```

where
- `-d` indicates that data should be written to a file
- `-n` is the number runs per produced data point
- `-t` is the number of teleportation programs
- `-l` is the number of local programs

To produce the data for the plot, run:

```
python eval_quantum_multitasking.py -d -t {tel} -l {loc} -n 200
```

**for each combintation of tel and loc from 1 to 15**.

This produces the folders:
```
data/sweep_teleport_local_{tel}_{loc}/
```
**for each combination of tel and loc**.

In each folder, a single simulation run (single run of the python script) produces one `.json` file, with the timestamp of running it as the filename.
Also the `LAST.json` file is always a copy of the most recently created data file.


## Bash script
The `run.sh` script may be used to speed up the simulation, by running the simulations for the different combinations of tel and loc in parallel.


# Produce plot
Given that the folders `data/qoala`, `data/fcfs` and `data/no_sched` exist, a plot can be created with

```
python plot_classical_multitasking.py
```

This produces two files:

```
plots/{timestamp}.png
plots/{timetsamp}_meta.json
```

These are the plot in `.png` format and a `_meta.json` file with information about which data files were used to procude the plot. Both files have the timestamp of generating the plot in the filename. The `LAST_meta.json` and `LAST.png` files are copies of the most recent two plot files.