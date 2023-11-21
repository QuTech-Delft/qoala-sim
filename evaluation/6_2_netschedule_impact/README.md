# Produce data

```
python eval_netschedule_impact.py -d -q {num_qubits} -s {num_steps} -n {num_runs}
```

where
- `-d` indicates that data should be written to a file
- `-q` is the number of qubits
- `-s` is the number of steps for the bin length
- `-n` is the number of runs per "qubit/bin length" pair

To produce the data for the plot, run:

```
python eval_netschedule_impact.py -d -q 1 -s 40 -n 1
python eval_netschedule_impact.py -d -q 2 -s 40 -n 1
python eval_netschedule_impact.py -d -q 5 -s 40 -n 1
```

This produces the folders:
```
data/sweep_bin_length_1_40/ (1 qubit, 40 steps)
data/sweep_bin_length_2_40/ (2 qubit, 40 steps)
data/sweep_bin_length_5_40/ (5 qubit, 40 steps)
```

In each folder, a single simulation run (single run of the python script) produces one `.json` file, with the timestamp of running it as the filename.
Also the `LAST.json` file is always a copy of the most recently created data file.


# Produce plot
Given that the folders `data/sweep_bin_length_{q}_{s}` for certain values of `q` and `s` exist, a plot can be created with

```
python plot_netschedule_impact.py -q {q} -s {s}
```

For the values `q = 1, 2, 5` and `s = 40`, run

```
python plot_netschedule_impact.py -q 1 2 5 -s 40
```

This produces two files:

```
plots/{timestamp}.png
plots/{timetsamp}_meta.json
```

These are the plot in `.png` format and a `_meta.json` file with information about which data files were used to procude the plot. Both files have the timestamp of generating the plot in the filename. The `LAST_meta.json` and `LAST.png` files are copies of the most recent two plot files.