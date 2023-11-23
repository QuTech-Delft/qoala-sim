Evaluation scripts and data used for section 6.2 of the paper.

In each scenario folder (like `bqc_qkd`), there is an `eval_XXX.py` script which starts the simulations for that scenario.
Upon completion, it produces the raw data in the `data/` folder in the scenario folder.
In the `paper/` folder of each scenario folder, the data is provided which was used for the paper.

The `eval_XXX.py` scripts take as input the number of instances (iterations), and other parameters depending on the simulation.
