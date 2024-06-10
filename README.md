# Bayesian optimization for molecules

A minimal proof-of-concept repo showing BO with the Tanimoto kernel to
perform well on the PMO benchmark.

Main script is `main.py`.
It contains code to run both the "screening" baselines and a basic BO implementation.
Assuming all required packages are installed,
the results can be reproduced by running:

```bash
bash print_all_commands.sh | xargs -I {} bash -c {}
```

Note: this will overwrite the existing results.
The AUC Top-10 results can be tabulated by running:

```bash
python parse_pmo_results.py bo_results/*.json
```

Log files can be found in `tanimoto_bo_logs.tar.gz`.
These logs contain the entire history of the BO trajectory.

## Installation

Running the code requires:

- the `mol_ga` library, which can be installed with pip <https://github.com/AustinT/mol_ga>
- The GP implementation from my _Tanimoto Random Features_ paper, found here: <https://github.com/AustinT/tanimoto-random-features-neurips23>. It is sufficient to clone the repo and add it to the `PYTHONPATH`.
- `pytorch`, `gpytorch`, and `botorch` (for the library above)
- The [Therapeutics Data Commons](https://pypi.org/project/PyTDC/) package.

## Development

Please use pre-commit for code formatting / linting.
