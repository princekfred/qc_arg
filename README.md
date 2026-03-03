qc_arg
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/qc_arg/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/qc_arg/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qc_arg/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/qc_arg/branch/main)


A set of quantum algorithms for near term quantum chemistry excited state calculations

### Copyright

Copyright (c) 2025, Prince Kwao


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
# oak

---

# Expanded README

Quantum chemistry workflows for near-term algorithms, focused on:
- ADAPT-VQE and UCCSD ground-state preparation
- QSC-EOM and QSE excited-state/response-style analysis
- Finite-shot studies (mean/variance vs shot count)
- Comparison against FCI references

This repo includes runnable examples for H2, H4, H6, N2, and NH3.

## Repository Layout

- `qsceom/ground state/`: ADAPT-VQE and MPI QSC-EOM implementations, plus ground-state examples.
- `qsceom/excited state/`: UCCSD and QSC-EOM excited-state modules and examples.
- `qse/`: QSE implementation and shot-study scripts (`run_qseshots.py`, `ongoing_qse.py`).
- `All/`: research notebooks and exploratory runs.

## Environment

`pyproject.toml` contains package metadata, but many scientific runtime dependencies are used directly by scripts. A typical environment needs:
- `numpy`, `scipy`, `matplotlib`
- `qiskit`, `qiskit-aer`, `qiskit-algorithms`, `qiskit-nature`
- `pennylane`, `pennylane-lightning`
- `pyscf`

Example install:

```bash
pip install numpy scipy matplotlib qiskit qiskit-aer qiskit-algorithms qiskit-nature pennylane pennylane-lightning pyscf
```

## Quick Start

Run from repo root:

```bash
python3 "qsceom/ground state/examples/H2/run_h2.py"
python3 "qsceom/ground state/examples/H6/run_h6.py"
python3 "qsceom/ground state/examples/shots/run_nh3_shots.py"
python3 qse/run_qseshots.py
```

## Typical Outputs

- Text summaries in `qsceom/ground state/examples/*` and `qse/`.
- Plots in `qsceom/ground state/examples/*` and `qse/`.

## Example Plot (H2)

![H2 Error Plot](qsceom/ground%20state/examples/H2/h2_error_plot.png)
