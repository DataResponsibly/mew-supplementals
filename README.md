# mew-supplementals

Supplemental materials for the paper of [Most-Expected-Winner](https://doi.org/10.1145/3588702).

## 1. Technical report

Please refer to our [technical report](https://arxiv.org/abs/2105.00082) for the additional proofs, experiments, and the solver for RSM voting profiles.

## 2. How to reproduce experiment results

### 2.1 Hardware information

All experiments were conducted on a machine with
- CPU: 2 x Intel Xeon Platinum 8268 24-core 2.9GHz Processor
- Memory: 384GB RAM

### 2.2 Experiment environment setup

The experiment code was in Python and all experiments were conducted on Linux. 

Below are the steps to prepare for an independent Python environment with required packages:

1. Install [Miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html)
2. `conda env create -f environment.yml`

### 2.3 Re-run all experiments

Execute `run_experiments.sh`

### 2.4 Generate all figures in paper

Each experiment has its own folder in `./experiments/real/` or `./experiments/synthetic/`. The figures in the paper are generated by the `plot.ipynb` in each experiment folder.
