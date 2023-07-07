# mew-supplementals

This repository contains supplemental materials for the paper of [Most-Expected-Winner](https://doi.org/10.1145/3588702).

Please refer to our [technical report](https://arxiv.org/abs/2105.00082) for the additional proofs, experiments, and the solver for RSM voting profiles.

## Steps to reproduce all experiment results

### 1. Hardware setup

We ran all experiments on a Linux machine with the following specs:
- CPU: 2 x Intel Xeon Platinum 8268 24-core 2.9GHz Processor
- Memory: 384GB RAM


### 2. Experiment environment setup

The experiment code was in Python and all experiments were conducted on Linux. 

Below are the steps to prepare for an independent Python environment with required packages:

1. Install [Miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html)
2. `conda env create -f environment.yml`

### 3. Run all experiments

Execute `run_experiments.sh`

### 4. Generate all figures in paper

Each experiment has its own folder in `./experiments/real/` or `./experiments/synthetic/`. 
- Fig. 3(a): `experiments/synthetic/posets_for_candidate_pruning/plot.ipynb`
- Fig. 3(b): `experiments/synthetic/posets_for_voter_grouping/plot.ipynb`
- Fig. 4: `experiments/synthetic/posets_for_parallelization/plot.ipynb`
- Fig. 5: `experiments/synthetic/posets/plot.ipynb`
- Fig. 6: `experiments/synthetic/posets_for_cover_width/plot.ipynb`
- Fig. 7: `experiments/synthetic/ppwm/plot.ipynb`
- Fig. 8: `experiments/synthetic/pp/plot.ipynb`
- Fig. 9: `experiments/synthetic/pc/plot.ipynb`
- Fig. 10: `experiments/synthetic/tr/plot.ipynb`
- Fig. 11: `experiments/synthetic/mallows/plot.ipynb`
- Fig. 12: `experiments/synthetic/rsm/plot.ipynb`
- Fig. 13(a): `experiments/synthetic/mallows_pp/plot.ipynb`
- Fig. 13(b): `experiments/synthetic/mallows_tr/plot.ipynb`
- Fig. 13(c): `experiments/synthetic/mallows_poset/plot.ipynb`
- Fig. 14: `experiments/real/crowdrank/plot.ipynb`
- Fig. 15: `experiments/synthetic/mpw_parallel_pluarlity/plot.ipynb`
- Fig. 16: `experiments/synthetic/mpw_parallel_borda/plot_new.ipynb`

The experiment results in Table 6:
- MovieLens: `experiments/real/movielens/movielens_output.tsv`
- Travel: `experiments/real/travel/travel_output.tsv`
