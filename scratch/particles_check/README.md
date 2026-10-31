# Check of the BPF and APF against the `particles` library

This folder is scratch code (CLAUDE.md rule 5). It checks that our BPF and APF (`gaussfiltax/filters/pf.py`) give the same accuracy as the BPF and APF of the `particles` library (Chopin and Papaspiliopoulos, 2020). The two libraries run on the same data sets of Exp. A.

## Setting

- Exp. A (`mtt`), a = 0.5, sigma2 = 2.5, 0.025 and 2.5e-05, M = 1000. These are the columns of Table I.
- The data sets are the 30 trials of the `main.py` runs with 1000 particles in `data.nosync/experiments/mtt/`.
- `main.py` splits the key again for each descriptor. Thus the BPF and APF runs of one trial have different data. Each library runs its BPF on the BPF data and its APF on the APF data.
- Ours: 20 runs on each data set, made with `main.py` and a data block (`run_samedata.zsh`). If these runs are not complete, `make_table.py` uses the single runs of the 30 trials.
- `particles`: 50 runs on each data set.
- Both libraries use multinomial resampling, the ESS threshold 0.5, the same auxiliary function, and the same initial step (see `run_particles.py`).
- `run_particles.py` checks its numpy model against the JAX dynamics and emission functions before it runs.

## Procedure

Run these commands in the repository root:

1. `SEED_BASE=5000 ./run_samedata.zsh 20 0.5 0.025,2.5,2.5e-05 apf` and the same with `bpf` and another `SEED_BASE` (conda environment `jaxenv`). Each setting takes about 20 minutes for each filter.
2. `python3 scratch/particles_check/export_gt.py 0.5 <sigma2>` (`jaxenv`), for each sigma2. This writes `out/gt_a0.5_sigma2_<sigma2>.npz`.
3. `~/.venvs/particles/bin/python scratch/particles_check/run_particles.py 0.5 <sigma2>`, for each sigma2. This writes `out/particles_a0.5_sigma2_<sigma2>.npz`. Each setting takes about 3 minutes.
4. `python3 scratch/particles_check/make_table.py 0.5 2.5,0.025,2.5e-05` (`jaxenv`). This writes `out/table.md` and `out/table.tex`.

Write sigma2 as in the folder names (2.5e-05, not 2.5e-5). Both libraries use `util.numerics.mse` for the MSE. The column definitions are in the docstring of `make_table.py`.

## Notes

- `particles` 0.4 (its `__version__` string reads 0.3alpha) has an underflow in `resampling.log_mean_exp`. This underflow makes all APF weights NaN in 3 of 1500 runs (sigma2 = 0.025, data set 27, step 73). `run_particles.py` replaces the function with a stable version that gives the same value. Our `pf.py` does not have this fault.
