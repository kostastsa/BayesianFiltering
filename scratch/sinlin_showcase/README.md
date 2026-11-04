# Showcase: a nonlinearity in one direction of a large state (plan step 6)

This folder is scratch code (CLAUDE.md rule 5). It is the showcase experiment of plan step 6. The results are not table results.

## Model

The simulator is `sinlin` (`gaussfiltax/simulators/sinlin.py`):

- Dynamics: x' = 0.9 x + q, with Q = 0.1 I and x_0 ~ N(0, I).
- Emission: y_1 = sin(2 a^T x) + r_1 (the curved channel), and y_j = x_j + r_j for j = 2 to d (the linear channels). R = 0.1 I.
- `rot: 0` gives a = e_1. `rot: 1` gives a fixed random unit vector a.
- T = 100 steps.

The emission is nonlinear only along a. The program (45) (`sdp`) can keep Delta = Sigma on the linear directions and shrink Delta only along a. The single rho of Eq. (47) (`auto`) cannot do this. Its value grows with d, because the linear channels add to Tr(Sigma J^T J) but not to the curvature term. `tests/test_sinlin.py` checks both statements at one component.

## Filters

All filters run on the same data. The EKF run simulates the data, and the other filters use it (data block).

- EKF: L-GSF with one component. UKF: U-GSF with one component (alpha = 1, beta = 0, kappa = 0). As in the GSF, the start mean is one draw from the prior.
- L-AGSF with M = 10 and N = L = 5, with the same rule in both steps: `prop` 0.9, `auto` 0.4 (r = 2/N), `sdp` 0.4, and `prop` 0.001 (Delta near 0, the Monte Carlo limit).
- BPF with 1000 particles and ess 0.5 (a Monte Carlo reference).

## Procedure

Do these steps in the repository root, in the conda environment `jaxenv`:

1. Run `python scratch/sinlin_showcase/make_descs.py`. This writes the files `descs/src_*.txt` and `descs/filters_*.txt`.
2. Run `./scratch/sinlin_showcase/run.zsh 1 20`. This runs `main.py trials` and `main.py errors` for all settings at the same time.
3. Run `python scratch/sinlin_showcase/compare.py 20`. This writes `out/compare.md` and `out/summary.json`.

The run time is about 1 h. Most of this time is `sdp` at d = 50 (about 150 s for each trial).

## Result (20 trials, 04-Nov-2026)

The full table is in `out/compare.md`. The page is https://claude.ai/artifact/CVBscspnJSX7P3yE6Y9JhD (private).

- The mechanism works. The rho of `auto` grows with d (0.891, 0.937, 0.948, 0.950). `sdp` keeps the eigenvalues of Sigma^-1 Lambda at about 0.65 (smallest) and 0.95 (largest) for each d.
- With a = e_1, the error does not follow. `sdp` has a lower x_1 error than `auto` in only 11, 7, 7 and 8 of 20 trials (d = 5, 10, 20, 50).
- With the rotated direction, `sdp` beats `auto` in 13 of 20 trials (d = 10) and 15 of 20 trials (d = 20). But `prop` 0.9 is as good as `sdp`.
- `sdp` takes 2 to 19 times as long as `prop` (d = 50: 151 s against 8 s).
- The BPF and the AGSF with Delta near 0 lose the linear coordinates when d increases. The Gaussian-mixture filters do not.
