# APF study for task A2 (T-SP-35143-2026)

These scripts check the auxiliary particle filter (APF) of `gaussfiltax/filters/pf.py` and compare it with the bootstrap particle filter (BPF). Run each script in the folder `out/`.

## Tables I and III (reported results)

The BPF and APF rows of Tables I and III come from `main.py`, not from the scripts below.

1. `python gaussfiltax/create_exps.py` writes the descriptor files `exps/mtt/state_dim_4/a_<a>_sigma2_<sigma2>/mtt_{bpf,apf,apf_ess2}_scan_num_prt.txt`.
2. `./run_tables_I_III.zsh 1 30 0.5` and `./run_tables_I_III.zsh 1 30 0.05` run the trials and the metrics (30 trials, seed 1000 + trial number). `SEED_BASE=2000 ./run_tables_I_III.zsh 1 30 <a> apf_ess2` runs the APF that resamples at every step.
3. `python gaussfiltax/main.py errors --no-plots 1 30 <descriptor files>` writes `Error summary.txt` for all trials.
4. `python exps/apf_study/tables_I_III.py` writes `tables_I_III_rows.tex` (rows of the two tables).
5. `python exps/apf_study/compare_apf_bpf_api.py` writes `apf_vs_bpf_api.tex` and `apf_ess2_api.tex` (Appendix D.4: medians, lost trials, tests).

The two scripts in steps 4 and 5 read the files that `main.py errors` saves. They compute no error measure. The logs of the runs are in `data.nosync/logs/`.

The scripts below are checks and diagnostics. They are not a template for experiments (see CLAUDE.md). The tables of Appendix D.1 and D.3 and the diagnostic table of Appendix D.4 (first-stage weight error, ESS, resampling rate) come from them.

## Environments

- Scripts that use JAX: `python3` of the conda environment `jaxenv`.
- Scripts that use the `particles` library (`cmp_particles.py`, `cmp_report.py`, `regime_particles.py`, `align_particles.py`): `~/.venvs/particles/bin/python`. This environment has `particles` 0.4, NumPy 1.26.4 and Numba.

## Run all steps

1. `./run_final.zsh` runs the steps 1 to 3 below, and then `run_rerun.zsh`. The runs take about two hours.
2. `python3 ../make_tables.py` (in `out/`) writes `apf_tables.tex` for Appendix D of `supplementary.tex`.
3. `python3 ../summarize_rerun.py` (in `out/`) prints the result of `expA_rerun.py`.

## Scripts

| Script | Purpose |
|---|---|
| `lgssm_convergence.py` | BPF and APF against the Kalman filter for growing M |
| `apf_regime.py` | Error of the first-stage weight, ESS and resampling rate (JAX) |
| `cmp_ours.py`, `cmp_particles.py`, `cmp_report.py` | Our filters and the `particles` library on the same data |
| `regime_particles.py` | The sweep of `apf_regime.py` with the `particles` library only |
| `align_particles.py` | Effect of the time alignment of the first step |
| `expA_rerun.py` | BPF and APF in the settings of Tables I and III (script version, 100 data sets for $M\leq1000$). It is not the source of the tables. |
| `expA_apf_check.py`, `expA_apf_variants.py`, `expA_beat_bpf.py`, `expA_oracle_floor.py`, `apf_vs_kf.py` | Earlier tests: variants of the APF, the 20-step look-ahead, the lower bound of the best initial path |
| `debug_nan.py` | Finds the first NaN in a BPF run (the fault in the normalization of the weights) |

## Notes

- `out_before_logfix/` holds the results of `expA_rerun.py` before the weights were normalized in the log domain. The BPF has NaN values in these files.
- The column `bpf_pre` of the files with $M\leq1000$ was computed with a copy of the BPF that still multiplies the weights in the linear domain. It has NaN values. Do not use it.
- The study was done on 21-Oct-2026. Task A2 in `tasks.md` of the review folder has the results.
- `make_tables.py` and `summarize_rerun.py` belong to the script version. The tables of the paper now come from `tables_I_III.py` and `compare_apf_bpf_api.py`.
