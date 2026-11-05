# Scatter panels of Exp. A

This folder is scratch code (CLAUDE.md rule 5). It makes figures for each (a, sigma2) setting of Tables I and III. Each figure has 8 columns (the filters at the middle sizes) and 3 rows (the best, median and worst table run of each filter). One set of figures ranks the runs by MSE, and one set ranks them by LPE. The panels come from `FilterRunner.plot_three_panel_scatter` in `standalone_filter_runner.py`.

## Runs

The figures show the table runs. Each filter has its own 100 trials, so each cell can show a different trajectory. The scripts repeat the selected trials exactly into `data.nosync/repro` (`GAUSSFILTAX_ROOT`, `main.py trials --only`). The table folders do not change. `check_v1.py` compares the MSE and the LPE of each repeat with the table.

A paired version (all filters on the same data sets) was removed on 05-Nov-2026 (D-44). Its scripts are in the git history (commit `29be2a3`).

## Filters

L-GSF and U-GSF with M = 100, L-AGSF and U-AGSF with M = 10 (N = L = 5, `prop` 0.9), and the BPF, APF, RPF and UPF with 1000 particles (`common.py`).

## Procedure

Do these steps in the repository root, in the conda environment `jaxenv`. Use `mse` or `lpe` as METRIC.

1. Run `./scratch/scatter_panels/run_v1.zsh METRIC`. This repeats the selected table trials (about 15 min).
2. Run `python scratch/scatter_panels/check_v1.py METRIC`. The result must be "0 differ from the table".
3. Run `python scratch/scatter_panels/plot_panels.py METRIC`.

The figures are in `figures/`: `v1_a<a>_s<s>` (MSE) and `v1_lpe_a<a>_s<s>` (LPE), in PNG and PDF. The selections are in `out/v1_selection.json` and `out/v1_selection_lpe.json`. The page source is `page/index.html`.
