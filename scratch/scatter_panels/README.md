# Scatter panels of Exp. A

This folder is scratch code (CLAUDE.md rule 5). It makes one figure for each (a, sigma2) setting of Tables I and III. Each figure has 8 columns (the filters at the middle sizes) and 3 rows (the best, median and worst run by MSE). The panels come from `FilterRunner.plot_three_panel_scatter` in `standalone_filter_runner.py`.

## Two versions

- **Version 1 (`v1`)**: the table runs. Each filter has its own 100 trials, so each cell can show a different trajectory. The scripts repeat the selected trials exactly into `data.nosync/repro` (`GAUSSFILTAX_ROOT`, `main.py trials --only`). The table folders do not change. `check_v1.py` shows that all 144 repeats give the MSE of the table.
- **Version 2 (`v2`)**: paired runs. All 8 filters run on the same 40 stored data sets (data block, the U-GSF row with M = 1). The data sets are ranked by the median MSE of the 8 filters. Each row shows one trajectory for all filters.

## Filters

L-GSF and U-GSF with M = 100, L-AGSF and U-AGSF with M = 10 (N = L = 5, `prop` 0.9), and the BPF, APF, RPF and UPF with 1000 particles (`common.py`).

## Procedure

Do these steps in the repository root, in the conda environment `jaxenv`:

1. Run `./scratch/scatter_panels/run_v1.zsh`. This repeats the selected table trials (about 20 min).
2. Run `python scratch/scatter_panels/check_v1.py`. The result must be "0 differ from the table".
3. Run `python scratch/scatter_panels/make_v2_descs.py`. This writes `descs/paired_*.txt`.
4. Run `./scratch/scatter_panels/run_v2.zsh 1 40`. This runs the paired trials (about 30 min).
5. Run `./scratch/scatter_panels/run_v2_select.zsh`. This runs the 3 selected trials of each setting again for `results.pkl`.
6. Run `python scratch/scatter_panels/plot_panels.py v1` and `python scratch/scatter_panels/plot_panels.py v2`.

The figures are in `figures/` (`v1_a<a>_s<s>` and `v2_a<a>_s<s>`, PNG and PDF). The selections are in `out/v1_selection.json` and `out/v2_selection.json`.
