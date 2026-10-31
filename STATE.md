# STATE: gaussfiltax (01-Nov-2026)

## Doing
Check of our BPF and APF against the `particles` library for the reviewer (Reviewer 1, point 4). The check is finished. Table R1 is in the response letter.

## Done (30-Oct to 01-Nov-2026)
- Moved `exps/apf_study/` to `scratch/apf_study/`. New rule 5 in `CLAUDE.md`: new experiment code goes in `gaussfiltax/` and runs through `main.py`. Scripts for side questions go in `scratch/<topic>/`.
- New optional `data` block in descriptors (BPF and APF only): the filter runs on the stored data of another experiment, with a new filter key (`experiment_descriptor.py`, `experiment_runner.py`, `create_exps.py`, files `exps/mtt/state_dim_4/a_*/mtt_{bpf,apf}_samedata_scan_num_prt.txt`, driver `run_samedata.zsh`).
- `main.py trials` now splits the key for each trial, also for a skipped trial. A run from an empty folder gets the same keys as before.
- `monitor.py` shows a, sigma2 and the data source.
- Ran 20 repetitions of our BPF and APF on each of the 30 data sets (M = 1000, a = 0.5, sigma2 = 2.5, 0.025, 2.5e-05). That is 3600 trials in `data.nosync/.../data_pf_*`. Their `results.pkl` files are deleted, and `mse.pkl` stays.
- `scratch/particles_check/`: `export_gt.py`, `run_particles.py` (50 `particles` runs on each data set), `make_table.py`. Result: `out/table.md`. 11 of 12 tests give p >= 0.10. The exception is the APF at sigma2 = 2.5 (sign test p = 0.01, not significant after Bonferroni).
- Found: `main.py` gives the BPF and the APF different data for the same trial number. `particles` 0.4 has an underflow in `log_mean_exp`, and `run_particles.py` replaces this function.
- Response letter (`~/Desktop/TSP Resubmission review/Response/response_letter_R2.tex`): Table R1 and text in Reviewer 1, point 4 (ii). The PDF is updated.

## Next
1. Re-run the GSF and AGSF rows of Tables I and III through `main.py` with the weighted metrics.
2. Re-run the BPF and APF rows of Exp. B (`lmsv`, Table II).
3. Add `jax.block_until_ready` in the `_run_*` methods of `experiment_runner.py` before you compare run times.
4. Then A1: RPF and UPF.

## Open
- Appendix D.3 of `supplementary.tex` still has the old `particles` table (`tab:apf ref`). Replace it with Table R1? The user said "not yet".
- Optional: a float64 run of our APF at sigma2 = 2.5, to find the cause of the small APF difference.
- 30 trials per BPF/APF entry in Tables I and III, against 100 for the other rows: extend or keep?
- To make the tables of `scratch/apf_study/` again, move their code into `gaussfiltax/` and `main.py` (rule 5).
- Initial-step mismatch (filters propagate the prior first): unchanged.
- Do not run `trials` again for a trial without `results.pkl`. This now includes the `*_samedata_*` runs. New trials need a new `SEED_BASE`.
- `tasks.md` (line 85) of the review folder still describes the old `particles` comparison.

## Files changed
`CLAUDE.md`, `STATE.md`, `.gitignore`, `run_samedata.zsh`, `gaussfiltax/{main,experiment_descriptor,experiment_runner,create_exps,monitor}.py`, `exps/mtt/state_dim_4/a_*/mtt_*_samedata_scan_num_prt.txt`, `scratch/apf_study/`, `scratch/particles_check/`, `exps/apf_study/` (deleted).
