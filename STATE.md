# STATE: gaussfiltax (21-Oct-2026)

## Doing
Re-running the BPF and APF rows of Exp. A (simulator `mtt`, Tables I with a=0.5 and III with a=0.05) through `main.py`, with the corrected filters (log-domain weights, APF resamples by the ESS rule). This part is finished. The other rows are not.

## Done (this session)
- Ran 48 experiments x 30 trials with `run_tables_I_III.zsh` (BPF and APF, 3 noise levels sigma2, M = 1e2..1e5, a = 0.5 and 0.05). Also the APF with `ess: 2.0` (resamples at every step, the old rule), M = 1000, 6 settings x 30 trials. Metrics and `Error summary.txt` are in `data.nosync/experiments/mtt/` (not in git). Raw `results.pkl` of M >= 1e4 are deleted (1.2 GB each at M = 1e5).
- Results: no APF-vs-BPF test is significant (0 of 24; Mann-Whitney smallest p = 0.052). For sigma2 in {2.5, 0.025} lost trials (MSE > 10): BPF 62/480, APF 72/480 (p = 0.40). The old rule loses more (61/180 vs 41/180, p = 0.026). Means are dominated by lost trials (median MSE of a lost trial 92, of the others 0.12).
- The M = 1e5 mismatch with the old script study was chance: the script loses 14/100, `main.py` 9/30. Script path and runner path give identical MSE for the same data and key.
- API changes: `mtt` descriptor with optional `a`, `sigma2`; `main.py errors --no-plots`, `Error summary.txt`, fallback to saved metrics; lighter `mse`/`bias`/`min_error`, `log_prob_error` via `lax.map` (values unchanged); `create_exps.py` for the six settings. Documented in `CLAUDE.md` ("Large runs").
- `gaussfiltax/monitor.py` (written by another session) was run on port 8001 and is now stopped.

## Next
1. Re-run the GSF and AGSF rows of Tables I and III through `main.py` with the weighted metrics (the old LPE and bias of the GSFs ignore the weights).
2. Re-run the BPF and APF rows of Exp. B (`lmsv`, Table II): still from the old filters.
3. Optional, to reach 100 trials per entry: `SEED_BASE=3000 ./run_tables_I_III.zsh 31 100 <a>`, then `python gaussfiltax/main.py errors --no-plots 1 100 <files>`, and set `N_TRIALS` in `exps/apf_study/compare_apf_bpf_api.py`. About 6 hours per table.
4. Add `jax.block_until_ready` in the `_run_*` methods of `experiment_runner.py` before comparing run times: `runtime.txt` records 0.8 s where the result is ready after 3.3 s (M = 1e5).
5. Then A1: RPF and UPF.

## Open
- 30 trials per BPF/APF entry against 100 for the other rows: extend (Next 3) or keep?
- The verification scripts in `exps/apf_study/` (Kalman check, `particles` comparison, first-stage weight diagnostics) are not `main.py` runs. Acceptable, or move into the API?
- Initial-step mismatch (filters propagate the prior first, the simulator draws x_0 directly): unchanged.
- Never run `trials` again for a trial without `results.pkl` (it deletes the folder and redoes the trial). New trials need a new `SEED_BASE`.
- The original `CLAUDE.md` is on branch `worktree-claude-md`. The root copy has extra edits.

## Files changed
- `gaussfiltax/`: `filters/pf.py`, `simulators/mtt.py`, `simulators/mtt_params.py`, `experiment_descriptor.py`, `experiment_runner.py`, `main.py`, `create_exps.py`, `util/numerics.py`
- `tests/test_pf_vs_kalman.py`, `tests/test_numerics_weights.py`
- `run_tables_I_III.zsh`, `CLAUDE.md`, `.gitignore`, `STATE.md`
- `exps/mtt/state_dim_4/a_<a>_sigma2_<sigma2>/mtt_{bpf,apf,apf_ess2}_scan_num_prt.txt`
- `exps/apf_study/`: `README.md`, `tables_I_III.py`, `compare_apf_bpf_api.py`, `tables_I_III_rows.tex`, `apf_vs_bpf_api.tex`, `apf_ess2_api.tex`, and the earlier scripts
