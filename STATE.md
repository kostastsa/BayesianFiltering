# STATE: gaussfiltax (01-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`.

## Doing
T1 is finished. The next task of `exps` is T2.

## Done (01-Nov-2026)
- Made the tracker (D-17, D-19).
- K1: `jax.block_until_ready` in the `_run_*` methods (`gaussfiltax/experiment_runner.py`).
- K4: log-domain weights in `gaussfiltax/filters/gsf.py` and `agsf.py` (D-23). Test: `tests/test_gsf_weights.py`.
- K5: LPE from clipped eigenvalues in float64 (`gaussfiltax/util/numerics.py`, D-24). Test: last test in `tests/test_numerics_weights.py`.
- T1: GSF and AGSF rows of Tables I and III, 20 trials (D-22), `SEED_BASE=2000`. Descriptors: `exps/mtt/state_dim_4/a_*/mtt_*_table_scan_num_prt.txt` (`create_exps.py`). Rows: `figures/tables/T1_gsf_agsf_rows.md`. Log: `data.nosync/logs/T1_gsf_agsf.log`.
- Decisions D-13 (deferred: 30 trials for BPF and APF) and D-15 (default).
- Old runs of 2025-12-18 are on the flash drive `ESD-ESK` (`data/mtt_acc=*`): 3 of 6 settings, 20 trials, old code. T1 does not use them.

## Next
1. T2: BPF and APF rows of Table II (Exp. B).
2. A6, then A7 (Exp. B).
3. A1 (RPF and UPF), then A3. For A3, `runtime.txt` includes the JIT compile time (see `CLAUDE.md`).

## Open
- The new T1 rows change the text of C6 and P2 (owner `paper`). Example: the U-GSF (M = 100) is now the best GSF and AGSF row at a = 0.5, sigma2 = 2.5, and the AGSF loses tracks there.
- Decisions for the user: D-12, D-14, D-16, D-18, D-20, D-21, and D-24 (a default).
- K2 (optional). Do not run `trials` again for a trial without `results.pkl`.

## Files changed
`gaussfiltax/{experiment_runner,create_exps}.py`, `gaussfiltax/filters/{gsf,agsf}.py`, `gaussfiltax/util/numerics.py`, `tests/test_gsf_weights.py`, `tests/test_numerics_weights.py`, `CLAUDE.md`, `exps/mtt/state_dim_4/a_*/mtt_*_table_scan_num_prt.txt`, `figures/tables/T1_gsf_agsf_rows.md`, `STATE.md`.
