# STATE: gaussfiltax (02-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`.

## Doing
Tables I, II and III are complete with the new code. The next task of `exps` is A6.

## Done (01 to 02-Nov-2026)
- A1: RPF and UPF in `gaussfiltax/filters/pf.py` (D-26, D-27, D-29). Test: `tests/test_pf_vs_kalman.py`.
- D-30: wrapped bearing for `mtt` (key `wrap: 1`, `NonlinearSSM.emission_residual`). Test: `tests/test_wrapped_bearing.py`. Pilot and diagnostics: `scratch/d30_pilot/`, `scratch/agsf_lost_tracks/`.
- T4: all rows of Tables I and III with the wrapped bearing: `figures/tables/T4_wrapped_tables_I_III.md`.
- K6 (D-31): NEES metric (`util/numerics.py`, `main.py errors`, `main.py table`). Test: `tests/test_nees.py`.
- K7: fixed bootstrap key in `main.py errors` and `ensemble`.
- P3: Tables I and III replaced in `main.tex` and `supplementary.tex` (generator `scratch/d30_pilot/latex/make_tables.py`, old files `*.before_T4_tables`).
- T2 (D-32): all rows of Table II (Exp. B): `figures/tables/T2_expB_table_II.md`. `create_exps.py lmsv`, `SIM=lmsv ./run_tables_I_III.zsh`.
- The raw-bearing `results.pkl` and `aux_outputs.pkl` are deleted. Their metrics and summaries stay.
- Previews on claude.ai: `scratch/d30_pilot/page/*.html`.

## Next
1. Replace Table II in `main.tex` (decide first: without the run time column, or after A3).
2. A6 (sweep of fixed rho2 in Exp. B), then A7 (Fig. 3 with U-GSF and U-AGSF).
3. A3: run time without the JIT compile time.
4. A5 (PCRB) if time remains.

## Open
- User: Table II run time column (A3)? Tables I and III have a very small font with 14 columns (remove the LPE sd, or move NEES and Lost to the Supplementary?).
- Owner `paper`: the text of Exp. A cites the old numbers (C6, P2). The text must define the NEES and the wrapped bearing. Table R1 keeps the raw model (D-30).
- Decisions for the user: D-12, D-14, D-16, D-18, D-20, D-21. Defaults to confirm: D-24, D-29, D-32.
- Do not run `trials` again for a trial without `results.pkl` (except the same-seed re-run, see `CLAUDE.md`).

## Files changed last
`gaussfiltax/{create_exps,main}.py`, `run_tables_I_III.zsh`, `exps/lmsv/state_dim_4/lmsv_*_tableII_scan_num_prt.txt`, `figures/tables/T2_expB_table_II.md`, `scratch/d30_pilot/{latex,page}/`, `STATE.md`.
