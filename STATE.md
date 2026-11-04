# STATE: gaussfiltax (04-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`.

## Doing
All experiments for the main text are done. The next task is A10 (the figure and the table of `sinlin` for the Supplementary).

## Done (04-Nov-2026)
- T5: Table R1, `scratch/particles_check/out/table_wrap.md`.
- A11 (D-38): `figures/tables/A11_r2auto_expB.md`.
- A7: `main.py time`, `figures/fig3_expB_time.pdf`. A copy replaces `errors_time.pdf` in the paper folder (the old file is `errors_time.pdf.before_A7`).
- A3 and K13: `main.py cost`, `run_timing.zsh`, `create_exps.py timing`, `GAUSSFILTAX_WARM_TIME=1`. Results: `figures/tables/A3_runtime_exp{A,B}.md`, `figures/cost_expA_sigma2_0.025.pdf`, `figures/cost_expB.pdf`.
- A8 (30 trials): `scratch/agsf_sdp_pilot/out/compare.md`.
- A10 runs (100 trials, with the U-AGSF): `scratch/sinlin_showcase/out/compare.md`.
- Decisions: D-12, D-38, D-39, D-42, D-43.
- Repair: a timing run deleted the metrics of T5 trials 1 to 5 (BPF and APF, sigma2 = 0.025). We ran them again with the old seeds, and Table R1 did not change. See the CAUTION in `CLAUDE.md`, "Storage".
- The `results.pkl` files of Exp. A and Exp. B are deleted. The metrics stay.

## Next
1. A10: make the figure (rho of Eq. (47) and the eigenvalues of `sdp` against d) and the table (the error of the coordinates 2 to d) through `main.py`.
2. A5 (PCRB), if time permits (D-43).

## Open
- User: D-40 (the A8 result is new evidence: with r = 2/N, the LPE in Exp. B is worse at N = 20 and N = 100), D-21, D-37, D-20, D-16, D-14, D-18. Defaults to confirm: D-24, D-29, D-32, D-41.
- Owner `paper`: C4 (rho2 is 0.95 and 0.47, not "close to one" and "approximately 0.5"), C6, C10 (A3 tables), C3 (A8, A10), P1 (D-12 is decided).
- The Desktop syncs to iCloud. Before a long run, check the free disk space and start the run with `caffeinate -i`.

## Files changed last
`gaussfiltax/main.py`, `gaussfiltax/experiment_runner.py`, `gaussfiltax/create_exps.py`, `run_timing.zsh`, `CLAUDE.md`, `STATE.md`.
