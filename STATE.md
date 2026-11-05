# STATE: gaussfiltax (05-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`.

## Doing
All experiments for the paper are done. The last task was A12 (scatter panels of Exp. A).

## Done (04-Nov and 05-Nov-2026)
- A10: `main.py dim`, `figures/dim_sinlin.pdf`, `figures/tables/A10_sinlin.md`, `exps/sinlin/sinlin_showcase.txt`. `errors` writes `eig2_mm.pkl`.
- A12: `scratch/scatter_panels/` (see its `README.md`). The table runs, ranked by MSE and by LPE. All repeats give the MSE and LPE of the table. Page: https://claude.ai/artifact/4cJAono68DqzZ4FLdjTyB3. The paired version and the pageview version are removed.
- The MSE figures are in the paper folder as `scatter_mse_a<a>_s<sigma2>.pdf` (D-45).
- New options: `main.py trials --only`, `GAUSSFILTAX_ROOT` (see `CLAUDE.md`).
- Decisions: D-14, D-18 (K3 deferred), D-20, D-37 (deferred), D-40, D-44, D-45.
- Earlier on 04-Nov: T5, A3 and K13, A7, A8, A11, see the tracker.

## Next
1. A5 (PCRB), if time permits (D-43).

## Open
- User: D-21 (the reviewer R2.4 asks for Algs. 2 to 4 in the main text), D-16 (the weighted metrics change the GSF numbers). Defaults to confirm: D-24, D-29, D-32, D-41.
- Owner `paper`: replace the old `comparison_scatter_*.pdf` and their captions with `scatter_mse_*.pdf` (D-45). C4 (rho2 is 0.95 and 0.47), C6, C10 (A3 tables), C3 (A8, A10), P1 (Table R1).
- The repeats of A12 are in `data.nosync/repro` (about 1.9 GB). Delete them when the figures are final.
- The Desktop syncs to iCloud. Before a long run, check the free disk space and start the run with `caffeinate -i`.

## Files changed last
`scratch/scatter_panels/*`, `gaussfiltax/main.py`, `gaussfiltax/misc.py`, `CLAUDE.md`, `.gitignore`, `STATE.md`.
