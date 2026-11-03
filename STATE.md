# STATE: gaussfiltax (03-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`.

## Doing
A6 and the Exp. A pilot (O3) are done. The next step waits for the user (D-38).

## Done (03-Nov-2026)
- P4: Tables I, II and III (100 trials, T3) are in `main.tex` and `supplementary.tex`. Generator: `scratch/d30_pilot/latex/make_tables.py` (now also `table_II.tex`). Copies of the old files: `*.before_T3_tables`. `main.pdf` has 14 pages.
- A6 (D-35): sweep of fixed rho2 in Exp. B, paired with Table II. Rows: `figures/tables/A6_rho2_sweep_expB.md`. Paired check and Fig. 3 curves: `scratch/a6_rho2/`. Page: https://claude.ai/artifact/R4RbVcirjgchT4mF7SehPU
- K8: data block for the L-AGSF and U-AGSF (`experiment_runner.py`). `main.py table` shows the rho2 rule of the AGSF rows.
- K9: seed fault in the chunk mode of `run_samedata.zsh` (cause of the T5 lost-track difference). Fixed, not run again.
- O3: pilot of the auto rule in Exp. A: `scratch/agsf_adaptive_expA/README.md`.

## Next
1. T5: remove trials 61 to 600 of the 6 `*_wrap_samedata` settings, run them again (`scratch/particles_check/run_r1_wrap.zsh`), and make Table R1 again.
2. D-38: if the user agrees, run the auto rule with r2 in {0.1, 0.2, 0.8} in Exp. B (data block on the Table II data, about 40 minutes).
3. A7: Fig. 3 through `main.py`, with the curve of the best fixed rho2.
4. A3: run time without the JIT compile time.
5. A5 (PCRB) if time remains. Exp. A auto/auto with 100 trials only if time remains.

## Open
- User: D-38 (claim of Sec. V-B), D-37 (font of Tables I and III, "later"), D-21 ("later"). Also D-12, D-14, D-16, D-18, D-20. Defaults to confirm: D-24, D-29, D-32.
- Owner `paper`: the text of Exp. A and Exp. B cites old numbers (N_sim = 30, 36 % and 28 %, "outperforms", rho2 near 0.5). C4 and C6 must use the A6 result.
- `run_r1_wrap.zsh` skips a setting whose `Error summary.txt` has 600 trials. Remove the old trial folders 61 to 600 first.

## Files changed last
`gaussfiltax/{experiment_runner,create_exps,main}.py`, `run_samedata.zsh`, `CLAUDE.md`, `exps/lmsv/state_dim_4/lmsv_*_rho2sweep_scan_num_prt.txt`, `scratch/{a6_rho2,agsf_adaptive_expA}/`, `scratch/d30_pilot/latex/`, `figures/tables/A6_rho2_sweep_expB.md`.
