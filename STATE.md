# STATE: gaussfiltax (03-Nov-2026)

Tracker: `~/Desktop/TSP Resubmission review/` (`TODO.md`, `DECISIONS.md`, `MAP.md`). This session is the owner `exps`. Plan of this work: `~/Desktop/TSP Resubmission review/implementation-improvement-plan.md` (steps 1 to 5).

## Doing
Plan step 4 (A8): a small pilot (D-39). It is designed but not run. This computer has no `data.nosync`, so run it on the computer with the stored data.

## Done (03-Nov-2026)
- K10 (plan step 1): `auto` and `sdp` scale the outputs of the update by (G R G^T)^-1/2. In Exp. A and Exp. B, rho does not change.
- K11 (plan step 2): the new rule `sdp` (program (45), FISTA, 200 iterations), descriptor value `ac1: sdp` / `ac2: sdp`, in `gaussfiltax/filters/agsf.py`. Settings: D-41. One Exp. A trial with M = 100 takes 50 s (`prop`: 3 s).
- Tests: `tests/test_cov_selection.py` (all pass). With `prop` / `auto`, the old and the new code give the same rho. Exp. A runs with `auto` in the update do not repeat bit for bit (float32 rounding, amplified by the resampling).
- `CLAUDE.md`: the rules `prop`, `auto` and `sdp`, and the new test.

## Next
1. A8 pilot (D-39): write `scratch/agsf_sdp_pilot/` (as `scratch/agsf_adaptive_expA/`: `make_descs.py`, `run_pilot.zsh`, `compare.py`). L-AGSF, M = 10, N = L in {5, 20, 100}, `prop` 0.9 / `auto` 2/N / `sdp` 2/N in both steps. Exp. A (a = 0.5, sigma2 = 0.025, data of `M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9`) and Exp. B (data of `M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4`). 5 trials. Report paired MSE and LPE, and the mean rho1 and rho2. Then show the result to the user.
2. T5: remove trials 61 to 600 of the 6 `*_wrap_samedata` settings, run them again (`scratch/particles_check/run_r1_wrap.zsh`), and make Table R1 again.
3. D-38: if the user agrees, run `auto` with r2 in {0.1, 0.2, 0.8} in Exp. B (about 40 minutes).
4. A7: Fig. 3 through `main.py`. Then A3 (run time without the compile time), A5 (PCRB) if time remains.

## Open
- User: D-39 full scope (after the pilot), D-40 (clip 0.95 and r = 2/L: paper sentence or code change), D-38, D-37 and D-21 ("later"), D-12, D-14, D-16, D-18, D-20. Defaults to confirm: D-24, D-29, D-32, D-41.
- Owner `paper`: plan steps 3 and 5 (text). The text of Exp. A and Exp. B cites old numbers. C4 and C6 must use the A6 result.
- `run_r1_wrap.zsh` skips a setting whose `Error summary.txt` has 600 trials. Remove the old trial folders 61 to 600 first.

## Files changed last
`gaussfiltax/filters/agsf.py`, `tests/test_cov_selection.py`, `CLAUDE.md`, `STATE.md`.
