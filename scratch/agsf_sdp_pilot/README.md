# Pilot A8: the split rules prop, auto and sdp for different N

This folder is scratch code (CLAUDE.md rule 5). It is the small pilot of task A8 (decision D-39). The results are not table results.

## Setting

- L-AGSF with M = 10 and N = L in {5, 20, 100}.
- Three rules, the same rule in the prediction and in the update: `prop` 0.9, `auto` 2/N (Eq. (47)) and `sdp` 2/N (program (45)). The rules are in `filters/agsf.py`.
- Exp. A (`mtt`), wrapped bearing, a = 0.5, sigma2 = 0.025. The runs use the stored data of the Table I row (L-AGSF, M = 100, prop 0.9 / prop 0.9).
- Exp. B (`lmsv`). The runs use the stored data of the Table II row (L-AGSF, M = 100, prop 0.9 / auto 0.4).
- Trials 1 to 5. Trial t uses the data of trial t of the reference row (data block). Thus the comparison is paired.

## Procedure

Do these steps in the repository root, in the conda environment `jaxenv`:

1. Run `python scratch/agsf_sdp_pilot/make_descs.py`. This writes `descs/pilot_expA.txt` and `descs/pilot_expB.txt`.
2. Run `./scratch/agsf_sdp_pilot/run_pilot.zsh 5`. This runs `main.py trials` and `main.py errors` for the two files at the same time.
3. Run `python scratch/agsf_sdp_pilot/compare.py 5`. This writes `out/compare.md`.

## Result (5 trials, 03-Nov-2026)

The full table is in `out/compare.md`. The run time was 11 minutes for Exp. A and Exp. B at the same time.

- Exp. A: no rule is clearly better than `prop` 0.9. The differences of the mean MSE are 10 % or less, and the paired counts are 2 to 4 of 5. `auto` keeps rho2 near the clip value 0.95. `sdp` at N = 100 gives a lower rho2 (0.80) and a worse LPE.
- Exp. B: `auto` and `sdp` give the same mean rho1 (0.95, the clip, because the dynamics are linear) and the same mean rho2. Both decrease the MSE by 15 % to 25 % in 5 of 5 trials at each N. At N = 20 and N = 100 the LPE is worse than with `prop` 0.9 (N = 100: 15.4 against 5.9).
- In Exp. B, "Lost" (MSE > 10) is 5 of 5 for all settings, also for the reference row. This threshold is for Exp. A only.
- Run time of one trial: `sdp` takes 2 to 4 times as long as `prop` (Exp. A, N = 100: 77 s against 24 s).
