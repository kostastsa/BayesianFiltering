# Pilot: the adaptive AGSF in Exp. A

This folder is scratch code (CLAUDE.md rule 5). It is a short pilot of the automatic rule for rho (`auto`, `filters/agsf.py`) in Exp. A. The results are not table results.

## Setting

- Exp. A (`mtt`), wrapped bearing, a = 0.5 (Table I), sigma2 = 2.5, 0.025 and 2.5e-05.
- L-AGSF and U-AGSF with M = 100, N = L = 5.
- Four settings of the split rules (prediction / update): prop 0.9 / auto 0.2, prop 0.9 / auto 0.4, prop 0.9 / auto 0.8, auto 0.4 / auto 0.4.
- Trials 1 to 12. Each run uses the stored data of the same trial of the Table I row with prop 0.9 / prop 0.9 (data block). Thus the comparison with the table row is paired.
- The run time was 8 minutes (three settings of sigma2 at the same time).

## Procedure

Do these steps in the repository root, in the conda environment `jaxenv`:

1. Run `python scratch/agsf_adaptive_expA/make_descs.py`. This writes `descs/pilot_sigma2_*.txt`.
2. Run `./scratch/agsf_adaptive_expA/run_pilot.zsh 12`. This runs `main.py trials` and `main.py errors`.
3. Run `python scratch/agsf_adaptive_expA/compare.py 12`. This writes `out/compare.md`.

## Result (12 trials)

- sigma2 = 2.5: no setting is better than prop 0.9 / prop 0.9. The lost tracks stay (L-AGSF: 5 to 8 of 12, U-AGSF: 3 to 5 of 12).
- sigma2 = 0.025 and 2.5e-05: auto 0.4 / auto 0.4 gives a lower MSE in 11 or 12 of 12 trials, for both filters, and a lower LPE. The decrease of the MSE is small (5 % to 15 %).
- prop 0.9 / auto r2 (the adaptive variant of Sec. V-B) has no clear effect at any noise value.
