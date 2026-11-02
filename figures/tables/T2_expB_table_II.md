# Table II (Exp. B, lmsv): new rows

Source: the `Error summary.txt` files of `main.py errors` (fixed bootstrap key). The rows come from `main.py table`.
Each cell: mean ± standard deviation of 100 bootstrap means over 20 (GSF, AGSF) or 30 (BPF, APF, RPF, UPF) trials.
Settings (D-32): AGSF with N = L = 5, rho1 = 0.9 fixed, rho2 from the automatic rule (r2 = 0.4). `SEED_BASE` = 4000.
NEES: the mean fraction of steps with NEES above the 0.999 quantile of chi2 (4 degrees of freedom). In parentheses: the trials with more than 20 % of such steps.
Lost (MSE above 10) is not useful here, because the MSE of a good filter is near 15. Do not report it for Exp. B.

## Table II

| Filter | M | MSE | LPE | NEES: outside 99.9 % (trials > 20 %) | Lost |
|---|---|---|---|---|---|
| L-GSF | 10 | 48.91 ± 1.76 | 6.54 ± 0.03 | 0.1 % (0 of 20) | 20 of 20 |
| L-GSF | 100 | 49.12 ± 1.40 | 6.55 ± 0.03 | 0.1 % (0 of 20) | 20 of 20 |
| L-GSF | 1000 | 49.57 ± 1.54 | 6.56 ± 0.04 | 0.1 % (0 of 20) | 20 of 20 |
| U-GSF | 10 | 48.68 ± 1.25 | 6.55 ± 0.03 | 0.1 % (0 of 20) | 20 of 20 |
| U-GSF | 100 | 49.10 ± 1.57 | 6.55 ± 0.04 | 0.1 % (0 of 20) | 20 of 20 |
| U-GSF | 1000 | 49.90 ± 1.66 | 6.55 ± 0.04 | 0.1 % (0 of 20) | 20 of 20 |
| BPF | 100 | 22.22 ± 0.34 | 2.04e+06 ± 6.64e+04 | 56.4 % (30 of 30) | 30 of 30 |
| BPF | 1000 | 16.02 ± 0.23 | 4.02e+05 ± 1.17e+04 | 49.0 % (30 of 30) | 30 of 30 |
| BPF | 10000 | 14.57 ± 0.27 | 1.24e+05 ± 7191.52 | 30.6 % (30 of 30) | 30 of 30 |
| APF | 100 | 19.37 ± 0.32 | 7.54e+05 ± 2.46e+04 | 52.8 % (30 of 30) | 30 of 30 |
| APF | 1000 | 16.06 ± 0.26 | 6.44e+04 ± 4055.12 | 48.0 % (30 of 30) | 30 of 30 |
| APF | 10000 | 15.84 ± 0.34 | 3.36e+04 ± 1.74e+04 | 28.9 % (30 of 30) | 30 of 30 |
| RPF | 100 | 22.31 ± 0.34 | 1.83e+06 ± 4.58e+04 | 54.3 % (30 of 30) | 30 of 30 |
| RPF | 1000 | 15.91 ± 0.23 | 4.09e+05 ± 1.27e+04 | 48.8 % (30 of 30) | 30 of 30 |
| RPF | 10000 | 14.61 ± 0.32 | 1.12e+05 ± 6182.83 | 29.0 % (30 of 30) | 30 of 30 |
| UPF | 100 | 17.18 ± 0.40 | 9.46e+04 ± 2.82e+04 | 5.9 % (0 of 30) | 30 of 30 |
| UPF | 1000 | 15.14 ± 0.28 | 2066.17 ± 1152.27 | 0.7 % (0 of 30) | 30 of 30 |
| UPF | 10000 | 15.15 ± 0.29 | 1075.18 ± 168.72 | 0.4 % (0 of 30) | 30 of 30 |
| L-AGSF | 10 | 18.12 ± 0.42 | 5.62 ± 0.03 | 0.2 % (0 of 20) | 20 of 20 |
| L-AGSF | 100 | 15.73 ± 0.33 | 5.45 ± 0.01 | 0.0 % (0 of 20) | 20 of 20 |
| L-AGSF | 1000 | 15.67 ± 0.36 | 5.47 ± 0.02 | 0.0 % (0 of 20) | 20 of 20 |
| U-AGSF | 10 | 18.17 ± 0.33 | 5.61 ± 0.02 | 0.2 % (0 of 20) | 20 of 20 |
| U-AGSF | 100 | 15.73 ± 0.33 | 5.45 ± 0.02 | 0.0 % (0 of 20) | 20 of 20 |
| U-AGSF | 1000 | 15.72 ± 0.36 | 5.47 ± 0.02 | 0.0 % (0 of 20) | 20 of 20 |

