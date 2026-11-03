### L-AGSF, M = 100: fixed rho2 against the adaptive rho2 (same 100 data sets)

Adaptive: median MSE 15.16, median LPE 5.449.

| rho2 | Median MSE | Median (MSE fixed - adaptive) | Trials fixed < adaptive (MSE) | p (MSE) | Median LPE | Median (LPE fixed - adaptive) | Trials fixed < adaptive (LPE) | p (LPE) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 14.953 | -0.361 | 79 of 100 | 9.5e-11 | 5.658 | +0.201 | 3 of 100 | 5.9e-18 |
| 0.3 | 14.968 | -0.209 | 74 of 100 | 6.2e-09 | 5.358 | -0.087 | 98 of 100 | 6.5e-18 |
| 0.5 | 15.277 | -0.007 | 52 of 100 | 0.87 | 5.458 | +0.015 | 28 of 100 | 1.7e-06 |
| 0.7 | 16.537 | +0.927 | 7 of 100 | 4.7e-17 | 5.633 | +0.176 | 0 of 100 | 3.9e-18 |
| 0.9 | 20.196 | +4.573 | 0 of 100 | 3.9e-18 | 5.870 | +0.415 | 0 of 100 | 3.9e-18 |

| rho2 | Mean MSE, linear steps (u = 0) | Mean MSE, MSV steps (u = 1) |
|---|---|---|
| adaptive | 0.393 | 30.364 |
| 0.1 | 0.397 | 29.516 |
| 0.3 | 0.393 | 29.816 |
| 0.5 | 0.393 | 30.350 |
| 0.7 | 0.393 | 32.313 |
| 0.9 | 0.393 | 39.644 |

Adaptive rho2 (20 trials with aux_outputs.pkl): mean 0.950 in the linear steps (u = 0), 0.475 in the MSV steps (u = 1).

### U-AGSF, M = 100: fixed rho2 against the adaptive rho2 (same 100 data sets)

Adaptive: median MSE 15.21, median LPE 5.438.

| rho2 | Median MSE | Median (MSE fixed - adaptive) | Trials fixed < adaptive (MSE) | p (MSE) | Median LPE | Median (LPE fixed - adaptive) | Trials fixed < adaptive (LPE) | p (LPE) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 14.901 | -0.508 | 80 of 100 | 1.3e-11 | 5.707 | +0.262 | 3 of 100 | 7.8e-18 |
| 0.3 | 15.087 | -0.226 | 73 of 100 | 4.9e-08 | 5.367 | -0.076 | 98 of 100 | 4.7e-18 |
| 0.5 | 15.140 | -0.014 | 52 of 100 | 0.67 | 5.459 | +0.015 | 26 of 100 | 3.4e-08 |
| 0.7 | 16.275 | +0.991 | 7 of 100 | 3.4e-17 | 5.614 | +0.176 | 0 of 100 | 3.9e-18 |
| 0.9 | 20.008 | +4.466 | 0 of 100 | 3.9e-18 | 5.866 | +0.419 | 0 of 100 | 3.9e-18 |

| rho2 | Mean MSE, linear steps (u = 0) | Mean MSE, MSV steps (u = 1) |
|---|---|---|
| adaptive | 0.393 | 30.358 |
| 0.1 | 0.397 | 29.471 |
| 0.3 | 0.394 | 29.807 |
| 0.5 | 0.393 | 30.413 |
| 0.7 | 0.393 | 32.306 |
| 0.9 | 0.393 | 39.805 |

Adaptive rho2 (20 trials with aux_outputs.pkl): mean 0.950 in the linear steps (u = 0), 0.476 in the MSV steps (u = 1).
