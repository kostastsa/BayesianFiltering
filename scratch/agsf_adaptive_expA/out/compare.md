Exp. A (mtt, wrapped bearing), a = 0.5, M = 100, N = L = 5. Trials 1 to 12 of the Table I data sets (paired). Lost: MSE > 10. NEES: mean fraction of steps outside the 99.9 % region.

### sigma2 = 2.5

| Filter | Rules (prediction / update) | Mean MSE | Median MSE | Lost | Mean LPE | NEES | MSE below the reference |
|---|---|---|---|---|---|---|---|
| L-AGSF | prop 0.9 / prop 0.9 (Table I) | 12.100 | 8.479 | 5 of 12 | 6.68 | 2.9 % | - |
| L-AGSF | prop 0.9 / auto 0.2 | 14.408 | 9.137 | 5 of 12 | 6.39 | 3.4 % | 6 of 12 |
| L-AGSF | prop 0.9 / auto 0.4 | 15.180 | 10.990 | 6 of 12 | 5.93 | 1.9 % | 5 of 12 |
| L-AGSF | prop 0.9 / auto 0.8 | 16.436 | 9.573 | 6 of 12 | 7.00 | 5.5 % | 6 of 12 |
| L-AGSF | auto 0.4 / auto 0.4 | 14.448 | 11.438 | 8 of 12 | 6.35 | 4.2 % | 5 of 12 |
| U-AGSF | prop 0.9 / prop 0.9 (Table I) | 8.358 | 8.586 | 3 of 12 | 4.87 | 5.2 % | - |
| U-AGSF | prop 0.9 / auto 0.2 | 8.772 | 6.743 | 5 of 12 | 4.28 | 6.9 % | 6 of 12 |
| U-AGSF | prop 0.9 / auto 0.4 | 8.188 | 7.581 | 3 of 12 | 4.00 | 5.0 % | 6 of 12 |
| U-AGSF | prop 0.9 / auto 0.8 | 8.150 | 7.328 | 4 of 12 | 4.01 | 4.8 % | 8 of 12 |
| U-AGSF | auto 0.4 / auto 0.4 | 9.876 | 7.714 | 4 of 12 | 6.96 | 10.2 % | 4 of 12 |

### sigma2 = 0.025

| Filter | Rules (prediction / update) | Mean MSE | Median MSE | Lost | Mean LPE | NEES | MSE below the reference |
|---|---|---|---|---|---|---|---|
| L-AGSF | prop 0.9 / prop 0.9 (Table I) | 0.313 | 0.256 | 0 of 12 | -0.44 | 0.0 % | - |
| L-AGSF | prop 0.9 / auto 0.2 | 0.298 | 0.256 | 0 of 12 | -0.45 | 0.0 % | 6 of 12 |
| L-AGSF | prop 0.9 / auto 0.4 | 0.306 | 0.254 | 0 of 12 | -0.44 | 0.0 % | 5 of 12 |
| L-AGSF | prop 0.9 / auto 0.8 | 0.304 | 0.245 | 0 of 12 | -0.45 | 0.0 % | 5 of 12 |
| L-AGSF | auto 0.4 / auto 0.4 | 0.297 | 0.234 | 0 of 12 | -0.66 | 0.0 % | 11 of 12 |
| U-AGSF | prop 0.9 / prop 0.9 (Table I) | 0.220 | 0.211 | 0 of 12 | -1.93 | 0.0 % | - |
| U-AGSF | prop 0.9 / auto 0.2 | 0.218 | 0.222 | 0 of 12 | -1.92 | 0.0 % | 5 of 12 |
| U-AGSF | prop 0.9 / auto 0.4 | 0.199 | 0.208 | 0 of 12 | -1.95 | 0.0 % | 7 of 12 |
| U-AGSF | prop 0.9 / auto 0.8 | 0.211 | 0.230 | 0 of 12 | -1.93 | 0.0 % | 6 of 12 |
| U-AGSF | auto 0.4 / auto 0.4 | 0.188 | 0.185 | 0 of 12 | -2.20 | 0.0 % | 12 of 12 |

### sigma2 = 2.5e-05

| Filter | Rules (prediction / update) | Mean MSE | Median MSE | Lost | Mean LPE | NEES | MSE below the reference |
|---|---|---|---|---|---|---|---|
| L-AGSF | prop 0.9 / prop 0.9 (Table I) | 0.012 | 0.012 | 0 of 12 | -3.52 | 0.0 % | - |
| L-AGSF | prop 0.9 / auto 0.2 | 0.012 | 0.012 | 0 of 12 | -3.53 | 0.0 % | 5 of 12 |
| L-AGSF | prop 0.9 / auto 0.4 | 0.011 | 0.012 | 0 of 12 | -3.53 | 0.0 % | 8 of 12 |
| L-AGSF | prop 0.9 / auto 0.8 | 0.011 | 0.012 | 0 of 12 | -3.53 | 0.0 % | 7 of 12 |
| L-AGSF | auto 0.4 / auto 0.4 | 0.010 | 0.011 | 0 of 12 | -3.78 | 0.0 % | 9 of 12 |
| U-AGSF | prop 0.9 / prop 0.9 (Table I) | 0.012 | 0.013 | 0 of 12 | -7.32 | 0.0 % | - |
| U-AGSF | prop 0.9 / auto 0.2 | 0.011 | 0.013 | 0 of 12 | -7.27 | 0.0 % | 10 of 12 |
| U-AGSF | prop 0.9 / auto 0.4 | 0.012 | 0.013 | 0 of 12 | -7.26 | 0.0 % | 9 of 12 |
| U-AGSF | prop 0.9 / auto 0.8 | 0.012 | 0.013 | 0 of 12 | -7.25 | 0.0 % | 8 of 12 |
| U-AGSF | auto 0.4 / auto 0.4 | 0.010 | 0.012 | 0 of 12 | -7.86 | 0.0 % | 11 of 12 |

