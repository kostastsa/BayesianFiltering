---
title: Scatter panels of the eight filters
subtitle: a = 0.05, σ² = 2.5×10⁻² (mid noise). Exp. A (mtt, wrapped bearing), table runs, task A12
date: 2026-11-05
tags: [6 settings, 8 filters, best / median / worst, 144 exact repeats per ranking]
view: page
toc: false
width: wide
---

For each setting of Tables I and III: the true position track (red), the component means (gold) and posterior samples or particles (black), for the best, median and worst table run of each filter, ranked by MSE or by LPE. Columns: L-GSF and U-GSF with M = 100, L-AGSF and U-AGSF with M = 10 (N = L = 5), and the BPF, APF, RPF and UPF with 1000 particles. Each filter has its own 100 trials, so each cell shows its own trajectory.

::: {.table}
| | σ² = 2.5 | σ² = 2.5×10⁻² | σ² = 2.5×10⁻⁵ |
|---|:-:|:-:|:-:|
| **a = 0.5** | [high noise](scatter_a0.5_s2.5.md) | [mid noise](scatter_panels.md) | [low noise](scatter_a0.5_s2.5e-05.md) |
| **a = 0.05** | [high noise](scatter_a0.05_s2.5.md) | **● mid noise** | [low noise](scatter_a0.05_s2.5e-05.md) |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.05, σ² = 2.5×10⁻²

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.0118 (trial 94) | 0.003 (trial 30) | 0.0627 (trial 41) | 0.0362 (trial 43) | 0.00206 (trial 88) | 0.00203 (trial 11) | 0.00342 (trial 18) | 0.00412 (trial 57) |
| median | 0.031 (trial 21) | 0.0262 (trial 100) | 0.422 (trial 77) | 0.376 (trial 25) | 0.0224 (trial 95) | 0.0262 (trial 83) | 0.0253 (trial 82) | 0.028 (trial 13) |
| worst | 0.304 (trial 97) | 0.28 (trial 91) | 1.87 (trial 21) | 2.84 (trial 92) | **1101 (trial 63)** ⚠ | **611.8 (trial 68)** ⚠ | 0.291 (trial 62) | **879.8 (trial 23)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -9.41 (trial 58, MSE 0.0323) | -10.1 (trial 55, MSE 0.00372) | -2.3 (trial 53, MSE 0.0643) | -3.37 (trial 43, MSE 0.0362) | 8.95 (trial 59, MSE 0.00978) | 12.6 (trial 51, MSE 0.012) | -2.48 (trial 45, MSE 0.0492) | 7.98 (trial 92, MSE 0.00678) |
| median | -4.62 (trial 24, MSE 0.101) | -7.4 (trial 16, MSE 0.052) | -1.29 (trial 93, MSE 0.367) | -2.06 (trial 7, MSE 0.254) | 388.8 (trial 38, MSE 0.0249) | 382.6 (trial 72, MSE 0.0179) | 8.42 (trial 30, MSE 0.0218) | 364.4 (trial 99, MSE 0.0578) |
| worst | 233.0 (trial 54, MSE 0.0476) | 0.0014 (trial 69, MSE 0.0866) | -0.335 (trial 21, MSE 1.87) | -0.343 (trial 92, MSE 2.84) | **5.33e8 (trial 63, MSE 1101)** ⚠ | **2.48e8 (trial 68, MSE 611.8)** ⚠ | 113.6 (trial 8, MSE 0.0735) | **4.21e8 (trial 23, MSE 879.8)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.0118 (trial 94) | 0.003 (trial 30) | 0.0627 (trial 41) | 0.0362 (trial 43) | 0.00206 (trial 88) | 0.00203 (trial 11) | 0.00342 (trial 18) | 0.00412 (trial 57) |
| median | 0.031 (trial 21) | 0.0262 (trial 100) | 0.422 (trial 77) | 0.376 (trial 25) | 0.0224 (trial 95) | 0.0262 (trial 83) | 0.0253 (trial 82) | 0.028 (trial 13) |
| worst | 0.304 (trial 97) | 0.28 (trial 91) | 1.87 (trial 21) | 2.84 (trial 92) | **1101 (trial 63)** ⚠ | **611.8 (trial 68)** ⚠ | 0.291 (trial 62) | **879.8 (trial 23)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -9.41 (trial 58, MSE 0.0323) | -10.1 (trial 55, MSE 0.00372) | -2.3 (trial 53, MSE 0.0643) | -3.37 (trial 43, MSE 0.0362) | 8.95 (trial 59, MSE 0.00978) | 12.6 (trial 51, MSE 0.012) | -2.48 (trial 45, MSE 0.0492) | 7.98 (trial 92, MSE 0.00678) |
| median | -4.62 (trial 24, MSE 0.101) | -7.4 (trial 16, MSE 0.052) | -1.29 (trial 93, MSE 0.367) | -2.06 (trial 7, MSE 0.254) | 388.8 (trial 38, MSE 0.0249) | 382.6 (trial 72, MSE 0.0179) | 8.42 (trial 30, MSE 0.0218) | 364.4 (trial 99, MSE 0.0578) |
| worst | 233.0 (trial 54, MSE 0.0476) | 0.0014 (trial 69, MSE 0.0866) | -0.335 (trial 21, MSE 1.87) | -0.343 (trial 92, MSE 2.84) | **5.33e8 (trial 63, MSE 1101)** ⚠ | **2.48e8 (trial 68, MSE 611.8)** ⚠ | 113.6 (trial 8, MSE 0.0735) | **4.21e8 (trial 23, MSE 879.8)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
