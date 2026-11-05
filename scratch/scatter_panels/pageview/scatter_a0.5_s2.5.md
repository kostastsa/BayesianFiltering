---
title: Scatter panels of the eight filters
subtitle: a = 0.5, σ² = 2.5 (high noise). Exp. A (mtt, wrapped bearing), table runs, task A12
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
| **a = 0.5** | **● high noise** | [mid noise](scatter_panels.md) | [low noise](scatter_a0.5_s2.5e-05.md) |
| **a = 0.05** | [high noise](scatter_a0.05_s2.5.md) | [mid noise](scatter_a0.05_s0.025.md) | [low noise](scatter_a0.05_s2.5e-05.md) |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.5, σ² = 2.5

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 1.31 (trial 93) | 0.294 (trial 14) | 2.22 (trial 90) | 1.19 (trial 44) | 0.137 (trial 82) | 0.0608 (trial 59) | 0.0585 (trial 79) | 0.164 (trial 59) |
| median | **65.7 (trial 59)** ⚠ | 2.51 (trial 68) | **20 (trial 51)** ⚠ | 9.99 (trial 10) | 1.92 (trial 47) | 2.12 (trial 20) | 1.47 (trial 33) | 1.69 (trial 8) |
| worst | **1342 (trial 90)** ⚠ | **89 (trial 76)** ⚠ | **431.4 (trial 13)** ⚠ | **835.0 (trial 61)** ⚠ | **64.4 (trial 30)** ⚠ | **691.6 (trial 73)** ⚠ | **158.4 (trial 13)** ⚠ | **295.3 (trial 52)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | **2093 (trial 9, MSE 291.9)** ⚠ | -4.19 (trial 14, MSE 0.294) | 5.63 (trial 90, MSE 2.22) | 2.52 (trial 60, MSE 1.45) | 35.9 (trial 36, MSE 0.727) | 31.4 (trial 32, MSE 0.178) | 29.8 (trial 87, MSE 0.191) | 31.4 (trial 18, MSE 0.187) |
| median | **10850 (trial 21, MSE 146.2)** ⚠ | 2.84 (trial 88, MSE 4.14) | **19.4 (trial 16, MSE 15.6)** ⚠ | 13.1 (trial 68, MSE 9.28) | 212.5 (trial 45, MSE 7.32) | 184.6 (trial 7, MSE 3.47) | 60.2 (trial 69, MSE 0.189) | 249.9 (trial 9, MSE 2.19) |
| worst | **62664 (trial 40, MSE 319.4)** ⚠ | **99.2 (trial 45, MSE 40.8)** ⚠ | **360.6 (trial 13, MSE 431.4)** ⚠ | **2041 (trial 61, MSE 835.0)** ⚠ | **5.84e6 (trial 46, MSE 56.3)** ⚠ | **9.15e7 (trial 73, MSE 691.6)** ⚠ | **2898 (trial 12, MSE 33.1)** ⚠ | **3.34e7 (trial 73, MSE 99.3)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 1.31 (trial 93) | 0.294 (trial 14) | 2.22 (trial 90) | 1.19 (trial 44) | 0.137 (trial 82) | 0.0608 (trial 59) | 0.0585 (trial 79) | 0.164 (trial 59) |
| median | **65.7 (trial 59)** ⚠ | 2.51 (trial 68) | **20 (trial 51)** ⚠ | 9.99 (trial 10) | 1.92 (trial 47) | 2.12 (trial 20) | 1.47 (trial 33) | 1.69 (trial 8) |
| worst | **1342 (trial 90)** ⚠ | **89 (trial 76)** ⚠ | **431.4 (trial 13)** ⚠ | **835.0 (trial 61)** ⚠ | **64.4 (trial 30)** ⚠ | **691.6 (trial 73)** ⚠ | **158.4 (trial 13)** ⚠ | **295.3 (trial 52)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | **2093 (trial 9, MSE 291.9)** ⚠ | -4.19 (trial 14, MSE 0.294) | 5.63 (trial 90, MSE 2.22) | 2.52 (trial 60, MSE 1.45) | 35.9 (trial 36, MSE 0.727) | 31.4 (trial 32, MSE 0.178) | 29.8 (trial 87, MSE 0.191) | 31.4 (trial 18, MSE 0.187) |
| median | **10850 (trial 21, MSE 146.2)** ⚠ | 2.84 (trial 88, MSE 4.14) | **19.4 (trial 16, MSE 15.6)** ⚠ | 13.1 (trial 68, MSE 9.28) | 212.5 (trial 45, MSE 7.32) | 184.6 (trial 7, MSE 3.47) | 60.2 (trial 69, MSE 0.189) | 249.9 (trial 9, MSE 2.19) |
| worst | **62664 (trial 40, MSE 319.4)** ⚠ | **99.2 (trial 45, MSE 40.8)** ⚠ | **360.6 (trial 13, MSE 431.4)** ⚠ | **2041 (trial 61, MSE 835.0)** ⚠ | **5.84e6 (trial 46, MSE 56.3)** ⚠ | **9.15e7 (trial 73, MSE 691.6)** ⚠ | **2898 (trial 12, MSE 33.1)** ⚠ | **3.34e7 (trial 73, MSE 99.3)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
