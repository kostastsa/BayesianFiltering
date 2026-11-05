---
title: Scatter panels of the eight filters
subtitle: a = 0.05, σ² = 2.5 (high noise). Exp. A (mtt, wrapped bearing), table runs, task A12
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
| **a = 0.05** | **● high noise** | [mid noise](scatter_a0.05_s0.025.md) | [low noise](scatter_a0.05_s2.5e-05.md) |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.05, σ² = 2.5

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.489 (trial 82) | 0.205 (trial 38) | 2.35 (trial 8) | 1.86 (trial 17) | 0.0817 (trial 78) | 0.0724 (trial 31) | 0.108 (trial 87) | 0.0981 (trial 50) |
| median | 4.09 (trial 75) | 1.75 (trial 11) | **34.6 (trial 73)** ⚠ | **30.9 (trial 89)** ⚠ | 1.62 (trial 87) | 1.71 (trial 75) | 1.63 (trial 49) | 1.5 (trial 65) |
| worst | **62.6 (trial 85)** ⚠ | **114.9 (trial 24)** ⚠ | **349.2 (trial 89)** ⚠ | **755.1 (trial 53)** ⚠ | **59.5 (trial 96)** ⚠ | **541.0 (trial 45)** ⚠ | **29.6 (trial 12)** ⚠ | **812.3 (trial 99)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -4.42 (trial 70, MSE 0.788) | -4.64 (trial 38, MSE 0.205) | 3.17 (trial 8, MSE 2.35) | 1.95 (trial 17, MSE 1.86) | 18.9 (trial 54, MSE 0.792) | 31.9 (trial 61, MSE 1.33) | **28.9 (trial 40, MSE 14.1)** ⚠ | 21.5 (trial 55, MSE 7.09) |
| median | 41.3 (trial 35, MSE 3.44) | 0.483 (trial 3, MSE 0.668) | **22.5 (trial 84, MSE 36.9)** ⚠ | **26.6 (trial 91, MSE 49.7)** ⚠ | 166.1 (trial 15, MSE 3.76) | 181.7 (trial 38, MSE 0.714) | 52 (trial 2, MSE 0.367) | **219.6 (trial 16, MSE 10)** ⚠ |
| worst | 539.3 (trial 27, MSE 2.18) | **81.7 (trial 24, MSE 114.9)** ⚠ | **363.3 (trial 89, MSE 349.2)** ⚠ | **1200 (trial 53, MSE 755.1)** ⚠ | **4.77e6 (trial 96, MSE 59.5)** ⚠ | **5.34e7 (trial 45, MSE 541.0)** ⚠ | **1429 (trial 12, MSE 29.6)** ⚠ | **1.33e7 (trial 36, MSE 324.7)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.489 (trial 82) | 0.205 (trial 38) | 2.35 (trial 8) | 1.86 (trial 17) | 0.0817 (trial 78) | 0.0724 (trial 31) | 0.108 (trial 87) | 0.0981 (trial 50) |
| median | 4.09 (trial 75) | 1.75 (trial 11) | **34.6 (trial 73)** ⚠ | **30.9 (trial 89)** ⚠ | 1.62 (trial 87) | 1.71 (trial 75) | 1.63 (trial 49) | 1.5 (trial 65) |
| worst | **62.6 (trial 85)** ⚠ | **114.9 (trial 24)** ⚠ | **349.2 (trial 89)** ⚠ | **755.1 (trial 53)** ⚠ | **59.5 (trial 96)** ⚠ | **541.0 (trial 45)** ⚠ | **29.6 (trial 12)** ⚠ | **812.3 (trial 99)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s2.5.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -4.42 (trial 70, MSE 0.788) | -4.64 (trial 38, MSE 0.205) | 3.17 (trial 8, MSE 2.35) | 1.95 (trial 17, MSE 1.86) | 18.9 (trial 54, MSE 0.792) | 31.9 (trial 61, MSE 1.33) | **28.9 (trial 40, MSE 14.1)** ⚠ | 21.5 (trial 55, MSE 7.09) |
| median | 41.3 (trial 35, MSE 3.44) | 0.483 (trial 3, MSE 0.668) | **22.5 (trial 84, MSE 36.9)** ⚠ | **26.6 (trial 91, MSE 49.7)** ⚠ | 166.1 (trial 15, MSE 3.76) | 181.7 (trial 38, MSE 0.714) | 52 (trial 2, MSE 0.367) | **219.6 (trial 16, MSE 10)** ⚠ |
| worst | 539.3 (trial 27, MSE 2.18) | **81.7 (trial 24, MSE 114.9)** ⚠ | **363.3 (trial 89, MSE 349.2)** ⚠ | **1200 (trial 53, MSE 755.1)** ⚠ | **4.77e6 (trial 96, MSE 59.5)** ⚠ | **5.34e7 (trial 45, MSE 541.0)** ⚠ | **1429 (trial 12, MSE 29.6)** ⚠ | **1.33e7 (trial 36, MSE 324.7)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
