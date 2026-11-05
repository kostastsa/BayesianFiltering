---
title: Scatter panels of the eight filters
subtitle: a = 0.05, σ² = 2.5×10⁻⁵ (low noise). Exp. A (mtt, wrapped bearing), table runs, task A12
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
| **a = 0.05** | [high noise](scatter_a0.05_s2.5.md) | [mid noise](scatter_a0.05_s0.025.md) | **● low noise** |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.05, σ² = 2.5×10⁻⁵

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 1.00e-4 (trial 92) | 9.00e-5 (trial 100) | 0.00157 (trial 28) | 0.0011 (trial 5) | 1.30e-4 (trial 51) | 1.00e-4 (trial 42) | 1.70e-4 (trial 51) | 2.00e-4 (trial 34) |
| median | 2.50e-4 (trial 5) | 2.40e-4 (trial 9) | 0.00348 (trial 82) | 0.00315 (trial 54) | 6.9 (trial 53) | 9.96 (trial 6) | 4.5 (trial 42) | 2.89 (trial 37) |
| worst | 0.00207 (trial 99) | 0.00256 (trial 35) | 0.0191 (trial 43) | 0.0148 (trial 27) | **656.6 (trial 82)** ⚠ | **2425 (trial 34)** ⚠ | **587.3 (trial 95)** ⚠ | **360.7 (trial 66)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -16.4 (trial 78, MSE 1.50e-4) | -17 (trial 13, MSE 1.50e-4) | -4.28 (trial 19, MSE 0.00189) | -8.58 (trial 38, MSE 0.00132) | -15.1 (trial 51, MSE 1.30e-4) | -6.27 (trial 42, MSE 1.00e-4) | -17.6 (trial 51, MSE 1.70e-4) | -14.4 (trial 53, MSE 2.20e-4) |
| median | -15.2 (trial 68, MSE 1.90e-4) | -15.3 (trial 85, MSE 1.30e-4) | -4.19 (trial 73, MSE 0.00304) | -7.7 (trial 69, MSE 0.0041) | 3.45e6 (trial 53, MSE 6.9) | 4.97e6 (trial 6, MSE 9.96) | 2.24e6 (trial 42, MSE 4.5) | 1.44e6 (trial 37, MSE 2.89) |
| worst | 7.43 (trial 21, MSE 3.90e-4) | -12.4 (trial 35, MSE 0.00256) | -3.73 (trial 43, MSE 0.0191) | -6.52 (trial 40, MSE 0.00426) | **3.28e8 (trial 82, MSE 656.6)** ⚠ | **1.21e9 (trial 34, MSE 2425)** ⚠ | **2.93e8 (trial 95, MSE 587.3)** ⚠ | **1.80e8 (trial 66, MSE 360.7)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.05_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 1.00e-4 (trial 92) | 9.00e-5 (trial 100) | 0.00157 (trial 28) | 0.0011 (trial 5) | 1.30e-4 (trial 51) | 1.00e-4 (trial 42) | 1.70e-4 (trial 51) | 2.00e-4 (trial 34) |
| median | 2.50e-4 (trial 5) | 2.40e-4 (trial 9) | 0.00348 (trial 82) | 0.00315 (trial 54) | 6.9 (trial 53) | 9.96 (trial 6) | 4.5 (trial 42) | 2.89 (trial 37) |
| worst | 0.00207 (trial 99) | 0.00256 (trial 35) | 0.0191 (trial 43) | 0.0148 (trial 27) | **656.6 (trial 82)** ⚠ | **2425 (trial 34)** ⚠ | **587.3 (trial 95)** ⚠ | **360.7 (trial 66)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.05_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | -16.4 (trial 78, MSE 1.50e-4) | -17 (trial 13, MSE 1.50e-4) | -4.28 (trial 19, MSE 0.00189) | -8.58 (trial 38, MSE 0.00132) | -15.1 (trial 51, MSE 1.30e-4) | -6.27 (trial 42, MSE 1.00e-4) | -17.6 (trial 51, MSE 1.70e-4) | -14.4 (trial 53, MSE 2.20e-4) |
| median | -15.2 (trial 68, MSE 1.90e-4) | -15.3 (trial 85, MSE 1.30e-4) | -4.19 (trial 73, MSE 0.00304) | -7.7 (trial 69, MSE 0.0041) | 3.45e6 (trial 53, MSE 6.9) | 4.97e6 (trial 6, MSE 9.96) | 2.24e6 (trial 42, MSE 4.5) | 1.44e6 (trial 37, MSE 2.89) |
| worst | 7.43 (trial 21, MSE 3.90e-4) | -12.4 (trial 35, MSE 0.00256) | -3.73 (trial 43, MSE 0.0191) | -6.52 (trial 40, MSE 0.00426) | **3.28e8 (trial 82, MSE 656.6)** ⚠ | **1.21e9 (trial 34, MSE 2425)** ⚠ | **2.93e8 (trial 95, MSE 587.3)** ⚠ | **1.80e8 (trial 66, MSE 360.7)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
