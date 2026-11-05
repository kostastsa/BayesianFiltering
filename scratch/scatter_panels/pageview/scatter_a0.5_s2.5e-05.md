---
title: Scatter panels of the eight filters
subtitle: a = 0.5, σ² = 2.5×10⁻⁵ (low noise). Exp. A (mtt, wrapped bearing), table runs, task A12
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
| **a = 0.5** | [high noise](scatter_a0.5_s2.5.md) | [mid noise](scatter_panels.md) | **● low noise** |
| **a = 0.05** | [high noise](scatter_a0.05_s2.5.md) | [mid noise](scatter_a0.05_s0.025.md) | [low noise](scatter_a0.05_s2.5e-05.md) |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.5, σ² = 2.5×10⁻⁵

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.00336 (trial 50) | 3.40e-4 (trial 61) | 0.0098 (trial 36) | 0.00126 (trial 42) | 3.90e-4 (trial 85) | 5.90e-4 (trial 84) | 2.90e-4 (trial 97) | 4.00e-4 (trial 58) |
| median | 0.0174 (trial 79) | 0.00448 (trial 11) | 0.0234 (trial 63) | 0.0122 (trial 21) | 0.0261 (trial 72) | 0.0135 (trial 74) | 0.00335 (trial 53) | 0.0114 (trial 53) |
| worst | 0.0378 (trial 84) | 0.0168 (trial 18) | 0.0459 (trial 11) | 0.0219 (trial 100) | **1398 (trial 100)** ⚠ | **1839 (trial 34)** ⚠ | **1341 (trial 20)** ⚠ | **103.1 (trial 32)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 168.0 (trial 30, MSE 0.00347) | -16.3 (trial 61, MSE 3.40e-4) | -3.75 (trial 85, MSE 0.0208) | -8.49 (trial 13, MSE 0.00197) | -14 (trial 51, MSE 5.50e-4) | -12 (trial 85, MSE 0.00112) | -15.4 (trial 97, MSE 2.90e-4) | -4.18 (trial 58, MSE 4.00e-4) |
| median | 1886 (trial 5, MSE 0.0143) | 9.14 (trial 24, MSE 0.0045) | -3.43 (trial 48, MSE 0.0266) | -7.39 (trial 77, MSE 0.00762) | 10862 (trial 72, MSE 0.0261) | 6186 (trial 74, MSE 0.0135) | 1170 (trial 5, MSE 0.00475) | 5245 (trial 45, MSE 0.0115) |
| worst | 4803 (trial 13, MSE 0.0315) | 733.7 (trial 18, MSE 0.0168) | -3.06 (trial 88, MSE 0.0406) | -5.55 (trial 100, MSE 0.0219) | **6.98e8 (trial 100, MSE 1398)** ⚠ | **9.19e8 (trial 34, MSE 1839)** ⚠ | **6.70e8 (trial 20, MSE 1341)** ⚠ | **5.15e7 (trial 32, MSE 103.1)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.00336 (trial 50) | 3.40e-4 (trial 61) | 0.0098 (trial 36) | 0.00126 (trial 42) | 3.90e-4 (trial 85) | 5.90e-4 (trial 84) | 2.90e-4 (trial 97) | 4.00e-4 (trial 58) |
| median | 0.0174 (trial 79) | 0.00448 (trial 11) | 0.0234 (trial 63) | 0.0122 (trial 21) | 0.0261 (trial 72) | 0.0135 (trial 74) | 0.00335 (trial 53) | 0.0114 (trial 53) |
| worst | 0.0378 (trial 84) | 0.0168 (trial 18) | 0.0459 (trial 11) | 0.0219 (trial 100) | **1398 (trial 100)** ⚠ | **1839 (trial 34)** ⚠ | **1341 (trial 20)** ⚠ | **103.1 (trial 32)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s2.5e-05.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 168.0 (trial 30, MSE 0.00347) | -16.3 (trial 61, MSE 3.40e-4) | -3.75 (trial 85, MSE 0.0208) | -8.49 (trial 13, MSE 0.00197) | -14 (trial 51, MSE 5.50e-4) | -12 (trial 85, MSE 0.00112) | -15.4 (trial 97, MSE 2.90e-4) | -4.18 (trial 58, MSE 4.00e-4) |
| median | 1886 (trial 5, MSE 0.0143) | 9.14 (trial 24, MSE 0.0045) | -3.43 (trial 48, MSE 0.0266) | -7.39 (trial 77, MSE 0.00762) | 10862 (trial 72, MSE 0.0261) | 6186 (trial 74, MSE 0.0135) | 1170 (trial 5, MSE 0.00475) | 5245 (trial 45, MSE 0.0115) |
| worst | 4803 (trial 13, MSE 0.0315) | 733.7 (trial 18, MSE 0.0168) | -3.06 (trial 88, MSE 0.0406) | -5.55 (trial 100, MSE 0.0219) | **6.98e8 (trial 100, MSE 1398)** ⚠ | **9.19e8 (trial 34, MSE 1839)** ⚠ | **6.70e8 (trial 20, MSE 1341)** ⚠ | **5.15e7 (trial 32, MSE 103.1)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
