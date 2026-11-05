---
title: Scatter panels of the eight filters
subtitle: a = 0.5, σ² = 2.5×10⁻² (mid noise). Exp. A (mtt, wrapped bearing), table runs, task A12
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
| **a = 0.5** | [high noise](scatter_a0.5_s2.5.md) | **● mid noise** | [low noise](scatter_a0.5_s2.5e-05.md) |
| **a = 0.05** | [high noise](scatter_a0.05_s2.5.md) | [mid noise](scatter_a0.05_s0.025.md) | [low noise](scatter_a0.05_s2.5e-05.md) |

**Setting.** Click a cell to open the page of that setting.
:::

### a = 0.5, σ² = 2.5×10⁻²

::::: tabs
:::: {.tab title="Both"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.0796 (trial 81) | 0.00137 (trial 30) | 0.103 (trial 53) | 0.0374 (trial 100) | 0.00427 (trial 7) | 0.0039 (trial 36) | 0.00407 (trial 18) | 0.0013 (trial 57) |
| median | 0.17 (trial 62) | 0.0304 (trial 59) | 0.289 (trial 3) | 0.19 (trial 12) | 0.0384 (trial 93) | 0.0433 (trial 90) | 0.0333 (trial 7) | 0.0832 (trial 62) |
| worst | 8.63 (trial 96) | 0.299 (trial 21) | 1.28 (trial 7) | 1.67 (trial 92) | **200.4 (trial 23)** ⚠ | **55.8 (trial 25)** ⚠ | 0.326 (trial 62) | **198.5 (trial 55)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 496.9 (trial 79, MSE 0.168) | -11.2 (trial 30, MSE 0.00137) | -1.23 (trial 53, MSE 0.103) | -2.89 (trial 46, MSE 0.044) | 19.8 (trial 5, MSE 0.0125) | 15 (trial 82, MSE 0.0368) | 18 (trial 18, MSE 0.00407) | 15.6 (trial 56, MSE 0.00974) |
| median | 2000 (trial 16, MSE 0.281) | 1.19 (trial 62, MSE 0.0232) | -0.365 (trial 23, MSE 0.213) | -1.87 (trial 3, MSE 0.226) | 706.1 (trial 75, MSE 0.0207) | 843.0 (trial 67, MSE 0.0135) | 34.7 (trial 41, MSE 0.0145) | 2936 (trial 30, MSE 0.0678) |
| worst | 32894 (trial 96, MSE 8.63) | 106.8 (trial 63, MSE 0.102) | 1.15 (trial 7, MSE 1.28) | 2.53 (trial 92, MSE 1.67) | **9.96e7 (trial 23, MSE 200.4)** ⚠ | **2.69e7 (trial 37, MSE 54.9)** ⚠ | 804.9 (trial 100, MSE 0.0382) | **9.63e7 (trial 55, MSE 198.5)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="MSE"}

#### Ranked by MSE

The rows are the trials with the lowest, median and highest MSE of each filter (the squared error of the weighted mean).

::: {.figures .wide cols=1}
![Ranked by MSE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_a0.5_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 0.0796 (trial 81) | 0.00137 (trial 30) | 0.103 (trial 53) | 0.0374 (trial 100) | 0.00427 (trial 7) | 0.0039 (trial 36) | 0.00407 (trial 18) | 0.0013 (trial 57) |
| median | 0.17 (trial 62) | 0.0304 (trial 59) | 0.289 (trial 3) | 0.19 (trial 12) | 0.0384 (trial 93) | 0.0433 (trial 90) | 0.0333 (trial 7) | 0.0832 (trial 62) |
| worst | 8.63 (trial 96) | 0.299 (trial 21) | 1.28 (trial 7) | 1.67 (trial 92) | **200.4 (trial 23)** ⚠ | **55.8 (trial 25)** ⚠ | 0.326 (trial 62) | **198.5 (trial 55)** ⚠ |

**Ranked by MSE.** Each cell: the MSE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::: {.tab title="LPE"}

#### Ranked by LPE

The rows are the trials with the lowest, median and highest LPE of each filter (the negative log density of the true state under the posterior).

::: {.figures .wide cols=1}
![Ranked by LPE: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.](../page/figs/v1_lpe_a0.5_s0.025.jpg)
:::

::: {.table .wide}
| Run | L-GSF | U-GSF | L-AGSF | U-AGSF | BPF | APF | RPF | UPF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| best | 496.9 (trial 79, MSE 0.168) | -11.2 (trial 30, MSE 0.00137) | -1.23 (trial 53, MSE 0.103) | -2.89 (trial 46, MSE 0.044) | 19.8 (trial 5, MSE 0.0125) | 15 (trial 82, MSE 0.0368) | 18 (trial 18, MSE 0.00407) | 15.6 (trial 56, MSE 0.00974) |
| median | 2000 (trial 16, MSE 0.281) | 1.19 (trial 62, MSE 0.0232) | -0.365 (trial 23, MSE 0.213) | -1.87 (trial 3, MSE 0.226) | 706.1 (trial 75, MSE 0.0207) | 843.0 (trial 67, MSE 0.0135) | 34.7 (trial 41, MSE 0.0145) | 2936 (trial 30, MSE 0.0678) |
| worst | 32894 (trial 96, MSE 8.63) | 106.8 (trial 63, MSE 0.102) | 1.15 (trial 7, MSE 1.28) | 2.53 (trial 92, MSE 1.67) | **9.96e7 (trial 23, MSE 200.4)** ⚠ | **2.69e7 (trial 37, MSE 54.9)** ⚠ | 804.9 (trial 100, MSE 0.0382) | **9.63e7 (trial 55, MSE 198.5)** ⚠ |

**Ranked by LPE.** Each cell: the LPE of the run and its trial. ⚠: the MSE is above 10 (a lost track, the rule of Tables I and III).
:::

::::
:::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
