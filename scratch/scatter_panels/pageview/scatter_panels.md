---
title: Scatter panels of the eight filters
subtitle: Exp. A (mtt, wrapped bearing), table runs, task A12
date: 2026-11-05
tags: [6 settings, 8 filters, best / median / worst, 144 exact repeats per ranking]
view: page
toc: false
width: wide
---

For each setting of Tables I and III: the true position track (red), the component means (gold) and posterior samples or particles (black), for the best, median and worst table run of each filter, ranked by MSE or by LPE. Columns: L-GSF and U-GSF with M = 100, L-AGSF and U-AGSF with M = 10 (N = L = 5), and the BPF, APF, RPF and UPF with 1000 particles. Each filter has its own 100 trials, so each cell shows its own trajectory.

::::::: tabs
:::::: {.tab title="a = 0.5, σ² = 2.5 (high noise)"}

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
::::::
:::::: {.tab title="a = 0.5, σ² = 2.5×10⁻² (mid noise)"}

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
::::::
:::::: {.tab title="a = 0.5, σ² = 2.5×10⁻⁵ (low noise)"}

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
::::::
:::::: {.tab title="a = 0.05, σ² = 2.5 (high noise)"}

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
::::::
:::::: {.tab title="a = 0.05, σ² = 2.5×10⁻² (mid noise)"}

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
::::::
:::::: {.tab title="a = 0.05, σ² = 2.5×10⁻⁵ (low noise)"}

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
::::::
:::::::

::: {.callout .note title="About the runs"}
The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). Source: `scratch/scatter_panels/`.
:::
