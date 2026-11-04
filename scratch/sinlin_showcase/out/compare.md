sinlin showcase: T = 100, kappa = 1.0, omega = 2.0, L-AGSF M = 10, N = L = 5. Trials 1 to 100, the same data for all filters (data block). RMSE: square root of the mean over trials. "sdp below auto": trials in which sdp has the lower mse_x1.

### d = 5, a = e_1

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 0.966 | 0.246 | 15.72 | 22.6 % | - | - | 0.5 |
| UKF (U-GSF, 1 comp.) | 1.301 | 0.246 | 2.554 | 2.3 % | - | - | 1.0 |
| L-AGSF prop 0.9 | 0.745 | 0.247 | 4.065 | 9.8 % | 0.900 | 0.900 | 1.6 |
| L-AGSF auto 2/N (Eq. 47) | 0.782 | 0.247 | 4.637 | 9.7 % | 0.888 | 0.888 | 1.9 |
| L-AGSF sdp 2/N (program 45) | 0.769 | 0.247 | 4.266 | 7.2 % | 0.631 | 0.950 | 3.7 |
| L-AGSF prop 0.001 (Delta near 0) | 1.179 | 0.279 | 57.68 | 36.4 % | 0.001 | 0.001 | 1.5 |
| U-AGSF prop 0.9 | 0.639 | 0.247 | 1.047 | 1.3 % | 0.900 | 0.900 | 2.1 |
| U-AGSF auto 2/N (Eq. 47) | 0.700 | 0.249 | 1.143 | 2.1 % | 0.763 | 0.763 | 2.3 |
| BPF, 1000 particles | 0.648 | 0.252 | 2058 | 5.9 % | - | - | 1.2 |

sdp below auto (mse_x1): 51 of 100

### d = 10, a = e_1

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 1.097 | 0.246 | 21.52 | 25.4 % | - | - | 0.5 |
| UKF (U-GSF, 1 comp.) | 0.719 | 0.246 | 1.2 | 0.2 % | - | - | 1.1 |
| L-AGSF prop 0.9 | 0.876 | 0.247 | 6.367 | 11.8 % | 0.900 | 0.900 | 2.1 |
| L-AGSF auto 2/N (Eq. 47) | 0.764 | 0.247 | 4.979 | 9.6 % | 0.936 | 0.936 | 2.5 |
| L-AGSF sdp 2/N (program 45) | 0.791 | 0.246 | 4.642 | 7.1 % | 0.643 | 0.950 | 8.0 |
| L-AGSF prop 0.001 (Delta near 0) | 1.244 | 0.339 | 105.1 | 90.8 % | 0.001 | 0.001 | 2.1 |
| U-AGSF prop 0.9 | 0.765 | 0.246 | 1.559 | 2.2 % | 0.900 | 0.900 | 2.9 |
| U-AGSF auto 2/N (Eq. 47) | 0.776 | 0.248 | 1.717 | 1.7 % | 0.853 | 0.853 | 3.1 |
| BPF, 1000 particles | 0.968 | 0.285 | 4.848e+04 | 55.4 % | - | - | 1.7 |

sdp below auto (mse_x1): 40 of 100

### d = 20, a = e_1

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 1.072 | 0.245 | 20.41 | 21.4 % | - | - | 0.5 |
| UKF (U-GSF, 1 comp.) | 0.757 | 0.245 | 1.348 | 1.0 % | - | - | 1.4 |
| L-AGSF prop 0.9 | 0.805 | 0.246 | 5.434 | 10.1 % | 0.900 | 0.900 | 3.5 |
| L-AGSF auto 2/N (Eq. 47) | 0.832 | 0.246 | 6.69 | 11.0 % | 0.948 | 0.948 | 3.9 |
| L-AGSF sdp 2/N (program 45) | 0.862 | 0.245 | 6.339 | 8.7 % | 0.649 | 0.950 | 31.0 |
| L-AGSF prop 0.001 (Delta near 0) | 1.125 | 0.451 | 232.7 | 100.0 % | 0.001 | 0.001 | 3.1 |
| U-AGSF prop 0.9 | 0.786 | 0.246 | 1.642 | 2.2 % | 0.900 | 0.900 | 4.9 |
| U-AGSF auto 2/N (Eq. 47) | 0.729 | 0.246 | 1.493 | 1.6 % | 0.907 | 0.907 | 5.3 |
| BPF, 1000 particles | 1.028 | 0.380 | 4.307e+05 | 99.9 % | - | - | 1.0 |

sdp below auto (mse_x1): 49 of 100

### d = 50, a = e_1

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 0.927 | 0.245 | 15.14 | 15.9 % | - | - | 0.4 |
| UKF (U-GSF, 1 comp.) | 0.708 | 0.246 | 1.703 | 0.2 % | - | - | 3.0 |
| L-AGSF prop 0.9 | 0.891 | 0.247 | 8.499 | 10.6 % | 0.900 | 0.900 | 5.3 |
| L-AGSF auto 2/N (Eq. 47) | 0.974 | 0.246 | 10.57 | 12.8 % | 0.950 | 0.950 | 6.6 |
| L-AGSF sdp 2/N (program 45) | 0.919 | 0.247 | 8.988 | 9.9 % | 0.657 | 0.950 | 177.9 |
| L-AGSF prop 0.001 (Delta near 0) | 1.107 | 0.614 | 929.9 | 100.0 % | 0.001 | 0.001 | 4.4 |
| U-AGSF prop 0.9 | 0.843 | 0.247 | 2.722 | 0.8 % | 0.900 | 0.900 | 21.7 |
| U-AGSF auto 2/N (Eq. 47) | 0.722 | 0.246 | 1.957 | 0.2 % | 0.947 | 0.947 | 23.2 |
| BPF, 1000 particles | 1.027 | 0.543 | 3.001e+06 | 100.0 % | - | - | 0.7 |

sdp below auto (mse_x1): 41 of 100

### d = 10, rotated direction a

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 1.366 | 0.248 | 14 | 18.5 % | - | - | 0.5 |
| UKF (U-GSF, 1 comp.) | 1.318 | 0.255 | 3.273 | 6.7 % | - | - | 1.1 |
| L-AGSF prop 0.9 | 0.873 | 0.246 | 3.457 | 5.2 % | 0.900 | 0.900 | 2.1 |
| L-AGSF auto 2/N (Eq. 47) | 1.014 | 0.246 | 5.047 | 7.6 % | 0.946 | 0.946 | 2.4 |
| L-AGSF sdp 2/N (program 45) | 0.654 | 0.245 | 2.257 | 3.0 % | 0.542 | 0.950 | 8.4 |
| L-AGSF prop 0.001 (Delta near 0) | 0.949 | 0.336 | 73.98 | 88.8 % | 0.001 | 0.001 | 2.0 |
| U-AGSF prop 0.9 | 0.576 | 0.245 | 0.8876 | 0.4 % | 0.900 | 0.900 | 2.9 |
| U-AGSF auto 2/N (Eq. 47) | 0.607 | 0.245 | 0.9841 | 0.4 % | 0.928 | 0.928 | 3.1 |
| BPF, 1000 particles | 0.702 | 0.281 | 3.174e+04 | 48.0 % | - | - | 1.6 |

sdp below auto (mse_x1): 65 of 100

### d = 20, rotated direction a

| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |
|---|---|---|---|---|---|---|---|
| EKF (L-GSF, 1 comp.) | 1.473 | 0.249 | 9.721 | 11.1 % | - | - | 0.5 |
| UKF (U-GSF, 1 comp.) | 0.747 | 0.245 | 1.355 | 0.4 % | - | - | 1.3 |
| L-AGSF prop 0.9 | 0.806 | 0.247 | 2.318 | 2.2 % | 0.900 | 0.900 | 3.5 |
| L-AGSF auto 2/N (Eq. 47) | 1.044 | 0.247 | 3.669 | 4.1 % | 0.949 | 0.949 | 3.8 |
| L-AGSF sdp 2/N (program 45) | 0.681 | 0.246 | 2.451 | 1.3 % | 0.505 | 0.950 | 32.5 |
| L-AGSF prop 0.001 (Delta near 0) | 0.983 | 0.447 | 214 | 100.0 % | 0.001 | 0.001 | 3.1 |
| U-AGSF prop 0.9 | 0.634 | 0.246 | 1.213 | 0.1 % | 0.900 | 0.900 | 4.9 |
| U-AGSF auto 2/N (Eq. 47) | 0.665 | 0.245 | 1.216 | 0.2 % | 0.949 | 0.949 | 5.3 |
| BPF, 1000 particles | 0.922 | 0.377 | 3.744e+05 | 99.9 % | - | - | 1.0 |

sdp below auto (mse_x1): 68 of 100

