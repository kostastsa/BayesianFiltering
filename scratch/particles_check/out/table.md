Exp. A (mtt), a = 0.5, M = 1000, 30 data sets for each setting and filter. Ours: main.py runs. particles: run_particles.py. MSE: util.numerics.mse. Lost track: MSE > 10.

| sigma2 | Filter | Runs per data set: ours / particles | Median MSE: ours | Median MSE: particles | Lost tracks: ours | Lost tracks: particles | p (lost) | Data sets: ours < particles | p (sign) |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | BPF | 20 / 50 | 0.9066 | 0.8888 | 10.7% | 10.3% | 0.87 | 15/30 | 1.00 |
| 2.5 | APF | 20 / 50 | 1.2551 | 1.1796 | 11.7% | 9.7% | 0.14 | 7/30 | 0.01 |
| 0.025 | BPF | 20 / 50 | 0.0383 | 0.0382 | 14.8% | 15.1% | 0.93 | 17/30 | 0.58 |
| 0.025 | APF | 20 / 50 | 0.0413 | 0.0476 | 16.0% | 16.3% | 0.88 | 20/30 | 0.10 |
| 2.5e-05 | BPF | 20 / 50 | 0.0791 | 0.0856 | 30.3% | 31.7% | 0.40 | 15/30 | 1.00 |
| 2.5e-05 | APF | 20 / 50 | 0.0119 | 0.0118 | 28.7% | 26.4% | 0.14 | 15/30 | 1.00 |
