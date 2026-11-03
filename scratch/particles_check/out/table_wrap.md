Exp. A (mtt), a = 0.5, M = 1000, 30 data sets for each setting and filter. Ours: main.py runs. particles: run_particles.py. MSE: util.numerics.mse. Lost track: MSE > 10.

| sigma2 | Filter | Runs per data set: ours / particles | Median MSE: ours | Median MSE: particles | Lost tracks: ours | Lost tracks: particles | p (lost) | Data sets: ours < particles | p (sign) |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | BPF | 20 / 50 | 1.6987 | 1.8928 | 15.0% | 15.1% | 0.97 | 11/30 | 0.20 |
| 2.5 | APF | 20 / 50 | 2.4579 | 2.4067 | 28.3% | 18.7% | 0.00 | 11/30 | 0.20 |
| 0.025 | BPF | 20 / 50 | 0.0314 | 0.0387 | 6.7% | 10.3% | 0.00 | 14/30 | 0.86 |
| 0.025 | APF | 20 / 50 | 0.0395 | 0.0409 | 16.7% | 10.4% | 0.00 | 11/30 | 0.20 |
| 2.5e-05 | BPF | 20 / 50 | 0.0577 | 0.0626 | 26.7% | 28.7% | 0.17 | 11/30 | 0.20 |
| 2.5e-05 | APF | 20 / 50 | 0.0105 | 0.0117 | 20.0% | 23.9% | 0.00 | 12/30 | 0.36 |
