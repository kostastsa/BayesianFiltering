Exp. A (mtt), a = 0.5, M = 1000, 30 data sets for each setting and filter. Ours: main.py runs. particles: run_particles.py. MSE: util.numerics.mse. Lost track: MSE > 10.

| sigma2 | Filter | Runs per data set: ours / particles | Median MSE: ours | Median MSE: particles | Lost tracks: ours | Lost tracks: particles | p (lost) | Data sets: ours < particles | p (sign) |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | BPF | 20 / 50 | 1.9898 | 1.8928 | 15.0% | 15.1% | 0.97 | 15/30 | 1.00 |
| 2.5 | APF | 20 / 50 | 2.4213 | 2.4067 | 19.2% | 18.7% | 0.79 | 14/30 | 0.86 |
| 0.025 | BPF | 20 / 50 | 0.0439 | 0.0387 | 11.2% | 10.3% | 0.49 | 16/30 | 0.86 |
| 0.025 | APF | 20 / 50 | 0.0390 | 0.0409 | 9.5% | 10.4% | 0.50 | 19/30 | 0.20 |
| 2.5e-05 | BPF | 20 / 50 | 0.0473 | 0.0626 | 27.7% | 28.7% | 0.48 | 16/30 | 0.86 |
| 2.5e-05 | APF | 20 / 50 | 0.0125 | 0.0117 | 23.5% | 23.9% | 0.85 | 10/30 | 0.10 |
