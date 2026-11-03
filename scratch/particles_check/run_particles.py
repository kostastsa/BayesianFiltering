"""Step 2 (~/.venvs/particles): run the BPF and APF of the particles library (Chopin and Papaspiliopoulos, 2020) on the data of gt.npz.

Usage (repository root):  ~/.venvs/particles/bin/python scratch/particles_check/run_particles.py A SIGMA2   (as for export_gt.py)
The particles BPF runs on the data of the main.py BPF runs, the particles APF on the data of the main.py APF runs.
N_RUNS runs per data set and filter (main.py has one run per data set), M = 1000, multinomial resampling, ESSrmin = 0.5, as in the main.py runs.
The model is a numpy copy of simulators/mtt.py. The script checks it against the JAX values in gt.npz before it runs.
Time order: particles starts with an unobserved x_0 = the initial particles. x_k (k >= 1) uses the input u[k-1] and the observation
obs[k-1]. This is the order of filters/pf.py, which propagates the initial particles before it uses obs[0]. The estimate after step k
is compared with states[k-1], as in util.numerics.mse. The APF auxiliary function is eta_t(x_t) = g(y_{t+1} | f(x_t, q=0)), as in pf.py.
particles resamples when the ESS of the auxiliary weights is low. Otherwise it keeps the weights w and multiplies them by g, as pf.py.
Output: scratch/particles_check/out/particles_a<A>_sigma2_<SIGMA2>.npz with the filter means (N_TRIALS, N_RUNS, T, 4) and the log-likelihoods.
"""
import os
import sys
import time
import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssms
from particles import resampling as rs
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
N_RUNS = 50


def _log_mean_exp(v, W=None):
    """Stable replacement of particles.resampling.log_mean_exp (version 0.4). The library shifts v by max(v) and then
    takes log(sum W exp(v - max v)), which underflows to log(0) = -inf when W is concentrated where v is small. Then all APF
    weights are NaN (3 of 1500 runs on data set 27). This computes the same value as log(sum W exp(v)) with the shift max(v + log W)."""
    if W is None:
        return logsumexp(v) - np.log(len(v))
    with np.errstate(divide='ignore'):
        return logsumexp(v + np.log(W))


rs.log_mean_exp = _log_mean_exp
G = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])


def mean_fn(x, u, acc):
    """F(x) x for a batch x of shape (N, 4). u = 0: constant velocity, 1: turn with +acc, 2: turn with -acc."""
    if u == 0:
        return np.stack([x[:, 0] + x[:, 1], x[:, 1], x[:, 2] + x[:, 3], x[:, 3]], axis=1)
    a = acc if u == 1 else -acc
    om = a / np.sqrt(x[:, 1] ** 2 + x[:, 3] ** 2)
    s, c = np.sin(om), np.cos(om)
    return np.stack([x[:, 0] + s / om * x[:, 1] - (1 - c) / om * x[:, 3],
                     c * x[:, 1] - s * x[:, 3],
                     (1 - c) / om * x[:, 1] + x[:, 2] + s / om * x[:, 3],
                     s * x[:, 1] + c * x[:, 3]], axis=1)


class Sampler(dists.ProbDist):
    dim = 4

    def __init__(self, fn):
        self.fn = fn

    def rvs(self, size=None):
        return self.fn(size)


class Obs(dists.ProbDist):
    """y = (atan2(x2, x0), sqrt(x0^2 + x2^2)) + N(0, R), R = diag(r)."""
    dim = 2

    def __init__(self, x, r):
        self.x, self.r = x, r

    def logpdf(self, y):
        if y is None:
            return np.zeros(self.x.shape[0])
        d = y - np.stack([np.arctan2(self.x[:, 2], self.x[:, 0]), np.sqrt(self.x[:, 0] ** 2 + self.x[:, 2] ** 2)], axis=1)
        if WRAP:   # the bearing is an angle (D-30): the same wrapped residual as simulators/mtt.py
            d[:, 0] = (d[:, 0] + np.pi) % (2 * np.pi) - np.pi
        return -np.log(2 * np.pi) - 0.5 * np.sum(np.log(self.r)) - 0.5 * np.sum(d ** 2 / self.r, axis=1)


class Mtt(ssms.StateSpaceModel):
    """Attributes (set in the constructor): m0, P0, u, acc, q (diagonal of Q), r (diagonal of R)."""

    def PX0(self):
        return Sampler(lambda n: np.random.multivariate_normal(self.m0, self.P0, size=n))

    def PX(self, t, xp):
        def draw(n):
            noise = (np.sqrt(self.q) * np.random.randn(n, 2)) @ G.T
            return mean_fn(xp, self.u[t - 1], self.acc) + noise
        return Sampler(draw)

    def PY(self, t, xp, x):
        return Obs(x, self.r)

    def logeta(self, t, x, data):
        return Obs(mean_fn(x, self.u[t], self.acc), self.r).logpdf(data[t + 1])


def run(fk, n, seed):
    np.random.seed(seed)
    pf = particles.SMC(fk=fk, N=n, resampling='multinomial', ESSrmin=0.5)
    means = []
    for _ in pf:
        if pf.t - 1 >= 1:
            means.append(pf.W @ pf.X)
    return np.array(means), pf.logLt


WRAP = len(sys.argv) > 3 and sys.argv[3] == 'wrap'    # wrapped bearing (D-33)
TAG = f'a{sys.argv[1]}_sigma2_{sys.argv[2]}' + ('_wrap' if WRAP else '')
d = np.load(os.path.join(HERE, 'out', f'gt_{TAG}.npz'))
assert np.allclose(np.diag(np.diag(d['Q'])), d['Q']) and np.allclose(np.diag(np.diag(d['R'])), d['R']), 'Q and R must be diagonal'
assert not np.any(d['q_bias']) and not np.any(d['r_bias']), 'the noise must have zero mean'
acc, n_prt = float(d['a']), int(d['M'])
model = Mtt(m0=d['m0'], P0=d['P0'], u=d['inputs'].astype(int), acc=acc, q=np.diag(d['Q']), r=np.diag(d['R']))

# check the numpy model against the JAX model
for ui in range(3):
    err_f = np.max(np.abs(mean_fn(d['xt'], ui, acc) - d['chk_f'][ui]))
    assert err_f < 1e-4, f'dynamics differ for u={ui}: {err_f}'
ref_g = Obs(d['xt'].astype(float), np.diag(d['R'])).logpdf(d['y_chk'].astype(float))   # float64, as the particles in the runs
err_g = np.max(np.abs(ref_g - d['chk_g']) / np.maximum(1.0, np.abs(d['chk_g'])))   # relative: JAX uses float32
assert err_g < 1e-5, f'emission log prob differs: {err_g}'
print(f'model check passed: max dynamics error {err_f:.1e}, emission rel. error {err_g:.1e}', flush=True)

n_trials, T = d['obs_bpf'].shape[:2]
res = {f'{k}_{f}': [] for k in ('means', 'll') for f in ('bpf', 'apf')}
for s in range(n_trials):
    for name, cls in (('bpf', ssms.Bootstrap), ('apf', ssms.AuxiliaryBootstrap)):
        data = [None] + list(d[f'obs_{name}'][s].astype(float))
        t0 = time.time()
        runs = [run(cls(ssm=model, data=data), n_prt, 1_000_000 * (name == 'apf') + 1000 * s + rep) for rep in range(N_RUNS)]
        assert all(m.shape == (T, 4) for m, _ in runs)
        res[f'means_{name}'].append([m for m, _ in runs])
        res[f'll_{name}'].append([ll for _, ll in runs])
        print(f'trial {s + 1} {name}: {N_RUNS} runs in {time.time() - t0:.1f} s', flush=True)
np.savez(os.path.join(HERE, 'out', f'particles_{TAG}.npz'), **{k: np.array(v) for k, v in res.items()})
print('done', flush=True)
