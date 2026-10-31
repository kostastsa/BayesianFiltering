"""Exp. A (mtt) with the particles library (Chopin and Papaspiliopoulos, 2020): BPF and APF on the data of cmp_ours.py.

Usage: ./pvenv/bin/python cmp_particles.py M reps [n_seeds]
The model is the numpy copy of simulators/mtt.py. It is checked against the values of our JAX code (chk_f, chk_g).
Time shift: particles starts with an unobserved x_0 = the initial particles. x_k (k >= 1) uses the input u[k-1]
and the observation obs[k-1]. This is the same order as pf.py, which propagates the initial particles before it
uses obs[0]. The APF auxiliary function is eta_t(x_t) = g(y_{t+1} | f(x_t, q=0)), as in pf.py.
Resampling: multinomial, ESSrmin = 0.5 on the (auxiliary) weights, as in pf.py.
Output: the weighted mean at the end of every step (before the next resampling).
"""
import os
import sys
import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssms

M, reps = int(sys.argv[1]), int(sys.argv[2])
CONFIGS = [tuple(float(v) for v in c.split(':')) for c in os.environ.get('CONFIGS', '1e-6:2.5,1e-4:0.025,1e-2:0.025').split(',')]
SUF = os.environ.get('SUF', '')
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
    dim = 2

    def __init__(self, x, r):
        self.x, self.r = x, r

    def logpdf(self, y):
        if y is None:
            return np.zeros(self.x.shape[0])
        d = y - np.stack([np.arctan2(self.x[:, 2], self.x[:, 0]), np.sqrt(self.x[:, 0] ** 2 + self.x[:, 2] ** 2)], axis=1)
        return -np.log(2 * np.pi) - np.log(self.r) - 0.5 * np.sum(d ** 2, axis=1) / self.r


class Mtt(ssms.StateSpaceModel):
    """Attributes (set in the constructor): m0, P0, u, acc, q, r."""

    def PX0(self):
        return Sampler(lambda n: np.random.multivariate_normal(self.m0, self.P0, size=n))

    def PX(self, t, xp):
        def draw(n):
            noise = np.sqrt(self.q) * np.random.randn(n, 2) @ G.T
            return mean_fn(xp, self.u[t - 1], self.acc) + noise
        return Sampler(draw)

    def PY(self, t, xp, x):
        return Obs(x, self.r)

    def logeta(self, t, x, data):
        return Obs(mean_fn(x, self.u[t], self.acc), self.r).logpdf(data[t + 1])


def run(fk, states, seed):
    np.random.seed(seed)
    pf = particles.SMC(fk=fk, N=M, resampling='multinomial', ESSrmin=0.5)
    err = []
    for _ in pf:
        k = pf.t - 1
        if k >= 1:
            est = pf.W @ pf.X
            err.append(np.sum((est - states[k - 1]) ** 2))
    return np.mean(err), pf.logLt


for q, r in CONFIGS:
    d = np.load(f'cmp_data_q{q:g}_r{r:g}{SUF}.npz')
    model = Mtt(m0=d['m0'], P0=d['P0'], u=d['inputs'].astype(int), acc=float(d['acc']), q=q, r=r)

    # check the numpy model against the JAX model
    for ui in range(3):
        err_f = np.max(np.abs(mean_fn(d['xt'], ui, float(d['acc'])) - d['chk_f'][ui]))
        assert err_f < 1e-4, f'dynamics differ for u={ui}: {err_f}'
    ref_g = Obs(d['xt'], r).logpdf(d['y_chk'])
    err_g = np.max(np.abs(ref_g - d['chk_g']) / np.maximum(1.0, np.abs(d['chk_g'])))   # relative: JAX uses float32
    assert err_g < 1e-5, f'emission log prob differs: {err_g}'

    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else len(d['obs'])
    res = {k: np.zeros((n_seeds, reps)) for k in ('mse_bpf', 'll_bpf', 'mse_apf', 'll_apf')}
    for s in range(n_seeds):
        data = [None] + list(d['obs'][s].astype(float))
        for rep in range(reps):
            for name, cls in (('bpf', ssms.Bootstrap), ('apf', ssms.AuxiliaryBootstrap)):
                res[f'mse_{name}'][s, rep], res[f'll_{name}'][s, rep] = run(cls(ssm=model, data=data), d['states'][s].astype(float), 10_000 * rep + s)
    np.savez(f'cmp_particles_q{q:g}_r{r:g}{SUF}.npz', **res)
    print(f'done q={q:g} r={r:g} (model check passed: max dyn err {err_f:.1e}, emission rel. err {err_g:.1e})', flush=True)
