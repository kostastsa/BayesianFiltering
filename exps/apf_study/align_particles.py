"""Exp. A (mtt): does the time alignment of pf.py change the results?

Usage: ./pvenv/bin/python align_particles.py M n_seeds T
simulate() draws x_0 from N(m0, P0) and y_0 from x_0 (no transition before y_0). Then x_t = f(x_{t-1}, u[t]) for t >= 1.
(A) 'repo'   : as pf.py. The initial particles z ~ N(m0, P0) are propagated with u[0] before y_0 is used.
               The filter prior of x_0 is then f(z, u[0]), not N(m0, P0).
(B) 'aligned': x_0 ~ N(m0, P0) is weighted with y_0 at once, then transitions with u[t], t >= 1 (the convention of simulate()).
Same data as regime_particles.py (seeds 5000 + s), same filters of the particles library (multinomial, ESSrmin 0.5).
Also: how far is the best prior draw of (A), z* = f(., u[0])^{-1}(x_0), from the prior N(m0, P0)?
"""
import os
import sys
import math
import warnings
import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssms
from joblib import Parallel, delayed

M, n_seeds, T = (int(a) for a in sys.argv[1:4])
QS = [float(v) for v in os.environ.get('QS', '1e-6,1e-4,1e-2,1e-1').split(',')]
RS = [float(v) for v in os.environ.get('RS', '2.5,0.025').split(',')]
G = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
base = np.load('cmp_data_q1e-06_r2.5.npz')
M0_, P0_, ACC = base['m0'], base['P0'], float(os.environ.get('ACC', base['acc']))
U = np.array([1] * int(2 * T / 5) + [0] * int(T / 5) + [2] * int(2 * T / 5))


def mean_fn(x, u, acc):
    if u == 0:
        return np.stack([x[:, 0] + x[:, 1], x[:, 1], x[:, 2] + x[:, 3], x[:, 3]], axis=1)
    a = acc if u == 1 else -acc
    om = a / np.sqrt(x[:, 1] ** 2 + x[:, 3] ** 2)
    s, c = np.sin(om), np.cos(om)
    return np.stack([x[:, 0] + s / om * x[:, 1] - (1 - c) / om * x[:, 3],
                     c * x[:, 1] - s * x[:, 3],
                     (1 - c) / om * x[:, 1] + x[:, 2] + s / om * x[:, 3],
                     s * x[:, 1] + c * x[:, 3]], axis=1)


def inverse_turn(x1, u, acc):
    """z such that mean_fn(z, u) = x1, for u = 1 or 2 (the turn keeps the speed, so the turn rate is the same)."""
    a = acc if u == 1 else -acc
    om = a / np.hypot(x1[1], x1[3])
    s, c = np.sin(om), np.cos(om)
    vx, vy = c * x1[1] + s * x1[3], -s * x1[1] + c * x1[3]            # rotate the velocity back by the angle om
    px = x1[0] - (s / om * vx - (1 - c) / om * vy)
    py = x1[2] - ((1 - c) / om * vx + s / om * vy)
    return np.array([px, vx, py, vy])


def emit(x):
    return np.stack([np.arctan2(x[:, 2], x[:, 0]), np.sqrt(x[:, 0] ** 2 + x[:, 2] ** 2)], axis=1)


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
        d = y - emit(self.x)
        return -np.log(2 * np.pi) - np.log(self.r) - 0.5 * np.sum(d ** 2, axis=1) / self.r


class Repo(ssms.StateSpaceModel):
    """(A): data = [None, y_0, ..., y_{T-1}]. Time k of the library = transition with u[k-1]."""

    def PX0(self):
        return Sampler(lambda n: np.random.multivariate_normal(M0_, P0_, size=n))

    def PX(self, t, xp):
        return Sampler(lambda n: mean_fn(xp, U[t - 1], ACC) + np.sqrt(self.q) * np.random.randn(n, 2) @ G.T)

    def PY(self, t, xp, x):
        return Obs(x, self.r)

    def logeta(self, t, x, data):
        return Obs(mean_fn(x, U[t], ACC), self.r).logpdf(data[t + 1])


class Aligned(Repo):
    """(B): data = [y_0, ..., y_{T-1}]. Time k of the library = x_k, transition with u[k]."""

    def PX(self, t, xp):
        return Sampler(lambda n: mean_fn(xp, U[t], ACC) + np.sqrt(self.q) * np.random.randn(n, 2) @ G.T)

    def logeta(self, t, x, data):
        return Obs(mean_fn(x, U[t + 1], ACC), self.r).logpdf(data[t + 1])


def simulate(seed, q, r):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(M0_, P0_)
    xs, ys = [x], [emit(x[None])[0] + np.sqrt(r) * rng.standard_normal(2)]
    for t in range(1, T):
        x = mean_fn(x[None], U[t], ACC)[0] + np.sqrt(q) * G @ rng.standard_normal(2)
        xs.append(x)
        ys.append(emit(x[None])[0] + np.sqrt(r) * rng.standard_normal(2))
    return np.array(xs), np.array(ys)


def run(fk, states, seed, shift):
    np.random.seed(seed)
    pf = particles.SMC(fk=fk, N=M, resampling='multinomial', ESSrmin=0.5)
    err = []
    for _ in pf:
        k = pf.t - 1
        if k >= shift:
            err.append(np.sum((pf.W @ pf.X - states[k - shift]) ** 2))
    return np.mean(err)


def one_seed(s, q, r):
    warnings.simplefilter('ignore')
    states, obs = simulate(5000 + s, q, r)
    A, B = Repo(q=q, r=r), Aligned(q=q, r=r)
    dA, dB = [None] + list(obs), list(obs)
    out = [run(ssms.Bootstrap(ssm=A, data=dA), states, 100 * s + 1, 1),
           run(ssms.AuxiliaryBootstrap(ssm=A, data=dA), states, 100 * s + 2, 1),
           run(ssms.Bootstrap(ssm=B, data=dB), states, 100 * s + 1, 0),
           run(ssms.AuxiliaryBootstrap(ssm=B, data=dB), states, 100 * s + 2, 0)]
    # prior mismatch of (A): log N(z*; m0, P0) - log N(x_0; m0, P0), and the squared Mahalanobis distances
    zs = inverse_turn(states[0], U[0], ACC)
    Pi = np.linalg.inv(P0_)
    m_x, m_z = (states[0] - M0_) @ Pi @ (states[0] - M0_), (zs - M0_) @ Pi @ (zs - M0_)
    return out + [m_x, m_z]


def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b):
    """geo. mean of b/a, seeds with b < a, sign test p."""
    ok = np.isfinite(a) & np.isfinite(b)
    w = int((b[ok] < a[ok]).sum())
    return np.exp(np.mean(np.log(b[ok] / a[ok]))), w, int(ok.sum()), sign_p(w, int(ok.sum()))


print(f'M={M}, T={T}, seeds={n_seeds}, acc={ACC}, u[0]={U[0]}')
print('(A) repo convention of pf.py, (B) aligned with simulate(). ratio = geo. mean over seeds of MSE(B)/MSE(A), wins = seeds with B < A.')
print(f'{"r":>6s} {"q":>7s} | {"BPF: med A":>10s} {"med B":>8s} {"ratio":>6s} {"wins":>8s} {"p":>6s} | '
      f'{"APF: med A":>10s} {"med B":>8s} {"ratio":>6s} {"wins":>8s} {"p":>6s} | {"APF/BPF in A":>12s} {"in B":>6s}')
maha = None
for r in RS:
    for q in QS:
        o = np.array(Parallel(n_jobs=-1)(delayed(one_seed)(s, q, r) for s in range(n_seeds)))
        bA, aA, bB, aB = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        if maha is None:
            maha = o[:, 4:6]          # the data states do not depend on r and q for x_0
        pb, pa = paired(bA, bB), paired(aA, aB)
        print(f'{r:6g} {q:7.0e} | {np.nanmedian(bA):10.3g} {np.nanmedian(bB):8.3g} {pb[0]:6.2f} {pb[1]:4d}/{pb[2]:<3d} {pb[3]:6.3f} | '
              f'{np.nanmedian(aA):10.3g} {np.nanmedian(aB):8.3g} {pa[0]:6.2f} {pa[1]:4d}/{pa[2]:<3d} {pa[3]:6.3f} | '
              f'{np.exp(np.nanmedian(np.log(aA / bA))):12.2f} {np.exp(np.nanmedian(np.log(aB / bB))):6.2f}', flush=True)

print('\nPrior mismatch of (A). Squared Mahalanobis distance to N(m0, P0), mean over seeds (4 = mean of a chi-square with 4 df):')
print(f'  true x_0 : mean {maha[:, 0].mean():.2f}, median {np.median(maha[:, 0]):.2f}')
print(f'  z*       : mean {maha[:, 1].mean():.2f}, median {np.median(maha[:, 1]):.2f}   (z* = the initial particle that reproduces the data path in (A))')
print(f'  seeds with z* farther from the prior than x_0: {(maha[:, 1] > maha[:, 0]).sum()}/{len(maha)}; with squared distance > 9.49 (95% quantile): '
      f'x_0 {(maha[:, 0] > 9.49).sum()}, z* {(maha[:, 1] > 9.49).sum()} of {len(maha)}')
