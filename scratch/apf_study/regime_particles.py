"""Exp. A (mtt): the sweep of apf_regime.py again, now with numpy data and the filters of the particles library only.

Usage: ./pvenv/bin/python regime_particles.py M n_seeds T
Data: numpy copy of the simulate() of the gaussfiltax repo (x_0 from the initial law, y_0 = g(x_0) + noise, then x_t with
input u[t], t = 1..T-1). The seeds are new, so the data sets are not the ones of apf_regime.py.
Filters: particles.state_space_models.Bootstrap (BPF) and AuxiliaryBootstrap (APF, eta = g(y_{t+1} | f(x_t, q=0))),
ESSrmin = 0.5. Main table: multinomial resampling (as in pf.py). Second table: the default of the library (systematic).
Time shift as in cmp_particles.py: the initial particles are propagated before the first observation is used.
Diagnostics (cloud of the BPF, mean over time, median over seeds): see apf_regime.py.
"""
import os
import sys
import math
import warnings
import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssms
from scipy.special import softmax
from joblib import Parallel, delayed

M, n_seeds, T = (int(a) for a in sys.argv[1:4])
QS = [float(v) for v in os.environ.get('QS', '1e-6,1e-4,1e-2,1e-1').split(',')]
RS = [float(v) for v in os.environ.get('RS', '2.5,0.025').split(',')]
G = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
base = np.load('cmp_data_q1e-06_r2.5.npz')
M0_, P0_, ACC = base['m0'], base['P0'], float(os.environ.get('ACC', base['acc']))
U = np.array([1] * int(2 * T / 5) + [0] * int(T / 5) + [2] * int(2 * T / 5))


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


class Mtt(ssms.StateSpaceModel):
    """Attributes (set in the constructor): q, r."""

    def PX0(self):
        return Sampler(lambda n: np.random.multivariate_normal(M0_, P0_, size=n))

    def PX(self, t, xp):
        def draw(n):
            return mean_fn(xp, U[t - 1], ACC) + np.sqrt(self.q) * np.random.randn(n, 2) @ G.T
        return Sampler(draw)

    def PY(self, t, xp, x):
        return Obs(x, self.r)

    def logeta(self, t, x, data):
        return Obs(mean_fn(x, U[t], ACC), self.r).logpdf(data[t + 1])


def simulate(seed, q, r):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(M0_, P0_)
    xs, ys = [x], [emit(x[None])[0] + np.sqrt(r) * rng.standard_normal(2)]
    for t in range(1, T):
        x = mean_fn(x[None], U[t], ACC)[0] + np.sqrt(q) * G @ rng.standard_normal(2)
        xs.append(x)
        ys.append(emit(x[None])[0] + np.sqrt(r) * rng.standard_normal(2))
    return np.array(xs), np.array(ys)


def wsd(w, a):
    m = w @ a
    return np.sqrt(w @ (a - m) ** 2)


def run(fk, states, seed, resampling, r, diag=False):
    np.random.seed(seed)
    pf = particles.SMC(fk=fk, N=M, resampling=resampling, ESSrmin=0.5)
    err, rs, dg = [], [], []
    for _ in pf:
        k = pf.t - 1
        if k < 1:
            continue
        err.append(np.sum((pf.W @ pf.X - states[k - 1]) ** 2))
        rs.append(pf.rs_flag)
        if diag:
            y = fk.data[k]
            ll = Obs(pf.X, r).logpdf(y)
            aux = Obs(mean_fn(pf.Xp, U[k - 1], ACC), r).logpdf(y)
            w = softmax(pf.wgts.lw - ll)
            dg.append([wsd(w, ll - aux), wsd(w, ll), 1.0 / np.sum(softmax(pf.wgts.lw - ll + aux) ** 2) / M, pf.wgts.ESS / M])
    return np.mean(err), np.mean(rs), (np.mean(dg, axis=0) if diag else None)


def one_seed(s, q, r):
    warnings.simplefilter('ignore')
    states, obs = simulate(5000 + s, q, r)
    data = [None] + list(obs)
    model = Mtt(q=q, r=r)
    out = {}
    for res in ('multinomial', 'systematic'):
        b = run(ssms.Bootstrap(ssm=model, data=data), states, 100 * s + 1, res, r, diag=(res == 'multinomial'))
        a = run(ssms.AuxiliaryBootstrap(ssm=model, data=data), states, 100 * s + 2, res, r)
        out[res] = (b[0], a[0], b[1], a[1], b[2])
    return out


def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(bpf, apf):
    ok = np.isfinite(bpf) & np.isfinite(apf)
    lr = np.log(apf[ok] / bpf[ok])
    wins = int((apf[ok] < bpf[ok]).sum())
    return np.nanmedian(bpf), np.nanmedian(apf), np.exp(np.median(lr)), wins, int(ok.sum()), sign_p(wins, int(ok.sum())), int((~ok).sum())


rows = {'multinomial': [], 'systematic': []}
saved = {}
for r in RS:
    for q in QS:
        outs = Parallel(n_jobs=-1)(delayed(one_seed)(s, q, r) for s in range(n_seeds))
        for res in rows:
            bpf = np.array([o[res][0] for o in outs])
            apf = np.array([o[res][1] for o in outs])
            rb = np.mean([o[res][2] for o in outs])
            ra = np.mean([o[res][3] for o in outs])
            dg = np.median(np.array([o['multinomial'][4] for o in outs]), axis=0) if res == 'multinomial' else None
            rows[res].append((r, q, summarize(bpf, apf), dg, np.median([o[res][2] for o in outs]), np.median([o[res][3] for o in outs])))
            saved[(res, r, q)] = dict(bpf=bpf, apf=apf)
        print(f'done r={r} q={q}', flush=True)

print(f'\nparticles library only (numpy data). M={M}, T={T}, seeds={n_seeds}, acc={ACC}. MSE = weighted mean, summed over states, averaged over time.')
print('ratio = median over seeds of APF/BPF (below 1: APF better). wins = seeds with APF MSE below BPF MSE. nan = runs left out.')
for res in rows:
    print(f'\n-- resampling: {res}')
    print(f'{"r":>6s} {"q":>7s} | {"med BPF":>9s} {"med APF":>9s} {"ratio":>6s} {"wins":>9s} {"p":>7s} {"nan":>3s} | '
          f'{"sd_d":>8s} {"sd_ll":>8s} {"ess1":>5s} {"ess2":>5s} {"BPF res":>8s} {"APF res":>8s}')
    for r, q, (mb, ma, ratio, wins, n, p, nn), dg, rb, ra in rows[res]:
        diag = f'{dg[0]:8.2e} {dg[1]:8.2e} {dg[2]:5.2f} {dg[3]:5.2f}' if dg is not None else f'{"":8s} {"":8s} {"":5s} {"":5s}'
        print(f'{r:6g} {q:7.0e} | {mb:9.3g} {ma:9.3g} {ratio:6.2f} {wins:5d}/{n:<3d} {p:7.3f} {nn:3d} | {diag} {rb:8.2f} {ra:8.2f}')
np.save(f'regime_particles_M{M}_n{n_seeds}_T{T}_a{ACC:g}.npy', saved, allow_pickle=True)
