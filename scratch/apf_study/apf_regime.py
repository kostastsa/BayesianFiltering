"""Exp. A (mtt): in which regime does the APF beat the BPF? Sweep of the process noise Q and the observation noise R.

Usage: python3 apf_regime.py M n_seeds T [a]
For each (q, r): data and filter use Q = q I and R = r I. BPF and APF are the filters of pf.py (ess_threshold 0.5).
Both see the same data and the same initial particles in each seed. MSE: weighted mean, summed over the state
dimensions, averaged over time.
Diagnostics (cloud of the BPF, averaged over time, median over seeds):
  sd_d  : weighted sd over particles of d = log g(y_t|x_t) - log g(y_t|mu(x_{t-1}))  (error of the first-stage weight)
  sd_ll : weighted sd of log g(y_t|x_t)                                              (spread of the BPF weights)
  ess1  : ESS/M of the first-stage weights of the APF       ess2: ESS/M of the BPF weights
"""
import os
import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import math
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from simulators import mtt
from util.param import sample_prior
from filters import pf
from filters.pf import _get_params, MVN

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
T = int(sys.argv[3]) if len(sys.argv) > 3 else 500
if len(sys.argv) > 4:
    mtt.acc = float(sys.argv[4])
CHUNK = 10
QS = [float(v) for v in os.environ.get('QS', '1e-6,1e-4,1e-2,1e-1').split(',')]
RS = [float(v) for v in os.environ.get('RS', '2.5,0.025').split(',')]

out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)
seeds = jnp.arange(n_seeds)
fkeys = vmap(jr.PRNGKey)(seeds)


def wmean_mse(flt):
    def one(o, k, s):
        outs, _ = flt.run(true_ps, o, M, k, inputs, 0.5)
        est = jnp.einsum('mt,mtd->td', outs['weights'], outs['particles'])
        return jnp.mean(jnp.sum((est - s) ** 2, axis=-1))
    return jax.jit(vmap(one))


def diag_one(obs, key):
    """Run the BPF cloud and record how well the first-stage weight of the APF predicts the true weight."""
    n = M
    dyn = lambda x, q, u: ssm.dynamics_function(true_ps, x, q, u)
    em = vmap(ssm.emission_log_prob, in_axes=(None, None, 0, None))
    sim = vmap(ssm.dynamics_simulator, in_axes=(0, None, 0, None))
    sm = jax.nn.softmax

    def wsd(w, a):
        m = w @ a
        return jnp.sqrt(w @ (a - m) ** 2)

    def step(carry, t):
        w, x, key = carry
        key, k_res, k_prop = jr.split(key, 3)
        q0 = _get_params(true_ps.dynamics.bias.value, 2, t)
        u, y = inputs[t], obs[t]
        aux = em(true_ps, y, vmap(dyn, in_axes=(0, None, None))(x, q0, u), u)
        new = sim(jr.split(k_prop, n), true_ps, x, u)
        ll = em(true_ps, y, new, u)
        lam = sm(jnp.log(w) + aux)
        w_new = sm(jnp.log(w) + ll)
        ess1, ess2 = 1.0 / jnp.sum(lam ** 2) / n, 1.0 / jnp.sum(w_new ** 2) / n
        res = ess2 < 0.5
        idx = jnp.where(res, jr.choice(k_res, n, (n,), p=w_new, replace=True), jnp.arange(n))
        w_out = jnp.where(res, jnp.ones(n) / n, w_new)
        return (w_out, new[idx], key), jnp.stack([wsd(w, ll - aux), wsd(w, ll), ess1, ess2, res * 1.0, (ess1 < 0.5) * 1.0])

    keys = jr.split(key, n + 1)
    x0 = vmap(MVN(loc=true_ps.initial.mean.value, covariance_matrix=true_ps.initial.cov.value).sample,
              in_axes=(None, 0))((), keys[1:])
    _, rec = lax.scan(step, (jnp.ones(n) / n, x0, keys[0]), jnp.arange(obs.shape[0]))
    return rec.mean(axis=0)


def sign_p(k, n):
    """Two-sided exact sign test."""
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def chunks(f, *arrs):
    return onp.concatenate([onp.array(f(*[a[i:i + CHUNK] for a in arrs])) for i in range(0, n_seeds, CHUNK)])


rows, saved = [], {}
for r in RS:
    for q in QS:
        true_ps.dynamics.cov.value = q * jnp.eye(2)
        true_ps.emissions.cov.value = r * jnp.eye(2)
        sim_all = jax.jit(vmap(lambda k: ssm.simulate(k, true_ps, T, inputs)))
        states, obs = sim_all(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))
        bpf = chunks(wmean_mse(pf.BPF(props, ssm)), obs, fkeys, states)
        apf = chunks(wmean_mse(pf.APF(props, ssm)), obs, fkeys, states)
        dg = chunks(jax.jit(vmap(diag_one)), obs, fkeys)
        wins = int((apf < bpf).sum())
        lr = onp.log(apf / bpf)
        rows.append((r, q, onp.median(bpf), onp.median(apf), onp.exp(onp.median(lr)), wins, sign_p(wins, n_seeds),
                     *onp.median(dg, axis=0)))
        saved[(r, q)] = dict(bpf=bpf, apf=apf, diag=dg)
        print(f'done r={r} q={q}', flush=True)

print(f'\nM={M}, T={T}, seeds={n_seeds}, acc={mtt.acc}. MSE = weighted mean, summed over states, averaged over time.')
print('ratio = median over seeds of APF/BPF (below 1: APF better). wins = seeds with APF MSE below BPF MSE.')
print(f'{"r":>6s} {"q":>7s} | {"med BPF":>9s} {"med APF":>9s} {"ratio":>6s} {"wins":>9s} {"p":>7s} | '
      f'{"sd_d":>8s} {"sd_ll":>8s} {"ess1":>5s} {"ess2":>5s} {"BPF res":>8s} {"APF res":>8s}')
for row in rows:
    r, q, mb, ma, ratio, wins, p, sdd, sdl, e1, e2, rb, ra = row
    print(f'{r:6g} {q:7.0e} | {mb:9.3g} {ma:9.3g} {ratio:6.2f} {wins:5d}/{n_seeds:<3d} {p:7.3f} | '
          f'{sdd:8.2e} {sdl:8.2e} {e1:5.2f} {e2:5.2f} {rb:8.2f} {ra:8.2f}')
onp.save(f'apf_regime_M{M}_n{n_seeds}_T{T}_a{mtt.acc}.npy', saved, allow_pickle=True)
