"""Exp. A (mtt): which design choices let the APF beat the BPF? Paired runs, vmapped over seeds.

Usage: python3 expA_beat_bpf.py M first_seed n_seeds [a]
Every variant sees the same data and the same initial particles in each seed.
Output of each variant: MSE of the weighted mean, averaged over time, per seed.
"""
import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
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
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 40
if len(sys.argv) > 4:
    mtt.acc = float(sys.argv[4])
T = 500
CHUNK = 20


class PFx(pf.ParticleFilter):
    """BPF (lookahead=False) and APF (lookahead=True) with a choice of resampling scheme.

    lookahead=True : resample the ancestors before the propagation if the ESS of the preweights is below thr * M.
    lookahead=False: resample after the weighting if the ESS of the weights is below thr * M (the BPF of pf.py).
    output='post'  : the BPF reports the resampled cloud (as in pf.py). output='pre': it reports the weighted cloud.
    """

    def __init__(self, props, ssm, lookahead=True, scheme='multinomial', output='post', window=1):
        super().__init__(props, ssm)
        self.lookahead, self.scheme, self.output, self.window = lookahead, scheme, output, window

    def _indices(self, key, w, n):
        if self.scheme == 'multinomial':
            return jr.choice(key, n, (n,), p=w, replace=True)
        if self.scheme == 'systematic':
            u = (jr.uniform(key) + jnp.arange(n)) / n
        else:  # stratified
            u = (jr.uniform(key, (n,)) + jnp.arange(n)) / n
        return jnp.clip(jnp.searchsorted(jnp.cumsum(w), u), 0, n - 1)

    def run(self, params, observations, num_particles, key, inputs, ess_threshold):
        ssm, n = self.ssm, num_particles
        dyn = lambda x, q, u: ssm.dynamics_function(params, x, q, u)
        em = vmap(ssm.emission_log_prob, in_axes=(None, None, 0, None))
        sim = vmap(ssm.dynamics_simulator, in_axes=(0, None, 0, None))
        sm = jax.nn.softmax

        def _step(carry, t):
            w, x, key = carry
            key, k_res, k_prop = jr.split(key, 3)
            prop_keys = jr.split(k_prop, n)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            u, y = inputs[t], observations[t]

            if self.lookahead:
                x_pred = vmap(dyn, in_axes=(0, None, None))(x, q0, u)
                aux = em(params, y, x_pred, u)
                # window > 1: look ahead over the next observations along the predicted mean path (not causal)
                for s in range(1, self.window):
                    ts = jnp.minimum(t + s, observations.shape[0] - 1)
                    x_pred = vmap(dyn, in_axes=(0, None, None))(x_pred, q0, inputs[ts])
                    aux = aux + jnp.where(t + s < observations.shape[0], em(params, observations[ts], x_pred, inputs[ts]), 0.0)
                lam = sm(jnp.log(w) + aux)
                res = 1.0 / jnp.sum(lam ** 2) < ess_threshold * n
                idx = jnp.where(res, self._indices(k_res, lam, n), jnp.arange(n))
                new = sim(prop_keys, params, x[idx], u)
                ll_new = em(params, y, new, u)
                w_new = sm(jnp.where(res, ll_new - aux[idx], jnp.log(w) + ll_new))
                est = w_new @ new
                return (w_new, new, key), est

            new = sim(prop_keys, params, x, u)
            w_new = sm(jnp.log(w) + em(params, y, new, u))
            res = 1.0 / jnp.sum(w_new ** 2) < ess_threshold * n
            idx = jnp.where(res, self._indices(k_res, w_new, n), jnp.arange(n))
            x_out = new[idx]
            w_out = jnp.where(res, jnp.ones(n) / n, w_new)
            est = (w_new @ new) if self.output == 'pre' else (w_out @ x_out)
            return (w_out, x_out, key), est

        keys = jr.split(key, n + 1)
        x0 = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample,
                  in_axes=(None, 0))((), keys[1:])
        _, ests = lax.scan(_step, (jnp.ones(n) / n, x0, keys[0]), jnp.arange(observations.shape[0]))
        return ests


out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)

seeds = jnp.arange(first, first + n_seeds)
sim_all = jax.jit(vmap(lambda k: ssm.simulate(k, true_ps, T, inputs)))
states, obs = sim_all(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))
fkeys = vmap(jr.PRNGKey)(seeds)


def evaluate(flt, thr):
    run_one = jax.jit(vmap(lambda o, k, s: jnp.mean(jnp.sum((flt.run(true_ps, o, M, k, inputs, thr) - s) ** 2, axis=-1))))
    res = []
    for i in range(0, n_seeds, CHUNK):
        sl = slice(i, i + CHUNK)
        res.append(onp.array(run_one(obs[sl], fkeys[sl], states[sl])))
    return onp.concatenate(res)


V = {}
mk = lambda **kw: PFx(props, ssm, **kw)
V['BPF   multinomial post  thr .5 (paper)'] = (mk(lookahead=False), 0.5)
V['BPF   multinomial pre   thr .5'] = (mk(lookahead=False, output='pre'), 0.5)
V['BPF   systematic  pre   thr .5'] = (mk(lookahead=False, scheme='systematic', output='pre'), 0.5)
V['BPF   systematic  pre   thr .2'] = (mk(lookahead=False, scheme='systematic', output='pre'), 0.2)
V['BPF   systematic  pre   thr .1'] = (mk(lookahead=False, scheme='systematic', output='pre'), 0.1)
V['APF   multinomial       thr .5 (pf.py)'] = (mk(lookahead=True), 0.5)
V['APF   systematic        thr .5'] = (mk(lookahead=True, scheme='systematic'), 0.5)
V['APF   systematic        thr .3'] = (mk(lookahead=True, scheme='systematic'), 0.3)
V['APF   systematic        thr .2'] = (mk(lookahead=True, scheme='systematic'), 0.2)
V['APF   systematic        thr .1'] = (mk(lookahead=True, scheme='systematic'), 0.1)
V['APF   stratified        thr .2'] = (mk(lookahead=True, scheme='stratified'), 0.2)

if len(sys.argv) > 5 and sys.argv[5] == 'la':
    V = {k: v for k, v in V.items() if 'paper' in k or 'systematic  pre   thr .5' in k}
    for L in (5, 20, 50):
        V[f'APF   systematic  window {L:2d}  thr .5'] = (mk(lookahead=True, scheme='systematic', window=L), 0.5)

if len(sys.argv) > 5 and sys.argv[5] == 'hold':
    V = {k: v for k, v in V.items() if 'paper' in k or 'systematic  pre   thr .5' in k or 'APF   multinomial' in k}
    V['APF   systematic  window 20  thr .5'] = (mk(lookahead=True, scheme='systematic', window=20), 0.5)

if len(sys.argv) > 5 and sys.argv[5] == 'bpfonly':
    V = {k: v for k, v in V.items() if 'paper' in k}

if len(sys.argv) > 5 and sys.argv[5] == 'low':
    ref_item = {k: v for k, v in V.items() if 'paper' in k}
    V = dict(ref_item)
    for thr in (0.05, 0.02, 0.0):
        V[f'BPF   systematic  pre   thr {thr}'] = (mk(lookahead=False, scheme='systematic', output='pre'), thr)
        V[f'APF   systematic        thr {thr}'] = (mk(lookahead=True, scheme='systematic'), thr)

R = {}
for name, (flt, thr) in V.items():
    R[name] = evaluate(flt, thr)
    print(f'done {name}', flush=True)

ref = R['BPF   multinomial post  thr .5 (paper)']
rng = onp.random.default_rng(0)
print(f'\nM={M}, seeds {first}..{first + n_seeds - 1}, acc={mtt.acc}, MSE of the weighted mean')
print(f'{"variant":42s} {"mean":>8s} {"median":>7s} {"fail>10":>8s} {"better than paper BPF":>22s}  95% CI of mean diff to paper BPF')
for name, a in R.items():
    d = a - ref
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = onp.percentile(boots, [2.5, 97.5])
    print(f'{name:42s} {a.mean():8.3f} {onp.median(a):7.3f} {(a > 10).sum():5d}/{len(a):<3d} {(a < ref).sum():12d}/{len(a):<3d}      [{lo:7.3f}, {hi:7.3f}]')
tag = '_' + sys.argv[5] if len(sys.argv) > 5 else ''
onp.save(f'beat_bpf_M{M}_s{first}_n{n_seeds}_a{mtt.acc}{tag}.npy', {k: v for k, v in R.items()}, allow_pickle=True)
