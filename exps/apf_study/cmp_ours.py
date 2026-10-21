"""Exp. A (mtt): run our BPF and APF (pf.py) on several (q, r) settings and save data, MSE and log-likelihoods.

Usage: python3 cmp_ours.py M n_seeds T reps
Saves cmp_data_q{q}_r{r}.npz for the comparison with the particles library (cmp_particles.py).
Same model as apf_regime.py: Q = q I, R = r I. Data seeds: PRNGKey(100 + s). Filter keys: PRNGKey(1000 * rep + s).
"""
import os
import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from simulators import mtt
from util.param import sample_prior
from filters import pf

M, n_seeds, T, reps = (int(a) for a in sys.argv[1:5])
mtt.acc = float(os.environ.get('ACC', mtt.acc))
CONFIGS = [tuple(float(v) for v in c.split(':')) for c in os.environ.get('CONFIGS', '1e-6:2.5,1e-4:0.025,1e-2:0.025').split(',')]
SUF = os.environ.get('SUF', '')
CHUNK = 10

out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)
seeds = jnp.arange(n_seeds)


def runner(flt):
    def one(o, k, s):
        outs, ll = flt.run(ps, o, M, k, inputs, 0.5)
        est = jnp.einsum('mt,mtd->td', outs['weights'], outs['particles'])
        return jnp.mean(jnp.sum((est - s) ** 2, axis=-1)), ll
    return jax.jit(vmap(one))


def chunks(f, *arrs):
    res = [f(*[a[i:i + CHUNK] for a in arrs]) for i in range(0, n_seeds, CHUNK)]
    return onp.concatenate([onp.array(r[0]) for r in res]), onp.concatenate([onp.array(r[1]) for r in res])


for q, r in CONFIGS:
    ps.dynamics.cov.value = q * jnp.eye(2)
    ps.emissions.cov.value = r * jnp.eye(2)
    states, obs = jax.jit(vmap(lambda k: ssm.simulate(k, ps, T, inputs)))(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))
    res = {}
    for name, flt in (('bpf', pf.BPF(props, ssm)), ('apf', pf.APF(props, ssm))):
        run = runner(flt)
        mse, ll = [], []
        for rep in range(reps):
            m, l = chunks(run, obs, vmap(lambda s: jr.PRNGKey(1000 * rep + s))(seeds), states)
            mse.append(m)
            ll.append(l)
        res[f'mse_{name}'], res[f'll_{name}'] = onp.stack(mse, 1), onp.stack(ll, 1)   # (n_seeds, reps)
    # values for checking the numpy re-implementation of the model in cmp_particles.py
    xt = onp.array(states[0, [0, 50, 150, 250, 350, 450]])
    chk_f = onp.stack([[onp.array(ssm.dynamics_function(ps, jnp.array(x), jnp.zeros(2), u)) for x in xt] for u in (0, 1, 2)])
    chk_g = onp.array([ssm.emission_log_prob(ps, obs[0, 10], jnp.array(x), 0) for x in xt])
    onp.savez(f'cmp_data_q{q:g}_r{r:g}{SUF}.npz', obs=onp.array(obs), states=onp.array(states), inputs=onp.array(inputs),
              m0=onp.array(ps.initial.mean.value), P0=onp.array(ps.initial.cov.value), acc=mtt.acc, q=q, r=r,
              xt=xt, chk_f=chk_f, chk_g=chk_g, y_chk=onp.array(obs[0, 10]), **res)
    print(f'done q={q:g} r={r:g}', flush=True)
