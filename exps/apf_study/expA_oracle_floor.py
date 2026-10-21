"""Exp. A: lower bound for any filter that cannot move its initial particles.

With Q = 1e-6 I the particles follow the dynamics almost exactly. Each initial particle has a fixed path.
The oracle floor of a seed is the smallest MSE over the M initial paths. Usage: python3 expA_oracle_floor.py M n_seeds a
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
from filters.pf import MVN

M, n_seeds, mtt.acc = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
T = 500
out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)
seeds = jnp.arange(n_seeds)
states, obs = jax.jit(vmap(lambda k: ssm.simulate(k, true_ps, T, inputs)))(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))


def floor(seed, s):
    keys = jr.split(jr.PRNGKey(seed), M + 1)
    x0 = vmap(MVN(loc=true_ps.initial.mean.value, covariance_matrix=true_ps.initial.cov.value).sample, in_axes=(None, 0))((), keys[1:])
    q0 = jnp.zeros(2)

    def step(x, u):
        x = vmap(lambda xi: ssm.dynamics_function(true_ps, xi, q0, u))(x)
        return x, x

    _, xs = lax.scan(step, x0, inputs)             # (T, M, 4)
    err = jnp.mean(jnp.sum((xs - s[:, None, :]) ** 2, axis=-1), axis=0)   # MSE of each initial path
    return jnp.min(err), jnp.argmin(err)


fl, _ = jax.jit(vmap(floor))(seeds, states)
fl = onp.array(fl)
R = onp.load(f'beat_bpf_M{M}_s0_n{n_seeds}_a{mtt.acc}.npy', allow_pickle=True).item()
bpf = R['BPF   multinomial post  thr .5 (paper)']
apf = R['APF   systematic        thr .5']
print(f'M={M}, acc={mtt.acc}: oracle floor (best initial path), median {onp.median(fl):.4f}, runs with floor > 10: {(fl > 10).sum()}/{n_seeds}')
print(f'BPF fails (MSE>10) in {(bpf > 10).sum()} seeds. Of these, floor > 1 in {((bpf > 10) & (fl > 1)).sum()}, floor > 0.1 in {((bpf > 10) & (fl > 0.1)).sum()}')
print(f'APF fails (MSE>10) in {(apf > 10).sum()} seeds. Of these, floor > 1 in {((apf > 10) & (fl > 1)).sum()}, floor > 0.1 in {((apf > 10) & (fl > 0.1)).sum()}')
print('floor of the failed BPF seeds:', onp.round(fl[bpf > 10], 3))
print('floor quantiles 10/50/90/99%:', onp.round(onp.percentile(fl, [10, 50, 90, 99]), 4))
