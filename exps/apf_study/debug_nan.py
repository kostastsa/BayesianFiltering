"""Find the first time step with a non-finite value in the BPF run of expA_rerun.py that gave a NaN MSE (a = 0.5, sigma2 = 0.025, M = 100)."""
import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import numpy as onp
import jax, jax.numpy as jnp, jax.random as jr
from jax import vmap
from simulators import mtt
from util.param import sample_prior
from filters import pf
a, r, M, n_sims, T = 0.5, 0.025, 100, 100, 500
mtt.acc = a
out = mtt.setup(4, 2, 0); ssm, props = out['ssm'], out['props']
ps = sample_prior(jr.PRNGKey(0), props, 1)[0]; ps.emissions.cov.value = r * jnp.eye(2)
inputs = mtt.get_inputs(None, T)
seeds = jnp.arange(n_sims)
states, obs = jax.jit(vmap(lambda k: ssm.simulate(k, ps, T, inputs)))(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))
fkeys = vmap(lambda s: jr.PRNGKey(1000 + s))(seeds)
d = onp.load(f'rerun_a{a:g}_r{r:g}_M{M}.npz'); s = int(onp.where(onp.isnan(d['bpf']))[0][0]); print('NaN seed', s)
flt = pf.BPF(props, ssm)
i0 = (s // 10) * 10
f = jax.jit(vmap(lambda o, k: flt.run(ps, o, M, k, inputs, 0.5)[0]))
outs = f(obs[i0:i0 + 10], fkeys[i0:i0 + 10])
w, x = onp.array(outs['weights'][s - i0]), onp.array(outs['particles'][s - i0])
print('non-finite: weights', int((~onp.isfinite(w)).sum()), ' particles', int((~onp.isfinite(x)).sum()))
bad = onp.where((~onp.isfinite(w)).any(0) | (~onp.isfinite(x)).any((0, 2)))[0]
t0 = int(bad[0]); print('first bad t =', t0, ' u =', int(inputs[t0]))
print('weights bad:', bool((~onp.isfinite(w[:, t0])).any()), ' particles bad:', bool((~onp.isfinite(x[:, t0])).any()))
sp = onp.sqrt(x[:, t0 - 1, 1] ** 2 + x[:, t0 - 1, 3] ** 2)
print('prev step: sum w', w[:, t0 - 1].sum(), ' min speed', sp.min(), ' min radius', onp.hypot(x[:, t0 - 1, 0], x[:, t0 - 1, 2]).min())
print('true state t0-1:', onp.array(states[s, t0 - 1]), ' obs t0:', onp.array(obs[s, t0]))
