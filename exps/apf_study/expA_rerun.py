"""Exp. A (mtt): BPF and APF of filters/pf.py in the settings of Tables I and III.

Usage: [FILTERS=bpf,apf,...] python3 expA_rerun.py a sigma2 M n_sims
Run it in exps/apf_study/out. It saves rerun_a{a}_r{sigma2}_M{M}.npz and prints one summary line per filter.

Settings: Q = 1e-6 I, R = sigma2 I, T = 500, the turn parameter a (mtt.acc), N_sim data sets (PRNGKey(100 + s)).
All filters see the same data and the same initial particles (filter key PRNGKey(1000 + s)).
Filters:
  bpf        BPF, resample if ESS < 0.5 M. Estimate: the cloud after resampling (the output of BPF.run).
  apf        APF, resample the ancestors if the first-stage ESS < 0.5 M. Estimate: the weighted propagated cloud.
  apf_every  APF, resample the ancestors at every step. This is the rule of the submitted tables.
  bpf_every  BPF, resample at every step.
  bpf_pre    BPF, resample if ESS < 0.5 M. Estimate: the weighted cloud before resampling (the convention of the APF
             and of the particles library).
MSE = time average of the squared error of the posterior mean, summed over the state dimensions.
A run "loses track" if its MSE is above 1 (or above 10, the scale of the distance that the target travels).
"""
import os
import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import time
import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from simulators import mtt
from util.param import sample_prior
from util.sample import resample
from filters import pf
from filters.pf import swap_axes_on_values, MVN

acc, r, M, n_sims = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
T = 500
mtt.acc = acc


class BPFPre(pf.BPF):
    """BPF.run with the output of the weighted cloud before the resampling."""

    def run(self, params, observations, num_particles, key, inputs, ess_threshold=0.5):
        num_timesteps = observations.shape[0]

        def _step(carry, t):
            weights, ll, prev_states, key = carry
            u, y = inputs[t], observations[t]
            keys = jr.split(key, num_particles + 1)
            next_key = keys[0]
            new_states = vmap(self.ssm.dynamics_simulator, in_axes=(0, None, 0, None))(keys[1:], params, prev_states, u)
            lls = vmap(self.ssm.emission_log_prob, in_axes=(None, None, 0, None))(params, y, new_states, u)
            log_weights = jnp.log(weights) + lls
            weights = jnp.exp(log_weights - jnp.max(log_weights))
            new_weights = weights / jnp.sum(weights)
            pre = {'weights': new_weights, 'particles': new_states}
            resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
            weights, new_states, next_key = lax.cond(resample_cond, resample, lambda *args: args, new_weights, new_states, next_key)
            return (weights, ll, new_states, next_key), pre

        keys = jr.split(key, num_particles + 1)
        particles = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None, 0))((), keys[1:])
        carry = (jnp.ones(num_particles) / num_particles, 0.0, particles, keys[0])
        _, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        return swap_axes_on_values(outputs), 0.0


out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
ps.emissions.cov.value = r * jnp.eye(2)
inputs = mtt.get_inputs(None, T)
seeds = jnp.arange(n_sims)
states, obs = jax.jit(vmap(lambda k: ssm.simulate(k, ps, T, inputs)))(vmap(lambda s: jr.PRNGKey(100 + s))(seeds))
fkeys = vmap(lambda s: jr.PRNGKey(1000 + s))(seeds)

FILTERS = {
    'bpf': (pf.BPF, 0.5), 'apf': (pf.APF, 0.5), 'apf_every': (pf.APF, 2.0), 'bpf_every': (pf.BPF, 2.0), 'bpf_pre': (BPFPre, 0.5),
}


def runner(cls, ess):
    flt = cls(props, ssm)

    def one(o, k, s):
        outs, _ = flt.run(ps, o, M, k, inputs, ess)
        est = jnp.einsum('mt,mtd->td', outs['weights'], outs['particles'])
        return jnp.mean(jnp.sum((est - s) ** 2, axis=-1))
    return jax.jit(vmap(one))


chunk = max(1, min(10, 20000 // M))
res = {}
selected = os.environ.get('FILTERS', ','.join(FILTERS)).split(',')
for name, (cls, ess) in [(k, v) for k, v in FILTERS.items() if k in selected]:
    run, t0 = runner(cls, ess), time.time()
    res[name] = onp.concatenate([onp.array(run(obs[i:i + chunk], fkeys[i:i + chunk], states[i:i + chunk])) for i in range(0, n_sims, chunk)])
    x = res[name]
    print(f'a={acc:g} sigma2={r:g} M={M} {name:9s} n={len(x)}  mean {x.mean():9.4g} sd {x.std(ddof=1):9.4g}  median {onp.median(x):9.4g}'
          f'  lost(>1) {(x > 1).sum():3d}  lost(>10) {(x > 10).sum():3d}  [{time.time() - t0:.0f} s]', flush=True)
onp.savez(f'rerun_a{acc:g}_r{r:g}_M{M}.npz', **res)
