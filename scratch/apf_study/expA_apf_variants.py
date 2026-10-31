import sys
sys.path.insert(0, '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from simulators import mtt
from util.param import sample_prior
from filters import pf
from filters.pf import _get_params, swap_axes_on_values, MVN

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
T = 500


class APFAdaptive(pf.ParticleFilter):
    """APF that resamples ancestors only when the ESS of the first-stage weights is below
    ess_threshold * M. Otherwise ancestors are the identity and the weights are the BPF update."""

    def run(self, params, observations, num_particles, key, inputs=None, ess_threshold=0.5):
        T_ = observations.shape[0]
        dynamics_fn = lambda x, q, u: self.ssm.dynamics_function(params, x, q, u)

        def _step(carry, t):
            weights, ll, prev_states, key = carry
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            u, y = inputs[t], observations[t]
            keys = jr.split(key, num_particles + 2)
            key, subkey = keys[0], keys[1]

            pred = vmap(dynamics_fn, in_axes=(0, None, None))(prev_states, q0, u)
            aux_ll = vmap(self.ssm.emission_log_prob, in_axes=(None, None, 0, None))(params, y, pred, u)
            log_lam = jnp.log(weights) + aux_ll
            lam_norm = jnp.exp(log_lam - jnp.max(log_lam))
            lam = lam_norm / jnp.sum(lam_norm)
            do_resample = 1.0 / jnp.sum(lam ** 2) < ess_threshold * num_particles

            idx_res = jr.choice(subkey, num_particles, (num_particles,), p=lam, replace=True)
            idx = jnp.where(do_resample, idx_res, jnp.arange(num_particles))
            ancestors = prev_states[idx]
            new_states = vmap(self.ssm.dynamics_simulator, in_axes=(0, None, 0, None))(keys[2:], params, ancestors, u)
            ll_new = vmap(self.ssm.emission_log_prob, in_axes=(None, None, 0, None))(params, y, new_states, u)
            # resampled: w ∝ g / m_a ; not resampled: w ∝ w_prev * g   (BPF update)
            log_w = jnp.where(do_resample, ll_new - aux_ll[idx], jnp.log(weights) + ll_new)
            log_w = log_w - jnp.max(log_w)
            w = jnp.exp(log_w)
            new_weights = w / jnp.sum(w)
            outputs = {'weights': new_weights, 'particles': new_states, 'u_weights': w}
            return (new_weights, ll, new_states, key), outputs

        keys = jr.split(key, num_particles + 1)
        weights = jnp.ones(num_particles) / num_particles
        particles = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample,
                         in_axes=(None, 0))((), keys[1:])
        out_carry, outputs = lax.scan(_step, (weights, 0.0, particles, keys[0]), jnp.arange(T_))
        return swap_axes_on_values(outputs), out_carry[1]


out = mtt.setup(4, 2, 0)
ssm, props = out['ssm'], out['props']
true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
inputs = mtt.get_inputs(None, T)

variants = [
    ('BPF  (resample if ESS<0.5M)', pf.BPF, 0.5),
    ('BPF  (resample every step) ', pf.BPF, 2.0),
    ('APF  (current: every step) ', pf.APF, 0.5),
    ('APF  (resample if ESS<0.5M)', APFAdaptive, 0.5),
]
res = {v[0]: [] for v in variants}
for seed in range(n_seeds):
    states, obs = ssm.simulate(jr.PRNGKey(100 + seed), true_ps, T, inputs)
    for name, cls, ess in variants:
        o, _ = cls(props, ssm).run(true_ps, obs, M, jr.PRNGKey(seed), inputs, ess)
        w, x = o['weights'], o['particles']
        w = w / jnp.sum(w, axis=0, keepdims=True)
        mean = jnp.einsum('it,itd->td', w, x)
        res[name].append(float(jnp.mean(jnp.sum((mean - states) ** 2, axis=-1))))
    print(seed, {k: round(v[-1], 3) for k, v in res.items()}, flush=True)

print(f'\nM={M}, {n_seeds} runs, weighted-mean MSE')
for name, v in res.items():
    a = onp.array(v)
    print(f'{name}: mean={a.mean():8.3f}  median={onp.median(a):7.3f}  max={a.max():8.3f}  runs with MSE>10: {(a > 10).sum()}/{len(a)}')
