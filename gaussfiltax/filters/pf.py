from abc import abstractmethod
from typing import Optional
import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore
from jax import lax, vmap, debug # type: ignore
from jaxtyping import Float, Int, Array # type: ignore
from util.sample import resample, resample2
from util.param import from_conditional, log_prior
from functools import partial # type: ignore
from parameters import ParamSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN #type: ignore


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y, 0)) if x is None else x
MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))
def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))

class ParticleFilter:

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    @abstractmethod
    def run(self, params, emissions, num_particles, key, inputs):
        """
        Run the particle filter.
        Args:
            params: Parameters of the nonlinear state space model.
            emissions: Emissions.
            num_particles: Number of particles.
            rng_key: Random number generator key.
            inputs: Inputs. 

        Returns:
            Posterior particle filtered.
        """
        pass

class BPF(ParticleFilter):

    def __init__(self, props, ssm):
        super().__init__(props, ssm)
        self.xparam = None
        self.time = None

    def run(self,
            params: ParamSSM, 
            observations: Float[Array, "..."],
            num_particles: Int,
            key: jr.PRNGKey = jr.PRNGKey(0),
            inputs: Optional[Float[Array, "..."]] = None, 
            ess_threshold: float = 0.5
           ):

        r"""
        Bootstrap particle filter for the nonlinear state space model.
        Args:
            params: Parameters of the nonlinear state space model.
            emissions: Emissions.
            num_particles: Number of particles.
            rng_key: Random number generator key.
            inputs: Inputs. 

        Returns:
            Posterior particle filtered.
        """

        num_timesteps = observations.shape[0]

        # Dynamics and emission functions
        inputs = inputs if inputs is not None else jnp.zeros((num_timesteps, self.ssm.input_dim))
        
        def _step(carry, t):
            weights, ll, prev_states, key = carry

            # Get parameters and inputs for time index t
            u = inputs[t]
            y = observations[t]
            
            # Sample new particles 
            keys = jr.split(key, num_particles+1)
            next_key = keys[0]
            get_new_states = vmap(self.ssm.dynamics_simulator, in_axes=(0,None,0,None))
            new_states = get_new_states(keys[1:], params, prev_states, u)

            # Compute weights 
            get_lps = vmap(self.ssm.emission_log_prob, in_axes=(None,None,0,None))
            lls = get_lps(params, y, new_states, u)
            # Normalize in the log domain, with the previous weights inside the max-shift. If only the likelihood is
            # shifted, the product with the weights can be zero for every particle, and then the weights are NaN.
            log_weights = jnp.log(weights) + lls
            log_weights_max = jnp.max(log_weights)
            weights = jnp.exp(log_weights - log_weights_max)
            ll += jnp.log(jnp.sum(weights)) + log_weights_max
            u_weights = weights
            nc = jnp.sum(weights)
            new_weights = weights / nc

            # Resample if necessary
            resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
            weights, new_states, next_key = lax.cond(resample_cond, resample, lambda *args: args, new_weights, new_states, next_key)

            outputs = {
                'weights': weights,
                'particles': new_states,
                'u_weights': u_weights
            }

            carry = (weights, ll, new_states, next_key)

            return carry, outputs
        
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        weights = jnp.ones(num_particles) / num_particles
        map_sample = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None,0))
        particles = map_sample((), keys[1:])

        carry = (weights, 0.0, particles, next_key)

        out_carry, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        ll = out_carry[1]
        outputs = swap_axes_on_values(outputs)
    
        return outputs, ll
    
    def logdensity_fn(self,
                      cond_params, 
                      key, 
                      emissions, 
                      num_prt, 
                      num_iters):
        
        params = from_conditional(self.xparam, self.props, cond_params)
        lps = []

        for _ in range(num_iters):

            key, subkey = jr.split(key)
            _, lp = self.bpf(params, emissions, num_prt, subkey)
            lp += log_prior(cond_params, self.props)
            lps.append(lp)

        return jnp.mean(jnp.array(lps))


class APF(ParticleFilter):

    def __init__(self, props, ssm):
        super().__init__(props, ssm)
        self.xparam = None
        self.time = None

    def run(self,
            params: ParamSSM, 
            observations: Float[Array, "..."],
            num_particles: Int,
            key: jr.PRNGKey = jr.PRNGKey(0),
            inputs: Optional[Float[Array, "..."]] = None,
            ess_threshold: float = 0.5
           ):

        num_timesteps = observations.shape[0]
        inputs = inputs if inputs is not None else jnp.zeros((num_timesteps, self.ssm.input_dim))

        dynamics_fn = partial(self.ssm.dynamics_function, params)

        def _step(carry, t):
            weights, ll, prev_states, key = carry

            # Get parameters and inputs for time index t
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            u = inputs[t]
            y = observations[t]

            keys = jr.split(key, num_particles + 1)
            key = keys[0]

            # Step 1: Compute predicted mean state
            get_pred = vmap(dynamics_fn, in_axes=(0, None, None))  
            pred_states = get_pred(prev_states, q0, u)

            # Step 2: Compute auxiliary weights
            get_aux_loglik = vmap(self.ssm.emission_log_prob, in_axes=(None, None, 0, None))
            aux_loglik = get_aux_loglik(params, y, pred_states, u)
            aux_log = jnp.log(weights) + aux_loglik
            aux_max = jnp.max(aux_log)
            aux_unnorm = jnp.exp(aux_log - aux_max)
            aux_sum = jnp.sum(aux_unnorm)
            aux_ws = aux_unnorm / aux_sum

            # Step 3: Resample ancestors if the ESS of the preweights is low, otherwise keep them
            key, subkey = jr.split(key)
            resample_cond = 1.0 / jnp.sum(jnp.square(aux_ws)) < ess_threshold * num_particles
            resampled_indices = jr.choice(subkey, num_particles, (num_particles,), p=aux_ws, replace=True)
            ancestor_indices = jnp.where(resample_cond, resampled_indices, jnp.arange(num_particles))
            ancestors = prev_states[ancestor_indices]

            # Step 4: Propagate new particles from ancestors
            get_prop = vmap(self.ssm.dynamics_simulator, in_axes=(0, None, 0, None))
            prop_particles = get_prop(keys[1:], params, ancestors, u)

            # Step 5: Compute true likelihoods and correct weights
            get_loglik = vmap(self.ssm.emission_log_prob, in_axes=(None, None, 0, None))
            logliks = get_loglik(params, y, prop_particles, u)
            # If the ancestors were resampled, w ~ g / m_a. Otherwise this is the BPF update, w ~ w_prev * g.
            log_weights = jnp.where(resample_cond, logliks - aux_loglik[ancestor_indices], jnp.log(weights) + logliks)
            log_weights_max = jnp.max(log_weights)
            ws = jnp.exp(log_weights - log_weights_max)
            sum_ws = jnp.sum(ws)

            # APF incremental likelihood (resampled ancestors):
            #   p_hat(y_t|y_{1:t-1}) = [sum_i w_{t-1}^i m_i] * (1/N) * sum_j [g(y_t|x_t^j) / m_{a_j}]
            # BPF incremental likelihood (kept ancestors): p_hat(y_t|y_{1:t-1}) = sum_j w_{t-1}^j g(y_t|x_t^j)
            # aux_max and log_weights_max undo the max-shifts used for numerical stability.
            ll_apf = jnp.log(aux_sum) + aux_max + jnp.log(sum_ws / num_particles) + log_weights_max
            ll_bpf = jnp.log(sum_ws) + log_weights_max
            ll += jnp.where(resample_cond, ll_apf, ll_bpf)
            u_weights = ws
            new_weights = ws / sum_ws

            # Ancestors are resampled only once per step, in Step 3. A second resampling pass
            # adds variance and often degrades performance relative to BPF.
            next_key = key

            carry = (new_weights, ll, prop_particles, next_key)
            outputs = {
                'weights': new_weights,
                'particles': prop_particles,
                'u_weights': u_weights  
            }

            return carry, outputs

        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        weights = jnp.ones(num_particles) / num_particles
        map_sample = vmap(MVN(loc=params.initial.mean.value, covariance_matrix=params.initial.cov.value).sample, in_axes=(None,0))
        particles = map_sample((), keys[1:])

        carry = (weights, 0.0, particles, next_key)
        out_carry, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        ll = out_carry[1]
        outputs = swap_axes_on_values(outputs)

        return outputs, ll
