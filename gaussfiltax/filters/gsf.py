import jax.numpy as jnp #type: ignore
import jax.random as jr #type: ignore
from functools import partial #type: ignore
from jax import lax, vmap, jacfwd, jit #type: ignore
from jax.scipy.linalg import cholesky #type: ignore
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN #type: ignore
from typing import Optional
from jaxtyping import Array, Float #type: ignore
from .steps import _condition_on, _ukf_condition_on_nonadditive, _predict, _ukf_predict_nonadditive, _sqrt_condition_on, _sqrt_predict, _ukf_sqrt_predict_nonadditive, _ukf_sqrt_condition_on_nonadditive

from parameters import ParamSSM
import time

import matplotlib.pyplot as plt #type: ignore

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y, 0)) if x is None else x
MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))
def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


class LGSF():


    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def run(
        self,
        params: ParamSSM,
        emissions: Float[Array, '...'],
        num_components: int = 1,
        num_iter: int = 1,
        rng_key = jr.PRNGKey(0),
        inputs: Optional[Float[Array, '...']] = None
    ):
        r"""Run an Gaussian sum filter, which is a mixture of (iterated) extended Kalman filters to produce the
        marginal likelihood and filtered state estimates.
        Args:
            params: model parameters.
            emissions: observation sequence.
            num_components: number of components in Gaussian sum approximation.
            num_iter: number of linearizations around posterior for update step (default 1).
            inputs: optional array of inputs.
            output_fields: list of fields to return in posterior object.
                These can take the values "filtered_means", "filtered_covariances",
                "predicted_means", "predicted_covariances", and "weights".
        Returns:
            post: posterior object.
        """
        num_timesteps = len(emissions)

        dynamics_fn = partial(self.ssm.dynamics_function, params)
        emission_fn = partial(self.ssm.emission_function, params)

        # Jacobian calculation
        F_x, H_x = jacfwd(dynamics_fn, argnums=0), jacfwd(emission_fn, argnums=0)
        F_q, H_r = jacfwd(dynamics_fn, argnums=1), jacfwd(emission_fn, argnums=1)
        inputs = _process_input(inputs, num_timesteps)

        def _step(carry, t):

            weights, pred_means, pred_sqrt_covs = carry

            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]

            # Condition on this emission using square root filter
            lls, filtered_means, filtered_sqrt_covs, _, _ = vmap(_sqrt_condition_on, in_axes=(0, 0, None, None, None, None, None, None, None))(pred_means, pred_sqrt_covs, emission_fn, H_x, H_r, R, r0, u, y)

            # Compute weights
            lls -= jnp.max(lls)
            loglik_weights = jnp.exp(lls)
            weights = jnp.multiply(loglik_weights, weights)
            weights /= jnp.sum(weights)

            # Predict the next state using square root filter
            pred_means, pred_sqrt_covs, _ = vmap(_sqrt_predict, in_axes=(0, 0, None, None, None, None, None, None))(filtered_means, filtered_sqrt_covs, dynamics_fn, F_x, F_q, Q, q0, u)

            # Convert square root covariances to full covariances for output
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrt_covs)
            pred_covs = vmap(lambda S: S @ S.T)(pred_sqrt_covs)

            # Build carry and output states
            carry = (weights, pred_means, pred_sqrt_covs)
            outputs = {
                "means": filtered_means,
                "covariances": filtered_covs,
                "predicted_means": pred_means,
                "predicted_covariances": pred_covs,
                "weights": weights
            }

            return carry, outputs

        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(num_components, rng_key)
        # Initialize with square root covariances
        initial_sqrt_cov = cholesky(params.initial.cov.value, lower=True)
        initial_sqrt_covs = jnp.array([initial_sqrt_cov for i in range(num_components)])
        carry = (jnp.ones(num_components)/num_components, initial_means, initial_sqrt_covs)

        _, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)

        return outputs

class UGSF():

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def run(
        self,
        key: jr.PRNGKey,
        params: ParamSSM,
        emissions: Float[Array, '...'],
        num_components: int = 1,
        num_iter: int = 1,
        alpha: Float = 1e-3, 
        beta: Float = 2.0, 
        kappa: Float = 0.0,
        inputs: Optional[Float[Array, '...']] = None
        ):

        r"""Run a Gaussian sum filter, which is a mixture of unscented Kalman filters to produce the
        marginal likelihood and filtered state estimates. This version uses square root covariances
        for improved numerical stability.
        Args:
            key: PRNG key for random number generation.
            params: model parameters.
            emissions: observation sequence.
            num_components: number of components in Gaussian sum approximation.
            num_iter: number of linearizations around posterior for update step (default 1).
            alpha: UKF spread parameter (default 1e-3).
            beta: UKF parameter for incorporating prior knowledge (default 2.0).
            kappa: UKF secondary scaling parameter (default 0.0).
            inputs: optional array of inputs.
        Returns:
            outputs: dictionary containing filtered means, covariances, and weights.
        """

        num_timesteps = len(emissions)
        inputs = _process_input(inputs, num_timesteps)

        dynamics_fn = partial(self.ssm.dynamics_function, params)
        emission_fn = partial(self.ssm.emission_function, params)

        def _step(carry, t):
            weights, pred_means, pred_sqrt_covs = carry

            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]

            # Condition on this emission using square root UKF
            lls, filtered_means, filtered_sqrt_covs = vmap(_ukf_sqrt_condition_on_nonadditive, in_axes=(0, 0, None, None, None, None, None, None, None, None))(pred_means, pred_sqrt_covs, emission_fn, R, u, y, alpha, beta, kappa, r0)

            # Compute weights
            lls -= jnp.max(lls)
            loglik_weights = jnp.exp(lls)
            weights = jnp.multiply(loglik_weights, weights)
            weights /= jnp.sum(weights)

            # Predict the next state using square root UKF
            pred_means, pred_sqrt_covs = vmap(_ukf_sqrt_predict_nonadditive, in_axes=(0, 0, None, None, None, None, None, None, None))(filtered_means, filtered_sqrt_covs, dynamics_fn, u, Q, alpha, beta, kappa, q0)

            # Convert square root covariances to full covariances for output
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrt_covs)

            # Build carry and output states
            carry = (weights, pred_means, pred_sqrt_covs)
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            return carry, outputs

        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(num_components, key)
        # Initialize with square root covariances
        initial_sqrt_cov = cholesky(params.initial.cov.value, lower=True)
        initial_sqrt_covs = jnp.array([initial_sqrt_cov for i in range(num_components)])
        carry = (jnp.ones(num_components) / num_components, initial_means, initial_sqrt_covs)

        carry, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)

        return outputs
