import jax.numpy as jnp
import jax.random as jr
#import jax
from jax import lax, vmap, jacfwd, debug, device_put
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
import gaussfiltax.utils as utils
from jaxtyping import Array, Float

from dynamax.utils.utils import psd_solve
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
import time

import matplotlib.pyplot as plt

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x
_MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))
def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


class PosteriorGaussianSumFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.
    :param weights: weights of the Gaussian components
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    """
    weights: Optional[Float[Array, "num_components ntime"]] = None
    filtered_means: Optional[Float[Array, "num_components ntime state_dim"]] = None
    filtered_covariances: Optional[Float[Array, "num_components ntime state_dim state_dim"]] = None
    predicted_means: Optional[Float[Array, "num_components ntime state_dim"]] = None
    predicted_covariances: Optional[Float[Array, "num_components ntime state_dim state_dim"]] = None


def _predict(m, P, f, F, Q, u):
    r"""Predict next mean and covariance using first-order additive EKF
        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)
    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.
    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    F_x = F(m, u)
    mu_pred = f(m, u)
    Sigma_pred = F_x @ P @ F_x.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, h, H, R, u, y, num_iter):
    r"""Condition a Gaussian potential on a new observation.
       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**
    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.
     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def gaussian_sum_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission dim"],
    num_components: int = 1,
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input dim"]] = None,
    output_fields: Optional[List[str]]=["weights", "filtered_means", "filtered_covariances", "predicted_means", "predicted_covariances"],
) -> PosteriorGaussianSumFiltered:
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

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F, H = jacfwd(f), jacfwd(h)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    f_vec, h_vec, F_vec, H_vec = (vmap(fn, in_axes=(0, None)) for fn in (f, h, F, H)) # vmap over components, component axis is 0
    inputs = _process_input(inputs, num_timesteps)
    MVN_log_prob_vec = vmap(_MVN_log_prob, in_axes=(0, 0, None))

    def _step(carry, t):
        weights, pred_means, pred_covs = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihoods
        H_x = H_vec(pred_means, u)
        ml_means = h_vec(pred_means, u)
        ml_covs = jnp.array([H_x[i] @ pred_covs[i] @ H_x[i].T + R for i in range(num_components)]) #TODO: check is this can be improved using an array generation approach
        lls = MVN_log_prob_vec(ml_means, ml_covs, y)
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights, weights)
        weights /= jnp.sum(weights)

        # Condition on this emission
        filtered_means, filtered_covs = vmap(_condition_on, in_axes=(0, 0, None, None, None, None, None, None))(pred_means, pred_covs, h, H, R, u, y, num_iter)

        # Predict the next state
        pred_means, pred_covs = vmap(_predict, in_axes=(0, 0, None, None, None, None))(filtered_means, filtered_covs, f, F, Q, u)

        # Build carry and output states
        carry = (weights, pred_means, pred_covs)
        outputs = {
            "filtered_means": filtered_means,
            "filtered_covariances": filtered_covs,
            "predicted_means": pred_means,
            "predicted_covariances": pred_covs,
            "weights": weights
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components, jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components)])
    carry = (jnp.ones(num_components)/num_components, initial_means, initial_covs)

    (lls, *_), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       **outputs,
    )

    return posterior_filtered
