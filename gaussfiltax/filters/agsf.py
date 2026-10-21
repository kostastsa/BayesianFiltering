import jax.numpy as jnp #type: ignore
import jax.random as jr #type: ignore
import jax.tree_util as jtu #type: ignore
from functools import partial #type: ignore
from jax import lax, vmap, jacfwd, jacrev, jit, debug #type: ignore
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN #type: ignore
from typing import List, Optional, NamedTuple, Union
from util.numerics import _resample, _resample2, psd_solve, refit_gmm
from containers import GaussianComponent, GaussianSum, _branches_from_tree1, _branches_from_tree2
import containers
from jaxtyping import Array, Float, Int #type: ignore
from .steps import (
    _sqrt_condition_on,
    _sqrt_predict,
    _ukf_sqrt_condition_on_nonadditive,
    _ukf_sqrt_predict_nonadditive,
)

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

def prop_cov_selection(m, P, jacobian, hessian_tensor, bias, u, alpha):

    cov = alpha * P

    return cov, alpha

def auto_cov_selection(m, P, jacobian, hessian_tensor, bias, u, alpha):

    hessian = hessian_tensor(m, bias, u)
    J = jacobian(m, bias, u)
    rho = jnp.minimum(.95, alpha * jnp.trace(P @ J.T @ J) / jnp.sum(jnp.trace((hessian) @ P, axis1=1, axis2=2)**2))
    cov = rho * P

    return cov, rho

class LAGSF():

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
        autocov1,
        r1,
        autocov2,
        r2,
        M,
        N,
        L,
        num_iter: int = 1,
        inputs: Optional[Float[Array, '...']] = None
        ):
        r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
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
        F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
        inputs = _process_input(inputs, num_timesteps)

        # Autocovariance functions
        if autocov1 == 'prop':

            autocov1 = prop_cov_selection

        elif autocov1 == 'auto':

            autocov1 = auto_cov_selection

        if autocov2 == 'prop':

            autocov2 = prop_cov_selection

        elif autocov2 == 'auto':

            autocov2 = auto_cov_selection

        def _step(carry, t):

            filtered_means, filtered_covs, weights, key = carry
            
            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]
            state_dim = filtered_covs[0].shape[0]

            # Autocov before predict
            Deltas, rho1s = vmap(autocov1, in_axes=(0, 0, None, None, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, q0, u, r1)

            # Compute the z-sample
            key, subkey = jr.split(key)
            iid_sample = jr.normal(subkey, shape=(M, state_dim, N))
            mlt_matrices = vmap(jnp.linalg.cholesky)(filtered_covs-Deltas)
            centered_sample = vmap(jnp.matmul, in_axes=(0,0))(mlt_matrices, iid_sample)
            z_sample = jnp.expand_dims(filtered_means, axis=2) + centered_sample
            z_sample = jnp.swapaxes(z_sample, 1, 2)

            # Predict (square root EKF)
            # Convert Deltas (covariances) to square roots
            delta_sqrts = vmap(jnp.linalg.cholesky)(Deltas)
            map_predict = lambda means, S: vmap(_sqrt_predict, in_axes=(0,None,None,None,None,None,None,None))(means, S, dynamics_fn, F_x, F_q, Q, q0, u)
            predicted_means, predicted_sqrts, _ = vmap(map_predict, in_axes=(0,0))(z_sample, delta_sqrts)
            predicted_weights = jnp.tile(weights, (N,1)).T / N

            predicted_means = jnp.reshape(predicted_means, (M * N, state_dim))
            predicted_sqrts = jnp.reshape(predicted_sqrts, (M * N, state_dim, state_dim))
            predicted_weights = jnp.reshape(predicted_weights, (M * N,))

            # Prior inflation (apply in covariance space then back to sqrt)
            predicted_covs = vmap(lambda S: S @ S.T)(predicted_sqrts)
            predicted_covs = vmap(lambda cov: cov + 0.01 * jnp.eye(state_dim))(predicted_covs)
            predicted_sqrts = vmap(jnp.linalg.cholesky)(predicted_covs)

            # Autocov before update
            # Compute Lambdas using covariance space; predicted_covs reconstructed above
            Lambdas, rho2s = vmap(autocov2, in_axes=(0, 0, None, None, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, r0, u, r2)

            # Compute the s-sample
            key, subkey = jr.split(key)
            s_iid_sample = jr.normal(subkey, shape=(M * N, state_dim, L))
            s_mlt_matrices = vmap(jnp.linalg.cholesky)(predicted_covs - Lambdas)
            s_centered_sample = vmap(jnp.matmul, in_axes=(0,0))(s_mlt_matrices, s_iid_sample)
            s_sample = jnp.expand_dims(predicted_means, axis=2) + s_centered_sample
            s_sample = jnp.swapaxes(s_sample, 1, 2)

            # Update (square root EKF)
            lambda_sqrts = vmap(jnp.linalg.cholesky)(Lambdas)
            map_condition_on = lambda means, S: vmap(_sqrt_condition_on, in_axes=(0,None,None,None,None,None,None,None,None))(means, S, emission_fn, H_x, H_r, R, r0, u, y)
            lls, updated_means, updated_sqrts, _, _ = vmap(map_condition_on, in_axes=(0,0))(s_sample, lambda_sqrts)
            updated_weights = jnp.tile(predicted_weights, (L, 1)).T / L

            nans = jnp.isnan(lls)
            lls = jnp.where(nans, -jnp.inf, lls)
            lls -= jnp.max(lls)
            ls = jnp.exp(lls)
            weights = jnp.multiply(ls, updated_weights)
            weights /= jnp.sum(weights)

            updated_means = jnp.reshape(updated_means, (M * N * L, state_dim))
            updated_sqrts = jnp.reshape(updated_sqrts, (M * N * L, state_dim, state_dim))
            weights = jnp.reshape(weights, (M * N * L,))

            # Resampling 
            key, subkey = jr.split(key)
            resampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(M, ), p=weights)
            filtered_means =  jnp.take(updated_means, resampled_idx, axis=0)
            filtered_sqrts = jnp.take(updated_sqrts, resampled_idx, axis=0)
            weights = jnp.ones(shape=(M, )) /M

            # Convert to covariances for output and add small inflation
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrts)
            filtered_covs = vmap(lambda cov: cov + 0.01 * jnp.eye(state_dim))(filtered_covs)

            # Build carry and output states
            carry = (filtered_means, filtered_covs, weights, key)
            
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            aux_outputs = {
                "Deltas": Deltas,
                "Lambdas": Lambdas,
                "rho1s": rho1s,
                "rho2s": rho2s                
            }

            return carry, (outputs, aux_outputs)
        
        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(M, key)
        initial_covs = jnp.array([params.initial.cov.value for i in range(M)])
        initial_weights = jnp.ones(shape=(M,)) / M

        carry = (initial_means, initial_covs, initial_weights, key)

        carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)

        return outputs, aux_outputs

class UAGSF():

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
        autocov1,
        r1,
        autocov2,
        r2,
        M,
        N,
        L,
        num_iter: int = 1,
        alpha: Float = 1e-3, 
        beta: Float = 2.0, 
        kappa: Float = 0.0,
        inputs: Optional[Float[Array, '...']] = None
    ):
        r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
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

        # Dynamics and emission functions and their Jacobians
        F_x, H_x = jacfwd(dynamics_fn, argnums=0), jacfwd(emission_fn, argnums=0)
        F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
        inputs = _process_input(inputs, num_timesteps)

        # Autocovariance functions
        if autocov1 == 'prop':

            autocov1 = prop_cov_selection

        elif autocov1 == 'auto':

            autocov1 = auto_cov_selection

        if autocov2 == 'prop':

            autocov2 = prop_cov_selection

        elif autocov2 == 'auto':

            autocov2 = auto_cov_selection

        def _step(carry, t):

            filtered_means, filtered_covs, weights, key = carry
            
            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]
            state_dim = filtered_covs[0].shape[0]

            # Autocov 1
            Deltas, rho1s = vmap(autocov1, in_axes=(0, 0, None, None, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, q0, u, r1)

            # Compute the z-sample
            key, subkey = jr.split(key)
            iid_sample = jr.normal(subkey, shape=(M, state_dim, N))
            mlt_matrices = vmap(jnp.linalg.cholesky)(filtered_covs-Deltas)
            centered_sample = vmap(jnp.matmul, in_axes=(0,0))(mlt_matrices, iid_sample)
            z_sample = jnp.expand_dims(filtered_means, axis=2) + centered_sample
            z_sample = jnp.swapaxes(z_sample, 1, 2)

            # Predict (square root UKF)
            delta_sqrts = vmap(jnp.linalg.cholesky)(Deltas)
            map_predict = lambda means, S: vmap(_ukf_sqrt_predict_nonadditive, in_axes=(0,None,None,None,None,None,None,None,None))(means, S, dynamics_fn, u, Q, alpha, beta, kappa, q0)
            predicted_means, predicted_sqrts = vmap(map_predict, in_axes=(0,0))(z_sample, delta_sqrts)
            predicted_weights = jnp.tile(weights, (N, 1)).T / N

            predicted_means = jnp.reshape(predicted_means, (M * N, state_dim))
            predicted_sqrts = jnp.reshape(predicted_sqrts, (M * N, state_dim, state_dim))
            predicted_weights = jnp.reshape(predicted_weights, (M * N,))

            # Prior inflation
            scale = 0.01
            predicted_covs = vmap(lambda S: S @ S.T)(predicted_sqrts)
            predicted_covs = predicted_covs + scale * jnp.eye(state_dim)[None, :, :]
            predicted_sqrts = vmap(jnp.linalg.cholesky)(predicted_covs)

            # Autocov before update
            Lambdas, rho2s = vmap(autocov2, in_axes=(0, 0, None, None, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, r0, u, r2) 

            # Compute the s-sample
            key, subkey = jr.split(key)
            s_iid_sample = jr.normal(subkey, shape=(M*N, state_dim, L))
            s_mlt_matrices = vmap(jnp.linalg.cholesky)(predicted_covs - Lambdas)
            s_centered_sample = vmap(jnp.matmul, in_axes=(0,0))(s_mlt_matrices, s_iid_sample)
            s_sample = jnp.expand_dims(predicted_means, axis=2) + s_centered_sample
            s_sample = jnp.swapaxes(s_sample, 1, 2)

            # Update (square root UKF)
            lambda_sqrts = vmap(jnp.linalg.cholesky)(Lambdas)
            map_condition_on = lambda means, S: vmap(_ukf_sqrt_condition_on_nonadditive, in_axes=(0,None,None,None,None,None,None,None,None,None))(means, S, emission_fn, R, u, y, alpha, beta, kappa, r0)
            lls, updated_means, updated_sqrts = vmap(map_condition_on, in_axes=(0,0))(s_sample, lambda_sqrts)
            updated_weights = jnp.tile(predicted_weights, (L,1)).T / L

            lls -= jnp.max(lls)
            ls = jnp.exp(lls)
            weights = jnp.multiply(ls, updated_weights)
            weights /= jnp.sum(weights)

            updated_means = jnp.reshape(updated_means, (M*N*L, state_dim))
            updated_sqrts = jnp.reshape(updated_sqrts, (M*N*L, state_dim, state_dim))
            weights = jnp.reshape(weights, (M*N*L,))

            # Resampling 
            key, subkey = jr.split(key)
            resampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(M, ), p=weights)
            filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
            filtered_sqrts = jnp.take(updated_sqrts, resampled_idx, axis=0)
            weights = jnp.ones(shape=(M,)) / M
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrts)

            # Build carry and output states
            carry = (filtered_means, filtered_covs, weights, key)
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            aux_outputs = {
                "Deltas": Deltas,
                "Lambdas": Lambdas,
                "rho1s": rho1s,
                "rho2s": rho2s
            }   

            return carry, (outputs, aux_outputs)

        
        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(M, key)
        initial_covs = jnp.array([params.initial.cov.value for i in range(M)])
        initial_weights = jnp.ones(shape=(M,)) / M

        carry = (initial_means, initial_covs, initial_weights, key)

        carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)
        
        return outputs, aux_outputs

