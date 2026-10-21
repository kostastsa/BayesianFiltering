import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as dist # type: ignore


def _init_vals(state_dim, emission_dim, input_dim):
        
    initial_mean = jnp.zeros(state_dim)
    initial_covariance = jnp.eye(state_dim) * 0.1

    dynamics_weights  = 0.9 * jnp.eye(state_dim)
    dynamics_bias = jnp.zeros(state_dim)
    dynamics_input_weights = jnp.zeros((state_dim, input_dim))
    dynamics_covariance = jnp.eye(state_dim) * 0.1

    emission_weights = jnp.eye(emission_dim, state_dim)
    emission_bias = jnp.zeros(emission_dim)
    emission_input_weights = jnp.zeros((emission_dim, input_dim))
    emission_covariance = jnp.eye(emission_dim) * 0.1

    init_vals = [[initial_mean, initial_covariance],
                    [dynamics_weights, dynamics_bias, dynamics_input_weights, dynamics_covariance],
                    [emission_weights, emission_bias, emission_input_weights, emission_covariance]]

    return init_vals

def _param_dists(state_dim, emission_dim, input_dim):

    # initial_mean 
    initial_mean_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # initial_covariance
    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # dynamics_weights
    mu = 0.8 * jnp.eye(state_dim)
    col_cov = jnp.eye(state_dim) * 0.01
    row_cov = jnp.eye(state_dim) * 0.01
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    dynamics_weights_dist = dist.MatrixNormal(
    loc=mu,
    scale_tril_row=scale_row,
    scale_tril_column=scale_column)

    # dynamics_bias
    dynamics_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag= 1.0 * jnp.ones(state_dim))

    # dynamics_input_weights
    mu = 0.0 * jnp.eye(state_dim, input_dim)
    col_cov = jnp.eye(input_dim) * 0.001
    row_cov = jnp.eye(state_dim) * 0.001
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    dynamics_input_weights_dist = dist.MatrixNormal(
    loc=mu,
    scale_tril_row=scale_row,
    scale_tril_column=scale_column)

    # dynamics_covariance
    dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=1.0*jnp.ones(m))

    # emission_weights
    mu = 1.0 * jnp.eye(emission_dim, state_dim)
    col_cov = jnp.eye(state_dim) * 0.1
    row_cov = jnp.eye(emission_dim) * 0.1
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    emission_weights_dist = dist.MatrixNormal(
    loc=mu,
    scale_tril_row=scale_row,
    scale_tril_column=scale_column)

    # emission_bias
    emission_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(emission_dim), scale_diag= 1.0 * jnp.ones(emission_dim))

    # emission_input_weights
    mu = 0.0 * jnp.eye(emission_dim, input_dim)
    col_cov = jnp.eye(input_dim) * 0.001
    row_cov = jnp.eye(emission_dim) * 0.001
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    emission_input_weights_dist = dist.MatrixNormal(
    loc=mu,
    scale_tril_row=scale_row,
    scale_tril_column=scale_column)

    # emission_covariance
    l = emission_dim * (emission_dim + 1) // 2
    emission_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(l), scale_diag=1.0*jnp.ones(l))

    param_dists = [[initial_mean_dist, initial_covariance_dist],
                    [dynamics_weights_dist, dynamics_bias_dist, dynamics_input_weights_dist, dynamics_covariance_dist],
                    [emission_weights_dist, emission_bias_dist, emission_input_weights_dist, emission_covariance_dist]]
    
    return param_dists

