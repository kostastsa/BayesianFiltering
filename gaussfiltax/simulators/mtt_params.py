import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as dist # type: ignore


def _init_vals(state_dim, emission_dim, input_dim, sigma2=25 * 1e-1):

    initial_mean = 1.0 * jnp.array([-0.05, 0.001, 0.7, -0.05])
    initial_covariance = 1.0 * jnp.array([[0.1, 0.0, 0.0, 0.0],
                                          [0.0, 0.005, 0.0, 0.0],
                                          [0.0, 0.0, 0.1, 0.0],
                                          [0.0, 0.0, 0.0, 0.01]])

    dynamics_bias = jnp.zeros(2)
    dynamics_covariance = jnp.eye(2) * 1e-6

    emission_bias = jnp.zeros(emission_dim)
    emission_covariance = jnp.eye(emission_dim) * sigma2

    init_vals = [[initial_mean, initial_covariance],
                    [dynamics_bias, dynamics_covariance],
                    [emission_bias, emission_covariance]]

    return init_vals

def _param_dists(state_dim, emission_dim, input_dim):

    # initial_mean 
    initial_mean_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # initial_covariance
    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # dynamics_bias
    dynamics_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # dynamics_covariance
    dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=1.0*jnp.ones(m))

    # emission_bias
    emission_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(emission_dim), scale_diag=jnp.ones(emission_dim))

    # emission_covariance
    l = emission_dim * (emission_dim + 1) // 2
    emission_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(l), scale_diag=1.0*jnp.ones(l))

    param_dists = [[initial_mean_dist, initial_covariance_dist],
                    [dynamics_bias_dist, dynamics_covariance_dist],
                    [emission_bias_dist, emission_covariance_dist]]
    
    return param_dists

