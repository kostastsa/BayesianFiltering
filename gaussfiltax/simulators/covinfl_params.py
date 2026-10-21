import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as dist # type: ignore


def _init_vals(state_dim, emission_dim, input_dim, a, b):
        
    initial_mean = jnp.zeros(state_dim)
    initial_covariance = jnp.eye(state_dim)

    dynamics_coeff = jnp.array([a])
    dynamics_bias = jnp.zeros(state_dim)
    dynamics_covariance = jnp.eye(state_dim)

    emission_coeff = jnp.array([b])
    emission_bias = jnp.zeros(emission_dim)
    emission_covariance = jnp.eye(emission_dim) 

    init_vals = [[initial_mean, initial_covariance],
                    [dynamics_coeff, dynamics_bias, dynamics_covariance],
                    [emission_coeff, emission_bias, emission_covariance]]

    return init_vals

def _param_dists(state_dim, emission_dim, input_dim):

    # initial_mean 
    initial_mean_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # initial_covariance
    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))

    # dynamics_coeff
    dynamics_coeff_dist = tfd.Normal(loc=10.0, scale=1.0)

    # dynamics_bias
    dynamics_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim))

    # dynamics_covariance
    dynamics_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=1.0*jnp.ones(m))

    # emission_coeff
    emission_coeff_dist = tfd.Normal(loc=1.0, scale=0.1)

    # emission_bias
    emission_bias_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(emission_dim), scale_diag=jnp.ones(emission_dim))

    # emission_covariance
    l = emission_dim * (emission_dim + 1) // 2
    emission_covariance_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(l), scale_diag=1.0*jnp.ones(l))

    param_dists = [[initial_mean_dist, initial_covariance_dist],
                    [dynamics_coeff_dist, dynamics_bias_dist, dynamics_covariance_dist],
                    [emission_coeff_dist, emission_bias_dist, emission_covariance_dist]]
    
    return param_dists

