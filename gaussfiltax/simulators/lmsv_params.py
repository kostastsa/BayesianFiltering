import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as dist # type: ignore


def _init_vals(state_dim, emission_dim, input_dim):
    """
    Initialize parameter values for Linear Multivariate Stochastic Volatility (LMSV) model.
    
    The LMSV model has:
    - Linear state dynamics with autoregressive structure
    - Nonlinear emission function with multiplicative stochastic volatility
    """
        
    # Initial state parameters
    initial_mean = jnp.zeros(state_dim)
    initial_covariance = jnp.eye(state_dim) * 1.0

    # Dynamics parameters (linear autoregressive)
    dynamics_weights = 0.8 * jnp.eye(state_dim)  # Phi matrix
    dynamics_bias = jnp.zeros(state_dim)
    dynamics_covariance = 10.0 * jnp.eye(state_dim)  # Q matrix

    # Emission parameters (nonlinear with stochastic volatility)
    # For LMSV: y_t = u_t * beta * exp(x_t / sigma) * r_t + (1 - u_t) * (H * x_t + r_t)
    emission_weights = jnp.eye(emission_dim, state_dim)  # H0 matrix
    emission_bias = 1e-4 * jnp.ones(emission_dim)  # r0 bias
    emission_covariance = 1e-1 * jnp.eye(emission_dim)  # R matrix
    
    # LMSV-specific coefficients
    sigma = jnp.array([4.0])  # volatility scaling parameter
    beta = jnp.array([0.5])   # volatility coefficient

    init_vals = [[initial_mean, initial_covariance],
                 [dynamics_weights, dynamics_bias, dynamics_covariance],
                 [emission_weights, emission_bias, emission_covariance, sigma, beta]]

    return init_vals


def _param_dists(state_dim, emission_dim, input_dim):
    """
    Define prior distributions for LMSV parameters.
    """

    # Initial state distribution
    initial_mean_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(state_dim), 
        scale_diag=jnp.ones(state_dim)
    )

    m = state_dim * (state_dim + 1) // 2
    initial_covariance_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(m), 
        scale_diag=jnp.ones(m)
    )

    # Dynamics weights (Phi matrix)
    mu = 0.8 * jnp.eye(state_dim)
    col_cov = jnp.eye(state_dim) * 0.01
    row_cov = jnp.eye(state_dim) * 0.01
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    dynamics_weights_dist = dist.MatrixNormal(
        loc=mu,
        scale_tril_row=scale_row,
        scale_tril_column=scale_column
    )

    # Dynamics bias
    dynamics_bias_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(state_dim), 
        scale_diag=1.0 * jnp.ones(state_dim)
    )

    # Dynamics covariance
    dynamics_covariance_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(m), 
        scale_diag=5.0 * jnp.ones(m)
    )

    # Emission weights (H matrix)
    mu = 1.0 * jnp.eye(emission_dim, state_dim)
    col_cov = jnp.eye(state_dim) * 0.1
    row_cov = jnp.eye(emission_dim) * 0.1
    scale_column = jnp.linalg.cholesky(col_cov)
    scale_row = jnp.linalg.cholesky(row_cov)

    emission_weights_dist = dist.MatrixNormal(
        loc=mu,
        scale_tril_row=scale_row,
        scale_tril_column=scale_column
    )

    # Emission bias
    emission_bias_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(emission_dim), 
        scale_diag=0.01 * jnp.ones(emission_dim)
    )

    # Emission covariance
    l = emission_dim * (emission_dim + 1) // 2
    emission_covariance_dist = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(l), 
        scale_diag=0.5 * jnp.ones(l)
    )

    # LMSV-specific parameter distributions
    sigma_dist = tfd.LogNormal(loc=jnp.log(4.0), scale=0.5)
    beta_dist = tfd.LogNormal(loc=jnp.log(0.5), scale=0.5)

    param_dists = [
        [initial_mean_dist, initial_covariance_dist],
        [dynamics_weights_dist, dynamics_bias_dist, dynamics_covariance_dist],
        [emission_weights_dist, emission_bias_dist, emission_covariance_dist, sigma_dist, beta_dist]
    ]

    return param_dists
