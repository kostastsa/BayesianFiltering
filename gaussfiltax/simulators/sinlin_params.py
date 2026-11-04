import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore


# Values of the showcase model (plan step 6): x0 ~ N(0, I), Q = 0.1 I, R = 0.1 I.
# Phi = 0.9 I, kappa, omega and the direction a are not parameters of the pytree. They are in simulators/sinlin.py.
X0_VAR = 1.0
Q_VAR = 0.1
R_VAR = 0.1


def _init_vals(state_dim, emission_dim, input_dim):

    initial_mean = jnp.zeros(state_dim)
    initial_covariance = X0_VAR * jnp.eye(state_dim)

    dynamics_bias = jnp.zeros(state_dim)
    dynamics_covariance = Q_VAR * jnp.eye(state_dim)

    emission_bias = jnp.zeros(emission_dim)
    emission_covariance = R_VAR * jnp.eye(emission_dim)

    return [[initial_mean, initial_covariance],
            [dynamics_bias, dynamics_covariance],
            [emission_bias, emission_covariance]]


def _param_dists(state_dim, emission_dim, input_dim):
    """Not used by the experiments (the parameters are fixed). Same layout as _init_vals."""

    m = state_dim * (state_dim + 1) // 2
    l = emission_dim * (emission_dim + 1) // 2
    return [[tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim)),
             tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))],
            [tfd.MultivariateNormalDiag(loc=jnp.zeros(state_dim), scale_diag=jnp.ones(state_dim)),
             tfd.MultivariateNormalDiag(loc=jnp.zeros(m), scale_diag=jnp.ones(m))],
            [tfd.MultivariateNormalDiag(loc=jnp.zeros(emission_dim), scale_diag=jnp.ones(emission_dim)),
             tfd.MultivariateNormalDiag(loc=jnp.zeros(l), scale_diag=jnp.ones(l))]]
