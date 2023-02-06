from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap
from jax import random as jr

import gaussfiltax.utils as utils
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM, NonlinearGaussianSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf

import matplotlib.pyplot as plt


f1 = lambda x: jnp.sin(10 * x)
g1 = lambda x: 0.1 * jnp.array([jnp.dot(x, x)])
f2 = lambda x: x
Hmat = jnp.array([[1, 0], [0, 1], [0, 0]])
#g2 = lambda x: Hmat @ x
def g2(x):
    # print('x', x)
    # print('x shape', x.shape)
    # print(Hmat.shape)
    return Hmat @ x

state_dim = 1
emission_dim = 1
seq_length = 100
mu0 = jnp.zeros(state_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 0.1 * jnp.eye(state_dim)
R = 1.0 * jnp.eye(emission_dim)


model = NonlinearGaussianSSM(state_dim, emission_dim)
params = ParamsNLGSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f1,
    dynamics_covariance=Q,
    emission_function=g1,
    emission_covariance=R,
)

key = jr.PRNGKey(0)
states, emissions = model.sample(params, key, seq_length)
num_components = 5

posterior_filtered = gf.gaussian_sum_filter(params, emissions, num_components, 1)
print(posterior_filtered.filtered_means)

plt.plot(states)
plt.plot(posterior_filtered.filtered_means[:, 0])
plt.plot(posterior_filtered.filtered_means[:, 1])
plt.show()