from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap
from jax import random as jr

import gaussfiltax.utils as utils
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM, NonlinearGaussianSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf

import matplotlib.pyplot as plt


# Model definition 
f1 = lambda x: jnp.sin(10 * x)
g1 = lambda x: 1 * jnp.array([jnp.dot(x, x)])
f2 = lambda x: 0.8 * x
Hmat = jnp.array([[1, 0], [0, 1], [0, 0]])
#g2 = lambda x: Hmat @ x
def g2(x):
    # print('x', x)
    # print('x shape', x.shape)
    # print(Hmat.shape)
    return Hmat @ x

state_dim = 2
emission_dim = 3
seq_length = 100
mu0 = jnp.zeros(state_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 0.1 * jnp.eye(state_dim)
R = 1.0 * jnp.eye(emission_dim)

model = NonlinearGaussianSSM(state_dim, emission_dim)
params = ParamsNLGSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f2,
    dynamics_covariance=Q,
    emission_function=g2,
    emission_covariance=R,
)

# Generate data and run inference
key = jr.PRNGKey(0)
states, emissions = model.sample(params, key, seq_length)
num_components = 5

posterior_filtered = gf.gaussian_sum_filter(params, emissions, num_components, 1)

means = posterior_filtered.filtered_means
covs = posterior_filtered.filtered_covariances
pred_means = posterior_filtered.predicted_means
pred_covs = posterior_filtered.predicted_covariances
weights = posterior_filtered.weights
point_estimates = jnp.sum(jnp.einsum('ijk,ij->ijk', means, weights), axis=0)


# Output plots and values

# print(means.shape)
# print(weights.shape)
# print(point_estimates.shape)



# fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 4))
# axes[0].plot(states, label="xs")
# leg = ["states"]
# for m in range(num_components):
#     axes[0].plot(pred_means[m])
#     leg.append("model {}".format(m))
# axes[0].legend(leg)

# for m in range(num_components):
#     axes[1].plot(pred_covs[m].squeeze())

# for m in range(num_components):
#     axes[2].plot(weights[m])

# axes[3].plot(states, label="xs")
# leg = ["states", "GSF"]
# axes[3].plot(point_estimates)
# axes[3].legend(leg)
# plt.show()
