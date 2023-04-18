from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax
from jax import random as jr
from jax import tree_util as jtu
import csv

import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from gaussfiltax.models import ParamsNLSSM, NonlinearGaussianSSM, NonlinearSSM, ParamsBPF
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf

import matplotlib.pyplot as plt

class TestInference:

    # Parameters
    state_dim = 1
    emission_dim = 1
    seq_length = 100
    mu0 = jnp.zeros(state_dim)
    Sigma0 = 1.0 * jnp.eye(state_dim)
    Q = 1.0 * jnp.eye(state_dim)
    R = 1.0 * jnp.eye(emission_dim)

    # Nonlinearities
    f1 = lambda x: jnp.sin(10 * x)
    g1 = lambda x: 0.1 * jnp.array([jnp.dot(x, x)])

    f2 = lambda x: 0.8 * x
    def g2(x):
        return 2 * x
    #    return jnp.array([[1, 0], [0, 1], [0, 0]]) @ x

    # stochastic growth model
    f3 = lambda x, u: x / 2. + 25. * x / (1+ jnp.power(x, 2)) + u
    g3 = lambda x, u: x**2/20.

    # Inputs
    inputs = 8. * jnp.cos(jnp.arange(seq_length))

    # Model definition 
    model = NonlinearGaussianSSM(state_dim, emission_dim)
    params = ParamsNLSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_function=f3,
        dynamics_noise_bias=jnp.zeros(Q.shape[0]),
        dynamics_noise_covariance=Q,
        emission_function=g3,
        emission_noise_bias=jnp.zeros(R.shape[0]),
        emission_noise_covariance=R,
    )

    # Generate synthetic data 
    key = jr.PRNGKey(0)
    states, emissions = model.sample(params, key, seq_length, inputs = inputs)


    def test_gaussian_sum_filter(self):
        num_components = 5
        posterior_filtered = gf.gaussian_sum_filter(self.params, self.emissions, num_components, 1, self.inputs)

    def test_augmented_gaussian_sum_filter(self):
        num_components = [5, 3, 3] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter(self.params, self.emissions, num_components, opt_args = (20, 0.001, 50), inputs=self.inputs)

# test = TestInference()

# tin = time.time()
# test.test_gaussian_sum_filter()
# tout = time.time()
# print('GSF time:', tout - tin)

# tin = time.time()
# test.test_augmented_gaussian_sum_filter()
# tout = time.time()
# print('AGSF time:', tout - tin)




# Parameters
state_dim = 1
state_noise_dim = 1
emission_dim = 1
emission_noise_dim = 1
seq_length = 500
mu0 = jnp.zeros(state_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 1.0 * jnp.eye(state_noise_dim)
R = 1.0 * jnp.eye(emission_noise_dim)

# Nonlinearities
alpha = 0.91
sigma = 1.0
beta = 0.5
f = lambda x, q, u: alpha * x + sigma * q
h = lambda x, r, u: beta * jnp.exp(x/2) * r

# Inputs
inputs = 100. * jnp.cos(jnp.arange(seq_length))

# Model definition 
model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
params = ParamsNLSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias=jnp.zeros(state_noise_dim),
    dynamics_noise_covariance=Q,
    emission_function=h,
    emission_noise_bias=jnp.zeros(emission_noise_dim),
    emission_noise_covariance=R,
)

# Generate synthetic data 
key = jr.PRNGKey(0)
states, emissions = model.sample(params, key, seq_length, inputs = inputs)

# GSF
M = 10
posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, M, 1, inputs)
point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)

# AGSF
num_components = [M, 1, 1] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
posterior_filtered_agsf, aux_outputs = gf.augmented_gaussian_sum_filter(params, emissions, num_components, opt_args = (0.0, 0.1), inputs=inputs)
point_estimate_agsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_agsf.means, posterior_filtered_agsf.weights), axis=0)

print(aux_outputs['timing'])

# BPF
num_particles = 10
def edlp(x,y,u):
    return MVN(loc = h(x, 0.0, u), covariance_matrix = h(x, 1.0, u)**2 * R).log_prob(y)

params_bpf = ParamsBPF(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias=jnp.zeros(state_noise_dim),
    dynamics_noise_covariance=Q,
    emission_function=h,
    emission_noise_bias=jnp.zeros(emission_noise_dim),
    emission_noise_covariance=R,
    emission_distribution_log_prob = edlp
)

posterior_bpf = gf.bootstrap_particle_filter(params_bpf, emissions, num_particles, key, inputs)
point_estimate_bpf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_bpf["particles"], posterior_bpf["weights"]), axis=0)



# Plots
# plt.plot(states)
# plt.plot(point_estimate_gsf)
# plt.plot(point_estimate_agsf)
# plt.plot(point_estimate_bpf)
# plt.plot(emissions)
# plt.legend(['x','gsf', 'agsf', 'bpf', 'y'])
# plt.show()


point_estimates = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_agsf.means, posterior_filtered_agsf.weights), axis=0)

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
fig.tight_layout(pad=3.0)
axes[0].plot(states, label="xs")
leg = ["states"]
for m in range(M):
    axes[0].plot(posterior_filtered_agsf.means[m])
    leg.append("model {}".format(m))
    axes[0].set_title("filtered means")

#axes[0].legend(leg)

for m in range(M):
    axes[1].plot(posterior_filtered_agsf.covariances[m].squeeze())
    axes[1].set_title("filtered covariances")

for m in range(M):
    axes[2].plot(posterior_filtered_agsf.weights[m])
    axes[2].set_title("weights")

axes[3].plot(states, label="xs")
leg = ["states", "AGSF"]
axes[3].plot(point_estimates)
axes[3].legend(leg)
axes[3].set_title("point estimate")

plt.show()