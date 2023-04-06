from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax
from jax import random as jr
from jax import tree_util as jtu

import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from gaussfiltax.models import ParamsNLSSM, NonlinearGaussianSSM, GeneralNonlinearSSM
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
inputs = 8. * jnp.cos(jnp.arange(seq_length))

# Model definition 
model = GeneralNonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
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


num_components = 5
posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, num_components, 1, inputs)
point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)


# Plot synthetic data
plt.plot(states)
plt.plot(point_estimate_gsf)
plt.plot(emissions)
plt.legend(['x','gsf', 'y'])
plt.show()
