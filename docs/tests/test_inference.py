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
state_dim = 3
state_noise_dim = 3
emission_dim = 3
emission_noise_dim = 3
seq_length = 100
mu0 = jnp.zeros(state_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 5.0 * jnp.eye(state_noise_dim)
R = 10.0 * jnp.eye(emission_noise_dim)

# ICASSP
f1 = lambda x, q, u: (1-u) * x / 2.  + u * jnp.sin(10 * x) + q
g1 = lambda x, r, u:  0.1 * jnp.dot(x, x) + r
def g1lp(x,y,u):
    return MVN(loc = g1(x, 0.0, u), covariance_matrix = R).log_prob(y)


# Lorenz 63
def lorentz_63(x, sigma=10, rho=28, beta=2.667, dt=0.01):
    dx = dt * sigma * (x[1] - x[0])
    dy = dt * (x[0] * rho - x[1] - x[0] *x[2]) 
    dz = dt * (x[0] * x[1] - beta * x[2])
    return jnp.array([dx+x[0], dy+x[1], dz+x[2]])
f63 = lambda x, q, u: lorentz_63(x) + q

# stochastic growth model
f3 = lambda x, q, u: x / 2. + 100. * x / (1 + jnp.power(x, 2)) * u + q
g3 = lambda x, r, u: 0.8 * x + r
def g3lp(x,y,u):
    return MVN(loc = g3(x, 0.0, u), covariance_matrix = R).log_prob(y)


# Stochastic Volatility
alpha = 0.91
sigma = 1.0
beta = 0.5
f = lambda x, q, u: alpha * x + sigma * q
h = lambda x, r, u: beta * jnp.exp(x/2) * r
def svlp(x,y,u):
    return MVN(loc = h(x, 0.0, u), covariance_matrix = h(x, 1.0, u)**2 * R).log_prob(y)

# Multivariate SV
Phi = 0.8 * jnp.eye(state_dim)
f_msv = lambda x, q, u: Phi @ x +  q
h_msv = lambda x, r, u: 0.5 * jnp.multiply(jnp.exp(x/2), r)
def msvlp(x,y,u):
    return MVN(loc = h_msv(x, 0.0, u), covariance_matrix = jnp.diag(jnp.exp(x/2.0)) @ R @ jnp.diag(jnp.exp(x/2.0))).log_prob(y)


# Inputs
# inputs = 1. * jnp.cos(0.1 * jnp.arange(seq_length))
sm = lambda x : jnp.exp(x) / (1+jnp.exp(x))
inputs = sm(jnp.arange(seq_length)-50) # off - on
# inputs = 1.0 * jnp.ones(seq_length) # on - on


f = f_msv
g = h_msv
glp = msvlp


# initialization 
model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
params = ParamsNLSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias=jnp.zeros(state_noise_dim),
    dynamics_noise_covariance=Q,
    emission_function=g,
    emission_noise_bias=jnp.zeros(emission_noise_dim),
    emission_noise_covariance=R,
)

# Generate synthetic data 
key = jr.PRNGKey(10000)
states, emissions = model.sample(params, key, seq_length, inputs = inputs)


# AGSF
M = 5
num_components = [M, 3, 3]
out = gf.augmented_gaussian_sum_filter(params, emissions, num_components, opt_args = (0.0, 0.1), inputs=inputs)
