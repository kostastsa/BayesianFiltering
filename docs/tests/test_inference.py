import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')
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
from gaussfiltax.utils import sdp_opt

import matplotlib.pyplot as plt

def lorentz_63(x, sigma=10, rho=28, beta=2.667, dt=0.01):
    dx = dt * sigma * (x[1] - x[0])
    dy = dt * (x[0] * rho - x[1] - x[0] *x[2]) 
    dz = dt * (x[0] * x[1] - beta * x[2])
    return jnp.array([dx+x[0], dy+x[1], dz+x[2]])

class TestInference:

    # Parameters
    state_dim = 1
    state_noise_dim = 1
    emission_dim = 1
    emission_noise_dim = 1
    seq_length = 100
    mu0 = jnp.zeros(state_dim)
    Sigma0 = 1.0 * jnp.eye(state_dim)
    Q = 1.0 * jnp.eye(state_dim)
    R = 1.0 * jnp.eye(emission_dim)

    # stochastic growth model
    f3 = lambda x, q, u: x / 2. + 25. * x / (1+ jnp.power(x, 2)) + u + q
    g3 = lambda x, r, u: x**2/20. + r

    # Lorenz 63
    f63 = lambda x, q, u: lorentz_63(x) + q
    g1 = lambda x, r, u:  0.05 * jnp.dot(x, x) + r

    # Inputs
    inputs = 8. * jnp.cos(jnp.arange(seq_length))

    # Model definition 
    model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
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
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=self.inputs)

    def test_augmented_gaussian_sum_filter_optimal(self):
        num_components = [5, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter_optimal(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=self.inputs)
        return posterior_filtered

if __name__ == "__main__":
    test = TestInference()
    # test.test_gaussian_sum_filter()
    out = test.test_augmented_gaussian_sum_filter_optimal()
    print(out.weights.shape)

    for m in range(5):
        plt.plot(out.weights[m], label = f"m = {m}")
    plt.legend()    
    plt.show()
    