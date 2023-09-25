import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')
from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax
from jax import random as jr
from jax import tree_util as jtu
import csv

from dynamax.utils.utils import psd_solve
import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from gaussfiltax.models import ParamsNLSSM, NonlinearGaussianSSM, NonlinearSSM, ParamsBPF
from gaussfiltax.inference import ParamsUKF
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf
from gaussfiltax.utils import sdp_opt

import matplotlib.pyplot as plt

# Parameters
state_dim = 4
state_noise_dim = 2
emission_dim = 1
emission_noise_dim = 1
seq_length = 30
# mu0 = 1.0 * jnp.array([-0.05, 0.001, 0.7, -0.05])
mu0 = jnp.ones(state_dim)
q0 = jnp.zeros(state_noise_dim)
r0 = jnp.zeros(emission_noise_dim)
Sigma0 = 1.0 * jnp.array([[0.1, 0.0, 0.0, 0.0],[0.0, 0.005, 0.0, 0.0],[0.0, 0.0, 0.1, 0.0],[0.0, 0.0, 0.0, 0.01]])
Q = 1 * jnp.eye(state_noise_dim)
R = 25*1e-6 * jnp.eye(emission_noise_dim)

dt = 0.5
FCV = jnp.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
acc = 0.5
Omega = lambda x, acc: 0.1 * acc / jnp.sqrt(x[1]**2 + x[3]**2)
FCT =  lambda x, a: jnp.array([[1, jnp.sin(dt * Omega(x, a)) / Omega(x, a), 0, -(1-jnp.cos(dt * Omega(x, a))) / Omega(x, a)],
                            [0, jnp.cos(dt * Omega(x, a)), 0, -jnp.sin(dt * Omega(x, a))],
                            [0, (1-jnp.cos(dt * Omega(x, a))) / Omega(x, a), 1, jnp.sin(dt * Omega(x, a)) / Omega(x, a)],
                            [0, jnp.sin(dt * Omega(x, a)), 0, jnp.cos(dt * Omega(x, a))]])

G = jnp.array([[0.5, 0],[1, 0],[0, 0.5],[0, 1]])
fBOT = lambda x, q, u: FCV @ x + G @ q
fManBOT = lambda x, q, u: (0.5*(u-1)*(u-2)*FCV - u*(u-2)*FCT(x, acc) + 0.5*u*(u-1) * FCT(x, -acc)) @ x + G @ q
gBOT = lambda x, r, u: jnp.arctan2(x[2], x[0]) + r
gBOTlp = lambda x, y, u: MVN(loc = gBOT(x, 0.0, u), covariance_matrix = R).log_prob(y)
# inputs = jnp.zeros((seq_length, 1))
inputs = jnp.array([1]*int(seq_length/3) + [0]*int(seq_length/3) + [2]*int(seq_length/3)) # maneuver inputs

f = fManBOT
g = gBOT

class TestInference:
    
    # Model definition 
    model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
    params = ParamsNLSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_function=f,
        dynamics_noise_bias=jnp.zeros(Q.shape[0]),
        dynamics_noise_covariance=Q,
        emission_function=g,
        emission_noise_bias=jnp.zeros(R.shape[0]),
        emission_noise_covariance=R,
    )

    # Generate synthetic data 
    key = jr.PRNGKey(0)
    states, emissions = model.sample(params, key, seq_length, inputs = inputs)

    def test_gaussian_sum_filter(self):
        num_components = 5
        posterior_filtered = gf.gaussian_sum_filter(self.params, self.emissions, num_components, 1, inputs)
        return posterior_filtered
    
    def test_pgsf(self):
        num_components = 5
        posterior_filtered = gf.pgsf(self.params, self.emissions, num_components, 1, inputs)
        return posterior_filtered

    def test_augmented_gaussian_sum_filter(self):
        num_components = [2, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=inputs)
        return posterior_filtered

    def test_augmented_gaussian_sum_filter_optimal(self):
        num_components = [5, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter_optimal(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=inputs)
        return posterior_filtered
    
    def test_unscented_gaussian_sum_filter(self):
        num_components = 5
        uparams = ParamsUKF()
        posterior_filtered = gf.unscented_gaussian_sum_filter(self.params, uparams, self.emissions, num_components, 1, inputs)
        return posterior_filtered
    
    def test_unscented_agsf(self):
        num_components = [2, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        uparams = ParamsUKF(alpha=1e-3, beta=2.0, kappa=0.0)
        posterior_filtered, aux_outputs = gf.unscented_agsf(self.params, uparams, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=inputs)
        return posterior_filtered

if __name__ == "__main__":
    test = TestInference()
    posterior_filtered = test.test_pgsf()
    print(posterior_filtered)