from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap
from jax import random as jr
from jax import tree_util as jtu

import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM, NonlinearGaussianSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf

import matplotlib.pyplot as plt

class TestInference:

    # Model definition 
    f1 = lambda x: jnp.sin(10 * x)
    g1 = lambda x: 1 * jnp.array([jnp.dot(x, x)])
    f2 = lambda x: 0.8 * x
    def g2(x):
        return jnp.array([[1, 0], [0, 1], [0, 0]]) @ x

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

    def test_gaussian_sum_filter(self):
        
        # Generate data and run inference
        key = jr.PRNGKey(0)
        states, emissions = self.model.sample(self.params, key, self.seq_length)
        num_components = 5

        posterior_filtered = gf.gaussian_sum_filter(self.params, emissions, num_components, 1)

        means = posterior_filtered.means
        covs = posterior_filtered.covariances
        pred_means = posterior_filtered.predicted_means
        pred_covs = posterior_filtered.predicted_covariances
        weights = posterior_filtered.weights
        point_estimates = jnp.sum(jnp.einsum('ijk,ij->ijk', means, weights), axis=0)


    def test_augmented_gaussian_sum_filter(self):
        def g2(x):
            return jnp.array([[1, 0], [0, 1], [0, 0]]) @ x

        key = jr.PRNGKey(0)
        states, emissions = self.model.sample(self.params, key, self.seq_length)
        num_components = [5, 3, 3] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered = gf.augmented_gaussian_sum_filter(self.params, emissions, num_components)


test = TestInference()

tin = time.time()
test.test_gaussian_sum_filter()
tout = time.time()
print('GSF time:', tout - tin)

tin = time.time()
test.test_augmented_gaussian_sum_filter()
tout = time.time()
print('AGSF time:', tout - tin)




