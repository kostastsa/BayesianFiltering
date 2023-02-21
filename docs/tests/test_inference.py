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

        means = posterior_filtered.filtered_means
        covs = posterior_filtered.filtered_covariances
        pred_means = posterior_filtered.predicted_means
        pred_covs = posterior_filtered.predicted_covariances
        weights = posterior_filtered.weights
        point_estimates = jnp.sum(jnp.einsum('ijk,ij->ijk', means, weights), axis=0)


        # Output plots and values

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


    def test_augmented_gaussian_sum_filter(self):
        def g2(x):
            return jnp.array([[1, 0], [0, 1], [0, 0]]) @ x

        key = jr.PRNGKey(0)
        states, emissions = self.model.sample(self.params, key, self.seq_length)
        num_components = [5, 4, 2]

        posterior_filtered = gf.augmented_gaussian_sum_filter(self.params, emissions, num_components)
        # hessian = jacfwd(jacfwd(g2))
        # args = (10, 0.1, 1, 10)
        # posterior_filtered = gf._autocov(self.mu0, self.Sigma0, hessian_tensor=hessian, args = args)
        print(posterior_filtered)

        

test = TestInference()
# test.test_gaussian_sum_filter()
test.test_augmented_gaussian_sum_filter()

# mean1 = jnp.array([0.0, 0.0])
# cov1 = 1.0*jnp.eye(2)
# mean2 = jnp.array([1.0, 1.0])
# cov2 = 2.0*jnp.eye(2)

# comp1 = containers.GaussianComponent(mean1, cov1, 0.5)
# comp2 = containers.GaussianComponent(mean2, cov2, 0.5)


# Delta1 = 0.1 * jnp.eye(2)
# Delta2 = 0.05 * jnp.eye(2)

# new_sum = containers._branches_from_tree([comp1, comp2], [Delta1, Delta2], [3, 2])

# print(new_sum)

# leaves, treedef = jtu.tree_flatten(new_sum, is_leaf=lambda x: isinstance(x, containers.GaussianComponent))
# print(leaves)
# print(treedef)

