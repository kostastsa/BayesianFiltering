import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.tree_util import register_pytree_node_class
from functools import partial
from jax import lax, vmap, jacfwd, jacrev, debug, device_put, jit
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
import gaussfiltax.utils as utils
from jaxtyping import Array, Float, Int
from typing import List



#@register_pytree_node_class
class GaussianComponent(NamedTuple):
    r"""Container object for a Gaussian mixture component.
        """

    mean: Array
    covariance: Array
    weight: Float

    # def tree_flatten(self):
    #     flat_contents = (self.mean, self.covariance, self.weight)
    #     return flat_contents, None

    # def tree_unflatten(cls, aux_data, flat_contents):
    #     return cls(*flat_contents)

class GaussianSum(NamedTuple):
    r"""Container object for a Gaussian mixture.
        """

    means: Array
    covariances: Array
    weights: Array

    _check_normalization = lambda self: jnp.allclose(jnp.sum(self.weights), 1.0)
    _sum_weights = lambda self: jnp.sum(self.weights)

def _gaussian_sum_to_components(gaussian_sum: GaussianSum):
    return jtu.tree_map(lambda vars: GaussianComponent(vars[0], vars[1], vars[2]), list(zip(gaussian_sum.means, gaussian_sum.covariances, gaussian_sum.weights)), is_leaf=lambda node: type(node) is tuple)

def _components_to_gaussian_sum(components)-> GaussianSum: # TODO: use scan instead of list comprehension
    # def _step(carry, component):
    #     means = carry.means
    #     covariances = carry.covariances
    #     weights = carry.weights
    #     means.append(component.mean)
    #     covariances.append(component.covariance)
    #     weights.append(component.weight)
    #     carry = GaussianSum(means, covariances, weights)
    #     return carry, carry
    # gaussian_sum = GaussianSum([], [], [])
    # _, gaussian_sum = lax.scan(_step, gaussian_sum, components)
    means = [component.mean for component in components]
    covariances = [component.covariance for component in components]
    weights = [component.weight for component in components]
    return GaussianSum(means, covariances, weights)

def _branches_from_node1(
    node_component: GaussianComponent, 
    splitting_cov: Array,
    num_particles: Int,
    key:jr.PRNGKey):
    r"""Split a Gaussian component into N Gaussian particles.
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        Delta (D_hid,D_hid): covariance of Gaussian particles.
        N (int): number of particles.

    Returns:
        particles (N,D_hid): array of particles.
    """
    # latent_dist = MVN(node_component.mean, node_component.covariance - splitting_cov)
    # new_means = latent_dist.sample((num_particles,), seed=subkey)
    num_particles = 3 # this has to be set manually for now. has to match the input given in the script.
    sampling_covariance = node_component.covariance-splitting_cov
    new_means = jr.multivariate_normal(key, node_component.mean, sampling_covariance, shape=(num_particles,))
    new_means = jnp.where(jnp.isnan(new_means), node_component.mean, new_means)
    new_covs = jnp.tile(splitting_cov, (num_particles, 1, 1))
    new_weights = jnp.array([node_component.weight / num_particles] * int(num_particles))
    branch_sum = GaussianSum(new_means, new_covs, new_weights)
    new_components = _gaussian_sum_to_components(branch_sum)
    return new_components

def _branches_from_tree1(
    components,
    split_covs_array:Float[Array, "num_components D_hid D_hid"],
    num_branch_array:Int[Array, "num_components"],
    key:jr.PRNGKey = jr.PRNGKey(0)
    ):
    r"""Branch out a Gaussian sum.
    Args:
        gauss_sum (GaussianSum): Gaussian sum.
        split_covs_array (num_components D_hid D_hid): array of splitting covariances.
        num_branch_array (num_components): array of number of branches for each of the initial components.
        key (int): random seed.

    Returns:
        branched_tree (GaussianSum): branched Gaussian sum. Depth one tree of Gaussian components.
    """
    num_components = len(num_branch_array)
    keys = list(jr.split(key, num_components))
    gauss_sum = _components_to_gaussian_sum(components)
    branched_tree = jtu.tree_map(_branches_from_node1, components, split_covs_array, num_branch_array, keys, is_leaf=lambda x: isinstance(x, GaussianComponent))
    # branched_tree = vmap(lambda mean, cov, weight, split_cov, num_prt, key: _branches_from_node(GaussianComponent(mean, cov, weight), split_cov, num_prt, key))(gauss_sum.means, gauss_sum.covariances, gauss_sum.weights, jnp.array(split_covs_array), jnp.array(num_branch_array), keys)
    return branched_tree

def _branches_from_node2(
    node_component: GaussianComponent, 
    splitting_cov: Array,
    num_particles: Int,
    key:jr.PRNGKey):
    r"""Split a Gaussian component into N Gaussian particles.
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        Delta (D_hid,D_hid): covariance of Gaussian particles.
        N (int): number of particles.

    Returns:
        particles (N,D_hid): array of particles.
    """
    # latent_dist = MVN(node_component.mean, node_component.covariance - splitting_cov)
    # new_means = latent_dist.sample((num_particles,), seed=subkey)
    num_particles = 6 # this has to be set manually for now. has to match the input given in the script.
    sampling_covariance = node_component.covariance-splitting_cov
    new_means = jr.multivariate_normal(key, node_component.mean, sampling_covariance, shape=(num_particles,))
    new_means = jnp.where(jnp.isnan(new_means), node_component.mean, new_means)
    new_covs = jnp.tile(splitting_cov, (num_particles, 1, 1))
    new_weights = jnp.array([node_component.weight / num_particles] * int(num_particles))
    branch_sum = GaussianSum(new_means, new_covs, new_weights)
    new_components = _gaussian_sum_to_components(branch_sum)
    return new_components

def _branches_from_tree2(
    components,
    split_covs_array:Float[Array, "num_components D_hid D_hid"],
    num_branch_array:Int[Array, "num_components"],
    key:jr.PRNGKey = jr.PRNGKey(0)
    ):
    r"""Branch out a Gaussian sum.
    Args:
        gauss_sum (GaussianSum): Gaussian sum.
        split_covs_array (num_components D_hid D_hid): array of splitting covariances.
        num_branch_array (num_components): array of number of branches for each of the initial components.
        key (int): random seed.

    Returns:
        branched_tree (GaussianSum): branched Gaussian sum. Depth one tree of Gaussian components.
    """
    num_components = len(num_branch_array)
    keys = list(jr.split(key, num_components))
    gauss_sum = _components_to_gaussian_sum(components)
    branched_tree = jtu.tree_map(_branches_from_node2, components, split_covs_array, num_branch_array, keys, is_leaf=lambda x: isinstance(x, GaussianComponent))
    # branched_tree = vmap(lambda mean, cov, weight, split_cov, num_prt, key: _branches_from_node(GaussianComponent(mean, cov, weight), split_cov, num_prt, key))(gauss_sum.means, gauss_sum.covariances, gauss_sum.weights, jnp.array(split_covs_array), jnp.array(num_branch_array), keys)
    return branched_tree



# how to assign value to array entry in JAX?

