from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Callable, Tuple
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.ssm import SSM
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from jax import jit, lax, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jr


tfd = tfp.distributions
tfb = tfp.bijectors

