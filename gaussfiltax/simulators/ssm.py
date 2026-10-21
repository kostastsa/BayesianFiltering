import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from abc import ABC
from abc import abstractmethod
from parameters import ParamSSM
from jaxtyping import Array, Float # type: ignore
from typing import Optional, Tuple, Callable 
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN # type: ignore
from jax.tree_util import tree_map # type: ignore
from jax import lax, jit, vmap, debug # type: ignore
from jax.scipy.special import logsumexp as lse, gammaln # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore
from util.types import PRNGKey # type: ignore
from functools import partial

class SSM(ABC):
    r"""A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * :meth:`initial_distribution` returns the distribution over the initial state given parameters
    * :meth:`dynamics_simulator` simulates the next state given the current state and parameters
    * :meth:`emission_simulator` simulates the emission given the current state and parameters
    * :meth:`log_prior` (optional) returns the log prior probability of the parameters
    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.
    """

    @abstractmethod
    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "..."]]
    ) -> tfd.Distribution:

        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \theta)$.

        """

        raise NotImplementedError

    @abstractmethod
    def dynamics_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]
    ) -> tfd.Distribution:

        r"""Next latent state given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \theta)$.

        """

        raise NotImplementedError

    @abstractmethod
    def dynamics_function(self, params, prev_state, noise_vec, inputs):

        r"""Next latent state given current state.

        Args:
            prev_state: current latent state $z_t$
            noise_vec: noise vector
            inputs: current inputs  $u_t$

        Returns:
            next latent state

        """

        raise NotImplementedError


    @abstractmethod
    def emission_simulator(
        self,
        key: PRNGKey, # type: ignore 
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None
    ) -> tfd.Distribution:

        r"""Return a new emission given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            current emission 

        """

        raise NotImplementedError
    
    @abstractmethod
    def emission_function(self, params, state, noise_vec, inputs):
        r"""Next latent state given current state.

        Args:
            state: current latent state $z_t$
            noise_vec: noise vector
            inputs: current inputs  $u_t$

        Returns:
            next emission

        """

        raise NotImplementedError
    
    @abstractmethod
    def emission_log_prob(
        self,
        params: ParamSSM,
        emissions: Float[Array, "..."],
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        r"""Return a distribution over emissions given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \theta)$

        """

        raise NotImplementedError

    # All SSMs support sampling
    def simulate(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        num_timesteps: int,
        inputs: Optional[Float[Array, "..."]]=None
    ) -> Tuple[Float[Array, "..."],
              Float[Array, "..."]]:

        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """

        def _step(prev_state, args):

            key, inpt = args
            key1, key2, key = jr.split(key, 3)
            state = self.dynamics_simulator(key1, params, prev_state, inpt)
            emission = self.emission_simulator(key2, params, state, inpt)

            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_simulator(key2, params, initial_state, initial_input)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

        return states, emissions


class LGSSM(SSM):

    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0
    ):

        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "..."]]=None
        ):
 
        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = MVN(mean, params.dynamics.cov.value).sample(1, key)

        return new_state.flatten()
    
    @partial(jit, static_argnums=(0, 1))
    def dynamics_function(self, 
                          params: ParamSSM, 
                          prev_state: Float[Array, "..."],
                          noise_vec: Float[Array, "..."],
                          inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ prev_state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = mean + noise_vec

        return new_state.flatten()

    def emission_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights.value @ state + params.emissions.input_weights.value @ inputs + params.emissions.bias.value
        new_emission =  MVN(mean, params.emissions.cov.value).sample(1, key)

        return new_emission.flatten()

    # @partial(jit, static_argnums=(0, 1))
    def emission_function(self,
                          params: ParamSSM,
                          state: Float[Array, "..."],
                          noise_vec: Float[Array, "..."],
                          inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights.value @ state + params.emissions.input_weights.value @ inputs + params.emissions.bias.value
        new_emission = mean + noise_vec

        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emission: Float[Array, "..."],
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):
        mean = params.emissions.weights.value @ state + params.emissions.input_weights.value @ inputs + params.emissions.bias.value
        cov = params.emissions.cov.value
        dist = MVN(mean, cov)
        return dist.log_prob(emission)


class SV(SSM):
    
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "..."]]=None) -> tfd.Distribution:
        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = MVN(mean, params.dynamics.cov.value).sample(1, key)

        return new_state.flatten()
    
    def dynamics_function(self,
                        params: ParamSSM,
                        prev_state: Float[Array, "..."],
                        noise_vec: Float[Array, "..."],
                        inputs: Optional[Float[Array, "..."]]=None):
        
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights.value @ prev_state + params.dynamics.input_weights.value @ inputs + params.dynamics.bias.value
        new_state = mean + noise_vec

        return new_state.flatten()

    def emission_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        dim = params.emissions.bias.value.shape[0]
        r = MVN(jnp.zeros(dim), jnp.eye(dim)).sample(1, key)
        chol = jnp.diag(jnp.exp(state)) @ params.emissions.corchol.value
        new_emission = params.emissions.bias.value + chol @ r.flatten()

        return new_emission.flatten()
    
    def emission_function(self,
                        params: ParamSSM,
                        state: Float[Array, "..."],
                        noise_vec: Float[Array, "..."],
                        inputs: Optional[Float[Array, "..."]]=None):
        
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        dim = params.emissions.bias.value.shape[0]
        chol = jnp.diag(jnp.exp(state)) @ params.emissions.corchol.value
        new_emission = params.emissions.bias.value + chol @ noise_vec.flatten()

        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emission: Float[Array, "..."],
        state: Float[Array, "..."],
        inputs: Optional[Float[Array, "..."]]=None):

        mean = params.emissions.bias.value
        cov = jnp.diag(jnp.exp(state)) @ params.emissions.corchol.value @ params.emissions.corchol.value.T @ jnp.diag(jnp.exp(state))
        dist = MVN(mean, cov)

        return dist.log_prob(emission)


class NonlinearSSM(SSM):
    """
    General Nonlinear State Space Model.

    The model is defined as follows

    $$z_t = f(z_{t-1}, q_t, u_t)$$
    $$y_t = h(z_t, r_t, u_t)$$

    where 

    $$q_t ~ N(0, Q_t)$$
    $$r_t ~ N(0, R_t)$$
    $$z_1 ~ N(m, S)$$

    and the model parameters are

    * $z_t$ = hidden variables of size `state_dim`,
    * $y_t$ = observed variables of size `emission_dim`
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).
    * $f$ = dynamics (transition) function
    * $h$ = emission (observation) function
    * $Q$ = covariance matrix of dynamics (system) noise
    * $R$ = covariance matrix for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state


    These parameters of the model are stored in a separate object of type :class:`ParamsNLGSSM`.
    """


    def __init__(self, state_dim: int, 
                 state_noise_dim:int,  
                 emission_dim: int, 
                 emission_noise_dim:int, 
                 dynamics_fn,
                 emission_fn,
                 input_dim: int = 0):

        self.state_dim = state_dim
        self.state_noise_dim = state_noise_dim
        self.emission_dim = emission_dim
        self.emission_noise_dim = emission_noise_dim
        self.dynamics_fn = dynamics_fn
        self.emission_fn = emission_fn
        self.input_dim = input_dim

    @property
    def emission_shape(self):

        return (self.emission_dim,)

    @property
    def inputs_shape(self):

        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamSSM,
        inputs: Optional[Float[Array, "..."]]=None) -> tfd.Distribution:

        return MVN(params.initial.mean.value, params.initial.cov.value)

    def dynamics_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, '...'],
        inputs: Optional[Float[Array, '...']] = None
    ):

        f = self.dynamics_fn
        q = MVN(params.dynamics.bias.value, params.dynamics.cov.value).sample(1, key)
        q = q.flatten()
        new_state = f(state, q, inputs, params)

        return new_state.flatten()

    def dynamics_function(
        self,
        params: ParamSSM,
        prev_state: Float[Array, '...'],
        noise_vec: Float[Array, '...'],
        inputs: Optional[Float[Array, '...']] = None
    ):

        new_state = self.dynamics_fn(prev_state, noise_vec, inputs, params)

        return new_state.flatten()

    def emission_simulator(
        self,
        key: PRNGKey, # type: ignore
        params: ParamSSM,
        state: Float[Array, '...'],
        inputs: Optional[Float[Array, '...']] = None
     ):

        h = self.emission_fn
        r = MVN(params.emissions.bias.value, params.emissions.cov.value).sample(1, key)
        new_emission = h(state, r, inputs, params)

        return new_emission.flatten()
    
    def emission_function(
        self,
        params: ParamSSM,
        state: Float[Array, '...'],
        noise_vec: Float[Array, '...'],
        inputs: Optional[Float[Array, '...']]=None):

        h = self.emission_fn
        new_emission = h(state, noise_vec, inputs, params)

        return new_emission.flatten()
    
    def emission_log_prob(
        self,
        params: ParamSSM,
        emissions: Float[Array, '...'],
        state: Float[Array, '...'],
        inputs: Optional[Float[Array, '...']]=None):

        '''
        Logprob of emissions given state for additive models!
        '''

        h = self.emission_fn
        mean = h(state, params.emissions.bias.value, inputs, params)
        dist = MVN(mean, params.emissions.cov.value)

        return dist.log_prob(emissions)