from jaxtyping import Array, Float
from typing import NamedTuple, Optional, Union, Callable, Tuple
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import tensorflow_probability.substrates.jax.distributions as tfd
from gaussfiltax.ssm import SSM
from gaussfiltax.parameters import ParameterSet, PropertySet
from gaussfiltax.types import PRNGKey, Scalar
import jax
from jax import jit, lax, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jr


tfd = tfp.distributions
tfb = tfp.bijectors


FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]


class ParamsNLSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """

    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_noise_bias: Float[Array, "state_noise_dim"]
    dynamics_noise_covariance: Float[Array, "state_noise_dim state_noise_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_noise_bias: Float[Array, "emission_noise_dim"] 
    emission_noise_covariance: Float[Array, "emission_noise_dim emission_noise_dim"]



class ParamsBPF(NamedTuple):
    """Parameters for a BPF model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_noise_bias: Float[Array, "state_noise_dim"]
    dynamics_noise_covariance: Float[Array, "state_noise_dim state_noise_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_noise_bias: Float[Array, "emission_noise_dim"] 
    emission_noise_covariance: Float[Array, "emission_noise_dim emission_noise_dim"]
    emission_distribution_log_prob: Callable

    def sample_dynamics_distribution(self, key, x, u):
        q = MVN(loc = self.dynamics_noise_bias, covariance_matrix=self.dynamics_noise_covariance).sample(seed=key)
        return self.dynamics_function(x, q, u)

class NonlinearGaussianSSM(SSM):
    """
    Nonlinear Gaussian State Space Model.

    The model is defined as follows

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    where the model parameters are

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


    def __init__(self, state_dim: int, emission_dim: int, input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamsNLSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsNLSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        f = params.dynamics_function
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_noise_covariance)

    def emission_distribution(
        self,
        params: ParamsNLSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
     ) -> tfd.Distribution:
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_noise_covariance)
    

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


    def __init__(self, state_dim: int, state_noise_dim:int,  emission_dim: int, emission_noise_dim:int, input_dim: int = 0):
        self.state_dim = state_dim
        self.state_noise_dim = state_noise_dim
        self.emission_dim = emission_dim
        self.emission_noise_dim = emission_noise_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamsNLSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsNLSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        f = params.dynamics_function
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_noise_covariance)

    def emission_distribution(
        self,
        params: ParamsNLSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
     ) -> tfd.Distribution:
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_noise_covariance)
    
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
              Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """

        f = params.dynamics_function
        h = params.emission_function

        def _step(prev_state, args):
            key, inpt = args
            key1, key2 = jr.split(key, 2)
            q = MVN(params.dynamics_noise_bias, params.dynamics_noise_covariance).sample(seed=key1)
            r = MVN(params.emission_noise_bias, params.emission_noise_covariance).sample(seed=key2)
            state = f(prev_state, q, inpt)
            emission = h(state, r, inpt)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        r0 = MVN(params.emission_noise_bias, params.emission_noise_covariance).sample(seed=key2)
        initial_emission = h(initial_state, r0, initial_input)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions