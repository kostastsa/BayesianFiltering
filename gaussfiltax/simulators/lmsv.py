import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
from jax import random as jr # type: ignore
import os
from simulators.ssm import NonlinearSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN # type: ignore
from util.bijectors import RealToPSDBijector # type: ignore
from simulators.lmsv_params import _init_vals, _param_dists
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
import util.io


# Linear Multivariate Stochastic Volatility (LMSV) model functions
# Dynamics: x_{t+1} = Phi * x_t + q_t (linear AR process)
def f(x, q, u, params):
    """
    Dynamics function for LMSV: linear autoregressive process
    x_{t+1} = Phi * x_t + q_t
    """
    return params.dynamics.weights.value @ x + q


# Emission: y_t = u_t * beta * exp(x_t / sigma) * r_t + (1 - u_t) * (H * x_t + r_t)
# When u=1: multiplicative stochastic volatility (nonlinear)
# When u=0: linear Gaussian (linear)
def g(x, r, u, params):
    """
    Emission function for LMSV: input-dependent stochastic volatility
    
    When u=1: y_t = beta * diag(exp(x / sigma)) * r
    When u=0: y_t = H * x + r
    
    This allows switching between high-volatility (nonlinear) and 
    low-volatility (linear) regimes based on input u.
    """
    sigma = params.emissions.sigma.value
    beta = params.emissions.beta.value
    H = params.emissions.weights.value
    
    # Multiplicative stochastic volatility term
    volatility_term = beta * jnp.multiply(jnp.exp(x / sigma), r)
    
    # Linear term
    linear_term = H @ x + r
    
    # Mix based on input u
    return u * volatility_term + (1.0 - u) * linear_term

def log_prob(params, emissions, state, inputs=None):
    """
    Log-probability of emissions for the mixed linear/MSV observation model.
    """
    u = inputs if inputs is not None else 0.0
    sigma = params.emissions.sigma.value
    beta = params.emissions.beta.value
    H = params.emissions.weights.value
    bias = params.emissions.bias.value
    cov = params.emissions.cov.value

    emission_dim = cov.shape[0]
    M = u * beta * jnp.diag(jnp.exp(state / sigma)) + (1.0 - u) * jnp.eye(emission_dim)
    mean = (1.0 - u) * (H @ state) + M @ bias
    scaled_cov = M @ cov @ M.T

    dist = MVN(mean, scaled_cov)
    return dist.log_prob(emissions)

def setup(state_dim, emission_dim, input_dim):
    """
    Setup the LMSV simulator with appropriate parameters and SSM structure.
    
    Args:
        state_dim: dimension of latent state
        emission_dim: dimension of observations
        input_dim: dimension of control inputs (typically 0 or 1)
    
    Returns:
        Dictionary containing:
        - ssm: NonlinearSSM object
        - props: initialized parameter properties
        - exp_info: experiment metadata
    """

    param_names = [
        ['mean', 'cov'],
        ['weights', 'bias', 'cov'],
        ['weights', 'bias', 'cov', 'sigma', 'beta']
    ]
    
    constrainers = [
        [None, RealToPSDBijector],
        [None, None, RealToPSDBijector],
        [None, None, RealToPSDBijector, None, None]
    ]
    
    init_vals = _init_vals(state_dim, emission_dim, input_dim)
    prior_fields = init_vals

    props = initialize(prior_fields, param_names, constrainers)
    out = {}
    
    # LMSV has same dimensions for state and state noise, emission and emission noise
    state_noise_dim = state_dim
    emission_noise_dim = emission_dim

    ssm = NonlinearSSM(
        state_dim, 
        state_noise_dim, 
        emission_dim, 
        emission_noise_dim, 
        f, 
        g
    )

    ssm.emission_log_prob = log_prob

    out['ssm'] = ssm
    out['props'] = props
    out['exp_info'] = {
        'sim': 'lmsv',
        'state_dim': state_dim,
        'emission_dim': emission_dim,
        'input_dim': input_dim,
        'param_names': param_names,
        'constrainers': constrainers,
        'prior_fields': prior_fields        
    }
    
    return out


def get_inputs(key, num_timesteps):
    """
    Returns the inputs for the LMSV simulator.
    
    Inputs control the volatility regime:
    - u=0: low volatility (linear regime)
    - u=1: high volatility (nonlinear stochastic volatility regime)
    - u in (0,1): mixture of both regimes
    
    Common patterns:
    - Random binary switching
    - Sinusoidal modulation
    - Step functions for regime changes
    """
    
    # Example: sinusoidal input modulating volatility regime
    # inputs = jnp.array([(1 - jnp.sin(0.2 * t)) / 2 for t in range(num_timesteps)])

    # Example: squared sine wave, used in the ablation study
    # inputs = jnp.sin(0.1 * jnp.arange(num_timesteps)) ** 2
    
    # Example: step function with regime change
    # inputs = jnp.array([0] * (num_timesteps // 2) + [1] * (num_timesteps // 2))

    # Step function: switch every 20 timesteps between 0 and 1
    inputs = (jnp.floor_divide(jnp.arange(num_timesteps), 20) % 2).astype(jnp.float32)
    
    # Default: random switching between regimes
    # inputs = jr.choice(key, jnp.array([0.0, 1.0]), shape=(num_timesteps,))
    
    return inputs


def get_root():
    """
    Returns the root folder for LMSV data storage.
    """
    return 'data/simulators/lmsv'


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """
    # Placeholder - to be populated with specific experiment data
    true_ps = jnp.log([0.8, 10.0, 0.1, 4.0, 0.5])  # [Phi, Q, R, sigma, beta]
    obs_xs = None  # Load from file if needed
    
    return true_ps, obs_xs


def get_disp_lims():
    """
    Returns the parameter display limits for visualization.
    """
    return [-3.0, 3.0]


def get_param_info():
    """
    Returns the parameter info module for LMSV.
    """
    import simulators.lmsv_params as param_info
    return param_info
