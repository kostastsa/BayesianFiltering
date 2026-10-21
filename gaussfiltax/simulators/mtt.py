import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
import jax # type: ignore
from jax import numpy as jnp # type: ignore
from jax import random as jr # type: ignore
import os
from simulators.ssm import NonlinearSSM
from util.bijectors import RealToPSDBijector # type: ignore
from simulators.mtt_params import _init_vals, _param_dists
from util.misc import get_prior_fields, get_bool_tree
from util.param import initialize
import util.io


dt = 1.0
FCV = 1.0 * jnp.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
acc = 0.05
acc *= 1
Omega = lambda x, acc: acc / jnp.sqrt(x[1]**2 + x[3]**2)
FCT =  lambda x, a: jnp.array([[1, jnp.sin(dt * Omega(x, a)) / Omega(x, a), 0, -(1-jnp.cos(dt * Omega(x, a))) / Omega(x, a)],
                            [0, jnp.cos(dt * Omega(x, a)), 0, -jnp.sin(dt * Omega(x, a))],
                            [0, (1-jnp.cos(dt * Omega(x, a))) / Omega(x, a), 1, jnp.sin(dt * Omega(x, a)) / Omega(x, a)],
                            [0, jnp.sin(dt * Omega(x, a)), 0, jnp.cos(dt * Omega(x, a))]])

G = jnp.array([[0.5, 0],[1, 0],[0, 0.5],[0, 1]])

f = lambda x, q, u, dyn_params: (0.5*(u-1)*(u-2) * FCV - u*(u-2) * FCT(x, acc) + 0.5*u*(u-1) * FCT(x, -acc)) @ x + G @ q
g = lambda x ,r, u, em_params: jnp.array([jnp.arctan2(x[2], x[0]), jnp.sqrt(x[0]**2 + x[2]**2)]) + r

def make_f(a):
    """Dynamics function with turn parameter a. The default f reads the module value of acc."""

    return lambda x, q, u, dyn_params: (0.5*(u-1)*(u-2) * FCV - u*(u-2) * FCT(x, a) + 0.5*u*(u-1) * FCT(x, -a)) @ x + G @ q

def setup(state_dim, emission_dim, input_dim, a=None, sigma2=None):
    """a: turn parameter (default: acc of this module). sigma2: variance of the observation noise (default 2.5)."""

    param_names = [['mean', 'cov'],
                ['bias', 'cov'],
                ['bias', 'cov']]
    
    constrainers  = [[None, RealToPSDBijector],
                     [None, RealToPSDBijector],
                     [None, RealToPSDBijector]]
    
    init_vals = _init_vals(state_dim, emission_dim, input_dim) if sigma2 is None else _init_vals(state_dim, emission_dim, input_dim, sigma2)
    prior_fields = init_vals

    props = initialize(prior_fields, param_names, constrainers)
    out = {}
    state_dim = 4
    state_noise_dim = 2
    emission_dim = 2
    emission_noise_dim = 2

    out['ssm'] = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim, f if a is None else make_f(a), g)
    out['props'] = props
    out['exp_info'] = {
        'sim' : 'mtt',
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
    Returns the inputs for the MTT simulator.
    """

    # Generate a random input sequence
    
    # inputs = jr.choice(key, jnp.array([0, 1, 2]), shape = (num_timesteps, 1)) # random inputs
    inputs = jnp.array([1]*int(2*num_timesteps/5) + [0]*int(num_timesteps/5) + [2]*int(2*num_timesteps/5)) # maneuver inputs

    return inputs


def get_root():
    """
    Returns the root folder.
    """

    return 'data/simulators/lgssm'


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_ps = jnp.log([0.01, 0.5, 1.0, 0.01])
    obs_xs = util.io.load(os.path.join(get_root(), 'obs_stats'))

    return true_ps, obs_xs


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-2.5, 2.5]


def get_param_info():
    import simulators.lgssm_params as param_info
    return param_info