"""Showcase model of plan step 6: a nonlinearity in one direction of a large state.

    x_{t+1} = PHI x_t + q_t,                 q_t ~ N(0, 0.1 I),  x_0 ~ N(0, I)
    y_1     = kappa sin(omega a^T x) + r_1   (the curved channel)
    y_j     = x_j + r_j,  j = 2..d           (the linear channels),  r ~ N(0, 0.1 I)

The emission is nonlinear only along the direction a. rot = 0: a = e_1. rot = 1: a is a fixed random unit vector
(the same for each d, from the key ROT_SEED), so that no coordinate-wise rho matches the curved direction.
"""
import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')
from jax import numpy as jnp # type: ignore
from jax import random as jr # type: ignore
from simulators.ssm import NonlinearSSM
from util.bijectors import RealToPSDBijector # type: ignore
from simulators.sinlin_params import _init_vals
from util.param import initialize

PHI = 0.9
ROT_SEED = 0


def direction(state_dim, rot):
    """The curved direction a: e_1 (rot = 0) or a fixed random unit vector (rot = 1)."""

    if not rot:
        return jnp.zeros(state_dim).at[0].set(1.0)
    a = jr.normal(jr.PRNGKey(ROT_SEED), (state_dim,))
    return a / jnp.linalg.norm(a)


def make_fg(state_dim, kappa, omega, rot):
    """The dynamics f(x, q, u, params) and the emission g(x, r, u, params). The simulator passes r with shape (1, d),
    the filters with shape (d,)."""

    a = direction(state_dim, rot)

    def f(x, q, u, params):
        return PHI * x + q

    def g(x, r, u, params):
        y = jnp.concatenate([jnp.atleast_1d(kappa * jnp.sin(omega * (a @ x))), x[1:]])
        return y + r

    return f, g


def setup(state_dim, emission_dim, input_dim, kappa=1.0, omega=2.0, rot=False):

    if emission_dim != state_dim:
        raise ValueError(f'sinlin needs emission_dim = state_dim (got {emission_dim} and {state_dim})')

    param_names = [['mean', 'cov'],
                   ['bias', 'cov'],
                   ['bias', 'cov']]
    constrainers = [[None, RealToPSDBijector],
                    [None, RealToPSDBijector],
                    [None, RealToPSDBijector]]
    prior_fields = _init_vals(state_dim, emission_dim, input_dim)
    props = initialize(prior_fields, param_names, constrainers)

    f, g = make_fg(state_dim, kappa, omega, rot)
    out = {}
    out['ssm'] = NonlinearSSM(state_dim, state_dim, emission_dim, emission_dim, f, g)
    out['props'] = props
    out['exp_info'] = {
        'sim': 'sinlin',
        'state_dim': state_dim,
        'emission_dim': emission_dim,
        'input_dim': input_dim,
        'kappa': kappa,
        'omega': omega,
        'rot': rot,
        'param_names': param_names,
        'constrainers': constrainers,
        'prior_fields': prior_fields
    }
    return out


def get_inputs(key, num_timesteps):
    """The model has no input. The filters still need an input sequence."""

    return jnp.zeros(num_timesteps)


def get_param_info():
    import simulators.sinlin_params as param_info
    return param_info
