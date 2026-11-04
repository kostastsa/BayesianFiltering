"""Checks of the showcase simulator sinlin (plan step 6, simulators/sinlin.py) and of the split error (util/numerics.py).

The key check is the mechanism of the showcase. At a component with Sigma = s^2 I and a mean with omega x_1 = pi/4,
the program (45) keeps Delta = Sigma on the linear coordinates and gives

    p_1 = alpha / (s^2 omega^2)        (the eigenvalue of Sigma^-1 Delta along e_1)

for each state dimension d. The single rho of Eq. (47) is rho* = alpha Tr(A) / b = alpha (d + 1) / (2 s^2 omega^2)
(kappa = 1), so it grows with d and stops at the clip RHO_MAX.

Run with pytest, or directly: python tests/test_sinlin.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import experiment_descriptor as ed
from filters.agsf import auto_cov_selection, sdp_cov_selection, RHO_MAX
from simulators import sinlin
from util.numerics import mse, mse_x1, mse_rest
from util.param import sample_prior

KAPPA, OMEGA, S2, ALPHA = 1.0, 2.0, 0.5, 0.4


def _model(d, rot=False):
    out = sinlin.setup(d, d, 0, KAPPA, OMEGA, rot)
    params = sample_prior(jr.PRNGKey(0), out['props'], 1)[0]
    return out['ssm'], params


def _rules_at(d, rot=False):
    """(Delta, rho) of 'auto' and 'sdp' in the update step, as the filter calls them, at m with omega a^T m = pi/4."""
    ssm, params = _model(d, rot)
    g = partial(ssm.emission_function, params)
    H_x, H_r = jax.jacfwd(g, 0), jax.jacfwd(g, 1)
    H_xx = jax.jacrev(H_x)
    a = sinlin.direction(d, rot)
    m = (jnp.pi / (4 * OMEGA)) * a
    P = S2 * jnp.eye(d)
    r0, R, u = params.emissions.bias.value, params.emissions.cov.value, 0.0
    auto = auto_cov_selection(m, P, H_x, H_xx, r0, u, ALPHA, H_r, R)
    sdp = sdp_cov_selection(m, P, H_x, H_xx, r0, u, ALPHA, H_r, R)
    return auto, sdp, P, a


def test_descriptor_round_trip():
    text = ed.SimulatorDescriptor.get_descriptor('sinlin').create_desc(10, 10, 100, 1.0, 2.0, True)
    desc = ed.SimulatorDescriptor.get_descriptor(text.split(':', 1)[1].strip())
    assert (desc.state_dim, desc.emission_dim, desc.num_timesteps, desc.kappa, desc.omega, desc.rot) == (10, 10, 100, 1.0, 2.0, True)
    assert desc.get_dir() == os.path.join('sinlin', 'state-dim_10_emission-dim_10_num-timesteps_100_kappa_1.0_omega_2.0_rot')
    again = ed.SimulatorDescriptor.get_descriptor(desc.pprint())
    assert again.get_dir() == desc.get_dir()


def test_emission_and_simulation():
    for rot in (False, True):
        d = 6
        ssm, params = _model(d, rot)
        a = sinlin.direction(d, rot)
        assert abs(float(jnp.linalg.norm(a)) - 1.0) < 1e-6
        x = jr.normal(jr.PRNGKey(1), (d,))
        y = ssm.emission_function(params, x, jnp.zeros(d), 0.0)
        expect = jnp.concatenate([jnp.array([KAPPA * jnp.sin(OMEGA * a @ x)]), x[1:]])
        np.testing.assert_allclose(y, expect, rtol=1e-6)
        lp = ssm.emission_log_prob(params, y + 0.1, x, 0.0)
        np.testing.assert_allclose(lp, MVN(expect, 0.1 * jnp.eye(d)).log_prob(y + 0.1), rtol=1e-5)
        states, obs = ssm.simulate(jr.PRNGKey(2), params, 50, sinlin.get_inputs(None, 50))
        assert states.shape == (50, d) and obs.shape == (50, d)
        assert bool(jnp.all(jnp.isfinite(states))) and bool(jnp.all(jnp.isfinite(obs)))
    # the linear dynamics: x_{t+1} = 0.9 x_t + q
    np.testing.assert_allclose(ssm.dynamics_function(params, jnp.ones(d), jnp.zeros(d), 0.0), 0.9 * jnp.ones(d), rtol=1e-6)


def test_sdp_keeps_the_linear_coordinates():
    """rot = 0: Sigma^-1 Delta of 'sdp' is diag(p_1, RHO_MAX, ..., RHO_MAX) with p_1 = alpha / (s^2 omega^2), for each d."""
    p1 = ALPHA / (S2 * OMEGA**2)
    for d in (2, 5, 10, 20):
        _, (Delta, _), P, _ = _rules_at(d)
        ratio = np.asarray(jnp.linalg.solve(P, Delta))
        np.testing.assert_allclose(ratio[0, 0], p1, atol=5e-3)
        np.testing.assert_allclose(np.diag(ratio)[1:], RHO_MAX, atol=5e-3)
        np.testing.assert_allclose(ratio - np.diag(np.diag(ratio)), 0.0, atol=5e-3)


def test_auto_rho_grows_with_d():
    """rot = 0: rho* of Eq. (47) is alpha (d + 1) / (2 s^2 omega^2), clipped at RHO_MAX. Thus it moves away from p_1."""
    for d in (1, 2, 5, 10, 20):
        (Delta, rho), _, P, _ = _rules_at(d)
        expect = min(RHO_MAX, ALPHA * (d + 1) / (2 * S2 * OMEGA**2))
        np.testing.assert_allclose(float(rho), expect, rtol=1e-4)
        np.testing.assert_allclose(Delta, rho * P, rtol=1e-5)
    # d = 1: no linear channel, the two rules agree
    (_, rho), (Delta, _), P, _ = _rules_at(1)
    np.testing.assert_allclose(float(Delta[0, 0] / P[0, 0]), float(rho), atol=5e-3)


def test_rotated_direction():
    """rot = 1: the eigenvalues of Sigma^-1 Delta of 'sdp' spread. The smallest is below the rho of 'auto' and the largest
    is near RHO_MAX, so no multiple of Sigma matches Delta. (The shrunk eigenvector mixes a and e_1: the linear channels
    observe x_2 to x_d, so e_1 has the least linear information.)"""
    for d in (5, 10):
        (_, rho), (Delta, _), P, a = _rules_at(d, rot=True)
        lam = jnp.linalg.eigvalsh(jnp.linalg.solve(P, Delta))
        assert float(lam[0]) < float(rho) - 0.1, (d, lam, rho)
        assert float(lam[-1]) > RHO_MAX - 0.05, (d, lam)


def test_split_error_metric():
    """mse_x1 + (d - 1) mse_rest = mse, for a mixture output of the GSF form."""
    k1, k2, k3 = jr.split(jr.PRNGKey(3), 3)
    K, T, d = 3, 7, 5
    data = {'means': jr.normal(k1, (K, T, d)), 'covariances': jnp.tile(jnp.eye(d), (K, T, 1, 1)),
            'weights': jr.uniform(k2, (K, T))}
    truth = jr.normal(k3, (T, d))
    total, total_t = mse(data, 'lgsf', truth)
    x1, x1_t = mse_x1(data, 'lgsf', truth)
    rest, rest_t = mse_rest(data, 'lgsf', truth)
    np.testing.assert_allclose(x1_t + (d - 1) * rest_t, total_t, rtol=1e-5)
    np.testing.assert_allclose(x1 + (d - 1) * rest, total, rtol=1e-5)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'{t.__name__}: ok')
    print(f'all {len(tests)} tests passed')
