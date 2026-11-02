"""Tests of the optional emission residual (D-30): the wrapped bearing of mtt.

1. The new code path with residual(a, b) = a - b gives the same results as the old path (residual None),
   for the EKF and UKF update steps and for the GSF, AGSF and particle filters on a linear Gaussian model.
2. At the branch cut (component just above the negative x1-axis, observation just below), the wrapped update moves
   the component very little, and the raw update moves it far.
3. The wrapped likelihood of mtt is continuous across the cut, and the simulator wraps the bearing.

Run directly: python tests/test_wrapped_bearing.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))
from functools import partial
import jax, jax.numpy as jnp, jax.random as jr
from jax import jacfwd

from filters.steps import _sqrt_condition_on, _ukf_sqrt_condition_on_nonadditive
from filters import gsf, agsf, pf
from simulators import mtt
from util.param import sample_prior
import test_pf_vs_kalman as lin

SUB = lambda a, b: a - b


def _close(a, b, tol=1e-4):
    return all(bool(jnp.allclose(x, y, atol=tol, rtol=tol)) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_subtraction_residual_equals_old_path():
    params, ssm = lin._model()
    ys = lin._observations()
    ref = {}
    for res in [None, SUB]:
        ssm.emission_residual = res
        out = {
            'lgsf': gsf.LGSF(None, ssm).run(params, ys, 5, 1, jr.PRNGKey(0)),
            'ugsf': gsf.UGSF(None, ssm).run(jr.PRNGKey(0), params, ys, 5, 1, 1.0, 0.0, 0.0),
            'lagsf': agsf.LAGSF(None, ssm).run(jr.PRNGKey(0), params, ys, 'prop', 0.9, 'prop', 0.9, 5, 3, 3, 1)[0],
            'bpf': pf.BPF(None, ssm).run(params, ys, 500, jr.PRNGKey(0))[0],
            'upf': pf.UPF(None, ssm).run(params, ys, 500, jr.PRNGKey(0))[0],
        }
        ref[res is None] = out
    for k in ref[True]:
        assert _close(ref[True][k], ref[False][k]), k
    print('PASS test_subtraction_residual_equals_old_path')


def _mtt_update(wrap, ukf, sd=0.002):
    setup = mtt.setup(4, 2, 1, 0.05, 2.5e-5, wrap)
    ssm, props = setup['ssm'], setup['props']
    params = sample_prior(jr.PRNGKey(0), props, 1)[0]
    h = lambda x, r, u: ssm.emission_function(params, x, r, u)
    R, r0 = params.emissions.cov.value, params.emissions.bias.value
    # Component just above the negative x1-axis (bearing +pi - 0.005), observation just below (bearing -pi + 0.005)
    m = jnp.array([-12.0, 0.0, 0.06, 0.0])
    S = sd * jnp.eye(4)
    y = jnp.array([-jnp.pi + 0.005, 12.0])
    res = ssm.emission_residual
    if ukf:
        _, mu, _ = _ukf_sqrt_condition_on_nonadditive(m, S, h, R, 0, y, 1.0, 0.0, 0.0, r0, res)
    else:
        H_x, H_r = jacfwd(h, argnums=0), jacfwd(h, argnums=1)
        _, mu, _, _, _ = _sqrt_condition_on(m, S, h, H_x, H_r, R, r0, 0, y, res)
    return float(jnp.linalg.norm(mu - m))


def _ukf_bearing_var(wrap, sd):
    """Predicted bearing variance of the UKF update for the component of _mtt_update."""
    setup = mtt.setup(4, 2, 1, 0.05, 2.5e-5, wrap)
    ssm, props = setup['ssm'], setup['props']
    params = sample_prior(jr.PRNGKey(0), props, 1)[0]
    h = lambda x, r, u: ssm.emission_function(params, x, r, u)
    m = jnp.array([-12.0, 0.0, 0.06, 0.0])
    # Unscented variance of the bearing (2d sigma points, lambda = 0), with the residual of the model
    L = sd * jnp.eye(4) * 2.0
    pts = jnp.concatenate([m + L, m - L])
    res = ssm.emission_residual or SUB
    yb = jax.vmap(lambda x: h(x, jnp.zeros(2), 0))(pts)
    yc = h(m, jnp.zeros(2), 0)
    mu = yc + jnp.mean(jax.vmap(res, in_axes=(0, None))(yb, yc), axis=0)
    d = jax.vmap(res, in_axes=(0, None))(yb, mu)
    return float(jnp.mean(d[:, 0] ** 2))


def test_update_at_the_cut():
    # Innovation-wrap mode: the component (sd 0.002) is fully on one side of the cut, the observation on the other.
    for ukf in [False, True]:
        raw, wrapped = _mtt_update(False, ukf), _mtt_update(True, ukf)
        name = 'UKF' if ukf else 'EKF'
        print(f'  {name}: |update| raw {raw:.3f}, wrapped {wrapped:.3f}')
        assert wrapped < 0.2, f'{name}: the wrapped update moves the component by {wrapped}'
        assert raw > 5 * wrapped, f'{name}: the raw update does not show the fault'
    # Blind mode: a wide component (sd 0.05) has sigma points on both sides of the cut. The raw bearing variance is
    # of order pi^2, thus the update ignores the bearing. The wrapped variance stays small.
    v_raw, v_wrap = _ukf_bearing_var(False, 0.05), _ukf_bearing_var(True, 0.05)
    print(f'  UKF bearing variance, sigma points on both sides: raw {v_raw:.3g}, wrapped {v_wrap:.3g}')
    assert v_raw > 1.0 and v_wrap < 1e-3
    print('PASS test_update_at_the_cut')


def test_likelihood_and_simulation():
    setup = mtt.setup(4, 2, 1, 0.05, 2.5e-5, True)
    ssm, props = setup['ssm'], setup['props']
    params = sample_prior(jr.PRNGKey(0), props, 1)[0]
    y = jnp.array([jnp.pi - 0.001, 12.0])
    above = ssm.emission_log_prob(params, y, jnp.array([-12.0, 0.0, 0.006, 0.0]), 0)
    below = ssm.emission_log_prob(params, y, jnp.array([-12.0, 0.0, -0.006, 0.0]), 0)
    assert abs(float(above) - float(below)) < 20.0, (above, below)
    states, obs = ssm.simulate(jr.PRNGKey(1), params, 500, mtt.get_inputs(None, 500))
    assert bool(jnp.all((obs[:, 0] >= -jnp.pi) & (obs[:, 0] < jnp.pi)))
    raw = mtt.setup(4, 2, 1, 0.05, 2.5e-5, False)['ssm']
    states_r, obs_r = raw.simulate(jr.PRNGKey(1), params, 500, mtt.get_inputs(None, 500))
    # Same key: the same states, and the observations differ only by the wrap
    assert bool(jnp.allclose(states, states_r))
    assert bool(jnp.allclose(obs[:, 0], mtt.wrap_angle(obs_r[:, 0]), atol=1e-5)) and bool(jnp.allclose(obs[:, 1], obs_r[:, 1]))
    print('PASS test_likelihood_and_simulation')


if __name__ == '__main__':
    test_subtraction_residual_equals_old_path()
    test_update_at_the_cut()
    test_likelihood_and_simulation()
    print('The wrapped bearing works.')
