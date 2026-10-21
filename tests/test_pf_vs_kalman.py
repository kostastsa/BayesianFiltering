"""BPF and APF against the exact Kalman filter on a linear Gaussian model.

Checks the filtering means and the log-likelihood estimates. The one-particle test in
test_apf.py cannot detect errors in the likelihood estimate, because with one particle
all the weights and normalizing constants are trivial.

Run with pytest, or directly: python tests/test_pf_vs_kalman.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))
from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
from jax import lax

from simulators.ssm import NonlinearSSM
from filters import pf

_leaf = lambda v: SimpleNamespace(value=jnp.asarray(v))

F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
H = jnp.array([[1.0, 0.0]])
Q = 0.05 * jnp.eye(2)
R = 0.5 * jnp.eye(1)
M0, P0 = jnp.zeros(2), jnp.eye(2)
T = 50


def _model():
    params = SimpleNamespace(
        initial=SimpleNamespace(mean=_leaf(M0), cov=_leaf(P0)),
        dynamics=SimpleNamespace(bias=_leaf(jnp.zeros(2)), cov=_leaf(Q)),
        emissions=SimpleNamespace(bias=_leaf(jnp.zeros(1)), cov=_leaf(R)),
    )
    ssm = NonlinearSSM(2, 2, 1, 1,
                       lambda x, q, u, p: F @ x + q,
                       lambda x, r, u, p: H @ x + r)
    return params, ssm


def _observations():
    k1, k2 = jr.split(jr.PRNGKey(1))
    qs = jr.multivariate_normal(k1, jnp.zeros(2), Q, (T,))
    rs = jr.multivariate_normal(k2, jnp.zeros(1), R, (T,))
    x, ys = jr.multivariate_normal(jr.PRNGKey(2), M0, P0), []
    for t in range(T):
        x = F @ x + qs[t]
        ys.append(H @ x + rs[t])
    return jnp.stack(ys)


def _kalman(ys):
    # Same timing as the particle filters: predict, then update, from t = 0.
    def step(c, y):
        m, P, ll = c
        m, P = F @ m, F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ jnp.linalg.inv(S)
        e = y - H @ m
        ll += -0.5 * (e @ jnp.linalg.solve(S, e) + jnp.log(jnp.linalg.det(2 * jnp.pi * S)))
        return (m + K @ e, (jnp.eye(2) - K @ H) @ P, ll), m + K @ e

    (_, _, ll), ms = lax.scan(step, (M0, P0, 0.0), ys)
    return ms, ll


def _check(filter_cls, num_particles=5000, mean_tol=0.03, ll_tol=0.5, ess_threshold=0.5, n_steps=T):
    params, ssm = _model()
    ys = _observations()[:n_steps]
    ms_kf, ll_kf = _kalman(ys)

    outputs, ll = filter_cls(None, ssm).run(params, ys, num_particles, jr.PRNGKey(0), None, ess_threshold)
    mean = jnp.einsum('it,itd->td', outputs['weights'], outputs['particles'])
    rmse = float(jnp.sqrt(jnp.mean((mean - ms_kf) ** 2)))

    assert rmse < mean_tol, f'filtering mean RMSE {rmse:.4f} against the Kalman filter'
    assert abs(float(ll) - float(ll_kf)) < ll_tol, f'log-likelihood {float(ll):.3f}, Kalman {float(ll_kf):.3f}'


def test_bpf_matches_kalman():
    _check(pf.BPF)


def test_apf_matches_kalman():
    _check(pf.APF)


def test_apf_resampling_at_every_step_matches_kalman():
    # An ESS threshold above 1 forces the resampling of the ancestors at every step.
    _check(pf.APF, ess_threshold=2.0)


def test_apf_without_resampling_matches_kalman():
    # An ESS threshold of 0 never resamples, so the APF reduces to the BPF without resampling.
    # The weights degenerate over a long horizon, so the test uses only 5 steps.
    _check(pf.APF, ess_threshold=0.0, n_steps=5)


if __name__ == '__main__':
    test_bpf_matches_kalman()
    test_apf_matches_kalman()
    test_apf_resampling_at_every_step_matches_kalman()
    test_apf_without_resampling_matches_kalman()
    print('BPF and APF agree with the Kalman filter.')
