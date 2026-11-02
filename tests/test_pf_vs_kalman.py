"""BPF, APF, RPF and UPF against the exact Kalman filter on a linear Gaussian model.

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
# Singular process noise, as in Exp. A (x' = F x + G q, q scalar): the transition density has no density in 2D.
G = jnp.array([[0.5], [1.0]])
Q1 = 0.05 * jnp.eye(1)


def _model(singular=False):
    Qn = Q1 if singular else Q
    params = SimpleNamespace(
        initial=SimpleNamespace(mean=_leaf(M0), cov=_leaf(P0)),
        dynamics=SimpleNamespace(bias=_leaf(jnp.zeros(Qn.shape[0])), cov=_leaf(Qn)),
        emissions=SimpleNamespace(bias=_leaf(jnp.zeros(1)), cov=_leaf(R)),
    )
    f = (lambda x, q, u, p: F @ x + G @ q) if singular else (lambda x, q, u, p: F @ x + q)
    ssm = NonlinearSSM(2, Qn.shape[0], 1, 1, f, lambda x, r, u, p: H @ x + r)
    return params, ssm


def _observations(singular=False):
    k1, k2 = jr.split(jr.PRNGKey(1))
    qs = jr.multivariate_normal(k1, jnp.zeros(2), Q, (T,)) if not singular else jr.multivariate_normal(k1, jnp.zeros(1), Q1, (T,)) @ G.T
    rs = jr.multivariate_normal(k2, jnp.zeros(1), R, (T,))
    x, ys = jr.multivariate_normal(jr.PRNGKey(2), M0, P0), []
    for t in range(T):
        x = F @ x + qs[t]
        ys.append(H @ x + rs[t])
    return jnp.stack(ys)


def _kalman(ys, singular=False):
    Qx = G @ Q1 @ G.T if singular else Q
    # Same timing as the particle filters: predict, then update, from t = 0.
    def step(c, y):
        m, P, ll = c
        m, P = F @ m, F @ P @ F.T + Qx
        S = H @ P @ H.T + R
        K = P @ H.T @ jnp.linalg.inv(S)
        e = y - H @ m
        ll += -0.5 * (e @ jnp.linalg.solve(S, e) + jnp.log(jnp.linalg.det(2 * jnp.pi * S)))
        return (m + K @ e, (jnp.eye(2) - K @ H) @ P, ll), m + K @ e

    (_, _, ll), ms = lax.scan(step, (M0, P0, 0.0), ys)
    return ms, ll


def _check(filter_cls, num_particles=5000, mean_tol=0.03, ll_tol=0.5, ess_threshold=0.5, n_steps=T, singular=False):
    params, ssm = _model(singular)
    ys = _observations(singular)[:n_steps]
    ms_kf, ll_kf = _kalman(ys, singular)

    outputs, ll = filter_cls(None, ssm).run(params, ys, num_particles, jr.PRNGKey(0), None, ess_threshold)
    mean = jnp.einsum('it,itd->td', outputs['weights'], outputs['particles'])
    rmse = float(jnp.sqrt(jnp.mean((mean - ms_kf) ** 2)))

    print(f'{filter_cls.__name__:4s} singular={singular!s:5s} ess={ess_threshold}: RMSE {rmse:.4f}, ll {float(ll):.3f}, Kalman ll {float(ll_kf):.3f}')
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


def test_rpf_matches_kalman():
    # The kernel adds the variance h^2 C after each resampling (h = 0.24 for N = 5000, d = 2), thus a small bias.
    _check(pf.RPF, mean_tol=0.05, ll_tol=1.0)


def test_upf_matches_kalman():
    # In a linear Gaussian model the UKF over the noise is exact, so the proposal is the optimal one.
    _check(pf.UPF, mean_tol=0.02, ll_tol=0.2)


def test_bpf_singular_noise_matches_kalman():
    _check(pf.BPF, singular=True)


def test_upf_singular_noise_matches_kalman():
    # Process noise through G (2 x 1): the noise-space UPF (D-26) needs no transition density.
    _check(pf.UPF, mean_tol=0.02, ll_tol=0.2, singular=True)


if __name__ == '__main__':
    test_bpf_matches_kalman()
    test_apf_matches_kalman()
    test_apf_resampling_at_every_step_matches_kalman()
    test_apf_without_resampling_matches_kalman()
    test_rpf_matches_kalman()
    test_upf_matches_kalman()
    test_bpf_singular_noise_matches_kalman()
    test_upf_singular_noise_matches_kalman()
    print('BPF, APF, RPF and UPF agree with the Kalman filter.')
