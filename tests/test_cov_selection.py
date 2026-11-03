"""Checks of the covariance rules of the AGSF (filters/agsf.py): the output rescale (plan step 1) and the solver of the
full program (45) (plan step 2). The plan is in 'TSP Resubmission review/implementation-improvement-plan.md'.

Run with pytest, or directly: python tests/test_cov_selection.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from filters import agsf
from filters.agsf import (solve_cov_sdp, sdp_objective, sdp_gradient, sdp_fw_gap, auto_cov_selection,
                          sdp_cov_selection, RHO_MAX, RHO_MIN)
from simulators import lmsv, mtt
from util.param import sample_prior


def _derivs(f):
    """jacobian(x, bias, u), hessian_tensor(x, bias, u) and noise_jacobian(x, bias, u) of f(x, r, u)."""
    return jax.jacfwd(f, 0), jax.jacrev(jax.jacfwd(f, 0)), jax.jacfwd(f, 1)


def _random_problem(seed, d, dy, c=0.2):
    k1, k2, k3 = jr.split(jr.PRNGKey(seed), 3)
    J = jr.normal(k1, (dy, d))
    H = jnp.exp(2 * jr.normal(k3)) * jr.normal(k2, (dy, d, d))
    H = 0.5 * (H + jnp.swapaxes(H, 1, 2))
    return J.T @ J, H, c


def _scale(A, H, c):
    gram = jnp.einsum('iab,jab->ij', H, H)
    return float(c * jnp.max(jnp.linalg.eigvalsh(A)) + 0.5 * jnp.max(jnp.linalg.eigvalsh(gram)))


def _rho_star(A, H, c):
    a, b = jnp.trace(A), jnp.sum(jnp.trace(H, axis1=1, axis2=2)**2)
    return float(jnp.minimum(2 * c * a / b, 1.0))


# Plan step 2: the solver of (45)

def test_fw_gap_is_small():
    """The Frank-Wolfe gap bounds Psi(P) - min Psi. After the fixed number of iterations it is small (float32)."""
    for d, dy in [(1, 1), (2, 2), (4, 2), (4, 4), (5, 3)]:
        for seed in range(20):
            A, H, c = _random_problem(seed, d, dy)
            P = solve_cov_sdp(A, H, c)
            gap = float(sdp_fw_gap(P, sdp_gradient(P, A, H, c)))
            assert gap <= 1e-3 * _scale(A, H, c), (d, dy, seed, gap)


def test_one_dimensional_quadratic():
    """f(x) = x^2/2, mu = 1, sigma2 = 1, N = 5: Psi = (1 - delta)/N + delta^2/4, so delta* = 2/N = 0.4."""
    f = lambda x, r, u: 0.5 * x**2 + r
    jac, hess, _ = _derivs(f)
    A = jnp.array([[1.0]])
    H = hess(jnp.array([1.0]), jnp.zeros(1), 0.0)
    P = solve_cov_sdp(A, H, 1 / 5)
    assert jnp.allclose(P, 0.4, atol=1e-4), P
    cov, rho = sdp_cov_selection(jnp.array([1.0]), jnp.eye(1), jac, hess, jnp.zeros(1), 0.0, 2 / 5)
    assert jnp.allclose(cov, 0.4, atol=1e-4) and jnp.allclose(rho, 0.4, atol=1e-4), (cov, rho)


def _range_bearing(p, r, u):
    return jnp.array([jnp.arctan2(p[1], p[0]), jnp.sqrt(p[0]**2 + p[1]**2)]) + r


def test_range_bearing_cigar():
    """Range-bearing, Sigma = s^2 I: Delta* = s^2 u u^T + min(2/N, s^2) u_perp u_perp^T (agsf_theorems.md, II.2).
    s = 1, r = 2, N = 5: Delta* = diag(1, 0.4) in the basis (u, u_perp)."""
    jac, hess, _ = _derivs(_range_bearing)
    for angle in [0.0, 0.7, 2.5, -1.9]:
        u = jnp.array([jnp.cos(angle), jnp.sin(angle)])
        u_perp = jnp.array([-u[1], u[0]])
        m = 2.0 * u
        J, H = jac(m, jnp.zeros(2), 0.0), hess(m, jnp.zeros(2), 0.0)
        P = solve_cov_sdp(J.T @ J, H, 1 / 5)
        B = jnp.stack([u, u_perp], axis=1)
        assert jnp.allclose(B.T @ P @ B, jnp.diag(jnp.array([1.0, 0.4])), atol=2e-3), (angle, B.T @ P @ B)
        # In the filter, the eigenvalues are clipped to [RHO_MIN, RHO_MAX]
        cov, _ = sdp_cov_selection(m, jnp.eye(2), jac, hess, jnp.zeros(2), 0.0, 2 / 5)
        assert jnp.allclose(B.T @ cov @ B, jnp.diag(jnp.array([RHO_MAX, 0.4])), atol=2e-3), (angle, B.T @ cov @ B)


def test_full_program_not_worse_than_rho_star():
    """The ray {rho Sigma} is in the feasible set of (45), so Psi(Delta*) <= Psi(rho* Sigma)."""
    for d, dy in [(2, 2), (4, 2), (5, 3)]:
        for seed in range(20):
            A, H, c = _random_problem(seed, d, dy)
            P = solve_cov_sdp(A, H, c)
            rs = _rho_star(A, H, c)
            f_sdp = float(sdp_objective(P, A, H, c))
            f_rho = float(sdp_objective(rs * jnp.eye(d), A, H, c))
            assert f_sdp <= f_rho + 1e-5 * _scale(A, H, c), (d, dy, seed, f_sdp, f_rho)


def test_limit_cases():
    """Affine f (H = 0): Delta* = Sigma. Zero Jacobian (J = 0) with positive definite H_i: Delta* = 0."""
    d = 3
    A, _, c = _random_problem(1, d, d)
    P = solve_cov_sdp(A, jnp.zeros((2, d, d)), c)
    assert jnp.allclose(P, jnp.eye(d), atol=1e-4), P

    k = jr.normal(jr.PRNGKey(2), (2, d, d))
    H = jnp.einsum('kab,kcb->kac', k, k) + 0.1 * jnp.eye(d)
    P = solve_cov_sdp(jnp.zeros((d, d)), H, c)
    assert jnp.allclose(P, 0.0, atol=1e-4), P


def test_sdp_selection_is_feasible():
    """In the filter, Sigma - Delta and Delta are positive definite (Cholesky factors exist)."""
    sim = mtt.setup(4, 2, 0, 0.5, 2.5)
    ssm, props = sim['ssm'], sim['props']
    params = sample_prior(jr.PRNGKey(0), props, 1)[0]
    f = lambda x, q, u: ssm.dynamics_function(params, x, q, u)
    jac, hess, _ = _derivs(f)
    for seed in range(10):
        k1, k2 = jr.split(jr.PRNGKey(seed))
        m = jnp.array([-0.05, 0.001, 0.7, -0.05]) + 0.3 * jr.normal(k1, (4,))
        L = 0.3 * jr.normal(k2, (4, 4))
        P = L @ L.T + 0.01 * jnp.eye(4)
        for u in [0, 1, 2]:
            cov, rho = sdp_cov_selection(m, P, jac, hess, jnp.zeros(2), u, 0.4)
            assert jnp.all(jnp.isfinite(jnp.linalg.cholesky(cov))), (seed, u)
            assert jnp.all(jnp.isfinite(jnp.linalg.cholesky(P - cov))), (seed, u)
            lam = jnp.linalg.eigvals(jnp.linalg.solve(P, cov)).real
            assert jnp.all(lam >= RHO_MIN - 1e-4) and jnp.all(lam <= RHO_MAX + 1e-4), (seed, u, lam)


# Plan step 1: the rescale of the outputs

def test_rescale_unit_invariance():
    """With the rescale, rho* and Delta* do not change under f -> diag(c) f (and R -> diag(c) R diag(c)).
    Without the rescale, rho* changes (the range-bearing sensor adds rad^2 and units^2, agsf_theorems.md, II.1)."""
    scale = jnp.array([1.0, 0.1])  # the range in units ten times larger
    g1 = _range_bearing
    g2 = lambda p, r, u: scale * _range_bearing(p, r / scale, u)
    R1 = jnp.diag(jnp.array([0.01, 4.0]))
    R2 = jnp.diag(scale) @ R1 @ jnp.diag(scale)
    m, P = jnp.array([0.6, 0.8]), jnp.array([[20.0, 3.0], [3.0, 10.0]])  # rho* below the clip
    out = {}
    for name, g, R in [('1', g1, R1), ('2', g2, R2)]:
        jac, hess, njac = _derivs(g)
        for rescale in [True, False]:
            out[name, rescale, 'auto'] = auto_cov_selection(m, P, jac, hess, jnp.zeros(2), 0.0, 0.4, njac, R, rescale)
            out[name, rescale, 'sdp'] = sdp_cov_selection(m, P, jac, hess, jnp.zeros(2), 0.0, 0.4, njac, R, rescale)
    for rule in ['auto', 'sdp']:
        assert jnp.allclose(out['1', True, rule][0], out['2', True, rule][0], rtol=1e-3, atol=1e-5), rule
    assert not jnp.allclose(out['1', False, 'auto'][1], out['2', False, 'auto'][1], rtol=1e-2)


def test_rescale_does_not_change_the_experiments():
    """Exp. A (R = sigma2 I, additive) and Exp. B (R = 0.1 I, MSV with multiplicative noise): the rescale changes
    the derivatives by a factor that is the same for all outputs, so rho* does not change. Thus the 'auto' rows of
    Table II and of the O3 pilot stay valid."""
    cases = []
    sim = lmsv.setup(4, 4, 1)
    params = sample_prior(jr.PRNGKey(0), sim['props'], 1)[0]
    g = lambda x, r, u, ssm=sim['ssm'], params=params: ssm.emission_function(params, x, r, u)
    cases.append((g, params.emissions.cov.value, params.emissions.bias.value, 4, [1.0]))
    sim = mtt.setup(4, 2, 0, 0.5, 2.5, wrap=True)
    params = sample_prior(jr.PRNGKey(0), sim['props'], 1)[0]
    g = lambda x, r, u, ssm=sim['ssm'], params=params: ssm.emission_function(params, x, r, u)
    cases.append((g, params.emissions.cov.value, params.emissions.bias.value, 4, [0, 1, 2]))
    for g, R, r0, d, inputs in cases:
        jac, hess, njac = _derivs(g)
        for seed in range(5):
            k1, k2 = jr.split(jr.PRNGKey(seed))
            m = jr.normal(k1, (d,))
            L = 0.5 * jr.normal(k2, (d, d))
            P = L @ L.T + 0.01 * jnp.eye(d)
            for u in inputs:
                _, rho_new = auto_cov_selection(m, P, jac, hess, r0, u, 0.4, njac, R, True)
                _, rho_old = auto_cov_selection(m, P, jac, hess, r0, u, 0.4, njac, R, False)
                assert jnp.allclose(rho_new, rho_old, rtol=1e-4), (seed, u, rho_new, rho_old)


def test_filters_run_with_sdp():
    """Short L-AGSF and U-AGSF runs on Exp. A with ac1 = ac2 = 'sdp': the outputs are finite."""
    sim = mtt.setup(4, 2, 0, 0.5, 2.5, wrap=True)
    ssm, props = sim['ssm'], sim['props']
    params = sample_prior(jr.PRNGKey(0), props, 1)[0]
    T = 20
    inputs = mtt.get_inputs(jr.PRNGKey(1), T)
    _, y = ssm.simulate(jr.PRNGKey(2), params, T, inputs)
    out, aux = agsf.LAGSF(props, ssm).run(jr.PRNGKey(3), params, y, 'sdp', 0.4, 'sdp', 0.4, 5, 3, 3, 1, inputs)
    assert jnp.all(jnp.isfinite(out['means'])) and jnp.all(jnp.isfinite(aux['rho2s']))
    out, aux = agsf.UAGSF(props, ssm).run(jr.PRNGKey(3), params, y, 'sdp', 0.4, 'sdp', 0.4, 5, 3, 3, 1,
                                          1.0, 0.0, 0.0, inputs)
    assert jnp.all(jnp.isfinite(out['means'])) and jnp.all(jnp.isfinite(aux['rho2s']))


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
