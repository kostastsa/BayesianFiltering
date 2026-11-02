"""Tests of util.numerics.nees (the NEES metric of 'main.py errors').

With the true state drawn from the filter posterior N(m_t, P_t), the fraction of steps with NEES > chi2_4(0.999) is
about 0.001. The test checks this for the three formats of the saved results: one Gaussian component (GSF), a mixture
of two components with the same moments (same NEES), and particles drawn from N(m_t, P_t) (PF). A posterior far from
the truth gives a fraction near 1.

Run directly: python tests/test_nees.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))
import numpy as np
from util.numerics import nees

rng = np.random.default_rng(0)
T, d = 20000, 4
A = rng.normal(size=(T, d, d)) * 0.5
P = A @ np.swapaxes(A, 1, 2) + 0.1 * np.eye(d)
m = rng.normal(size=(T, d)) * 3.0
L = np.linalg.cholesky(P)
x = m + np.einsum('tij,tj->ti', L, rng.normal(size=(T, d)))


def test_one_component():
    frac, n1 = nees({'means': m[None], 'covariances': P[None], 'weights': np.ones((1, T))}, 'lgsf', x)
    assert 0.0004 < frac < 0.002, frac
    # NEES of the exact posterior has mean d
    assert abs(float(np.mean(n1)) - d) < 0.1, float(np.mean(n1))
    print(f'PASS test_one_component (fraction {frac:.4f}, mean NEES {float(np.mean(n1)):.3f})')
    return n1


def test_mixture_same_moments(n1):
    # Two components m +- delta with covariance P - delta delta^T: the mixture has mean m and covariance P
    delta = 0.3 * L[:, :, 0]
    S = P - np.einsum('ti,tj->tij', delta, delta)
    data = {'means': np.stack([m + delta, m - delta]), 'covariances': np.stack([S, S]), 'weights': np.full((2, T), 0.5)}
    _, n2 = nees(data, 'uagsf', x)
    assert np.allclose(np.asarray(n2), np.asarray(n1), rtol=1e-6, atol=1e-8)
    print('PASS test_mixture_same_moments')


def test_particles():
    K = 200
    parts = m[None] + np.einsum('tij,ktj->kti', L, rng.normal(size=(K, T, d)))
    frac, _ = nees([{'particles': parts, 'weights': np.ones((K, T))}], 'bpf', x)
    # The sample covariance of 200 particles adds a little noise, thus a slightly larger fraction
    assert frac < 0.004, frac
    print(f'PASS test_particles (fraction {frac:.4f})')


def test_far_from_truth():
    shifted = m + 10.0 * L[:, :, 0]
    frac, _ = nees({'means': shifted[None], 'covariances': P[None], 'weights': np.ones((1, T))}, 'lgsf', x)
    assert frac > 0.99, frac
    print(f'PASS test_far_from_truth (fraction {frac:.4f})')


if __name__ == '__main__':
    n1 = test_one_component()
    test_mixture_same_moments(n1)
    test_particles()
    test_far_from_truth()
    print('The NEES metric works.')
