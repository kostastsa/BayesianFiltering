"""The error metrics use the weights of the particles and of the mixture components.

Run with pytest, or directly: python tests/test_numerics_weights.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))

import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.scipy.special import logsumexp as lse

from util import numerics

EPS = 1e-6
W = jnp.array([[0.75], [0.25]])           # (n_prt, T)
X = jnp.array([[[0.0]], [[1.0]]])         # (n_prt, T, d)
TRUE = jnp.array([[0.3]])                 # (T, d)


def _cloud(weights):
    return ({'particles': X, 'weights': weights}, 0.0)


def test_mse_and_bias_use_weights():
    mean = 0.75 * 0.0 + 0.25 * 1.0
    mse, _ = numerics.mse(_cloud(W), 'bpf', TRUE)
    bias, _ = numerics.bias(_cloud(W), 'bpf', TRUE)
    assert abs(float(mse) - (mean - 0.3) ** 2) < 1e-6
    assert abs(float(bias) - abs(mean - 0.3)) < 1e-6


def test_equal_weights_give_the_plain_mean():
    mse, _ = numerics.mse(_cloud(jnp.full((2, 1), 0.5)), 'bpf', TRUE)
    assert abs(float(mse) - (0.5 - 0.3) ** 2) < 1e-6


def test_log_prob_error_of_a_weighted_cloud():
    # Weighted covariance 0.1875 / (1 - sum w^2) = 0.5. Each kernel has the covariance 0.5 / n_prt.
    var = 0.5 / 2 + 2 * EPS
    logp = lse(jnp.log(W[:, 0]) + jss.norm.logpdf(0.3, X[:, 0, 0], jnp.sqrt(var)))
    lpe, _ = numerics.log_prob_error(_cloud(W), 'bpf', TRUE)
    assert abs(float(lpe) + float(logp)) < 1e-3, f'{float(lpe)} against {-float(logp)}'


def test_mixture_weights_are_used():
    # A Gaussian sum filter output with unequal weights, as in LGSF and UGSF.
    out = {'means': X, 'covariances': jnp.full((2, 1, 1, 1), 0.2), 'weights': W}
    mse, _ = numerics.mse(out, 'lgsf', TRUE)
    lpe, _ = numerics.log_prob_error(out, 'lgsf', TRUE)
    logp = lse(jnp.log(W[:, 0]) + jss.norm.logpdf(0.3, X[:, 0, 0], jnp.sqrt(0.2 + EPS)))
    assert abs(float(mse) - (0.25 - 0.3) ** 2) < 1e-6
    assert abs(float(lpe) + float(logp)) < 1e-4


if __name__ == '__main__':
    test_mse_and_bias_use_weights()
    test_equal_weights_give_the_plain_mean()
    test_log_prob_error_of_a_weighted_cloud()
    test_mixture_weights_are_used()
    print('The metrics use the weights.')
