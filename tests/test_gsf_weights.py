"""The weights of the Gaussian sum filters stay finite when all likelihoods underflow in float32.

Exp. A (mtt) with a = 0.5 and sigma2 = 2.5e-5: the weights of the L-GSF and U-GSF collapse to one component. At the first
mode switch (t = 200) the likelihood of that component is far below the largest likelihood. With the old update (shift the
log-likelihoods by their maximum, multiply by the weights in the linear domain) all weights became 0 and then NaN.

Run with pytest, or directly: python tests/test_gsf_weights.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaussfiltax'))

import jax.numpy as jnp
import jax.random as jr

from filters import gsf, agsf
from simulators import mtt
from util.param import sample_prior

T = 500


def _data(seed=4):  # seed 4: the old update gives NaN weights from t = 200 (L-GSF)
    sim = mtt.setup(4, 2, 0, 0.5, 2.5e-5)
    ssm, props = sim['ssm'], sim['props']
    inputs = mtt.get_inputs(jr.PRNGKey(seed), T)
    true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
    _, observations = ssm.simulate(jr.PRNGKey(seed + 1), true_ps, T, inputs)
    return ssm, props, true_ps, observations, inputs


def _check(weights, name):
    w = jnp.asarray(weights)
    bad = int(jnp.sum(~jnp.all(jnp.isfinite(w), axis=0)))
    assert bad == 0, f'{name}: {bad} steps with non-finite weights (first: {int(jnp.argmax(~jnp.all(jnp.isfinite(w), axis=0)))})'
    assert jnp.allclose(jnp.sum(w, axis=0), 1.0, atol=1e-4), f'{name}: the weights do not sum to 1'


def test_lgsf_weights_are_finite():
    ssm, props, true_ps, y, u = _data()
    out = gsf.LGSF(props, ssm).run(true_ps, y, 100, 1, jr.PRNGKey(2), u)
    _check(out['weights'], 'L-GSF')


def test_ugsf_weights_are_finite():
    ssm, props, true_ps, y, u = _data()
    out = gsf.UGSF(props, ssm).run(jr.PRNGKey(2), true_ps, y, 100, 1, 1.0, 0.0, 0.0, u)
    _check(out['weights'], 'U-GSF')


def test_lagsf_weights_are_finite():
    ssm, props, true_ps, y, u = _data()
    out, _ = agsf.LAGSF(props, ssm).run(jr.PRNGKey(2), true_ps, y, 'prop', 0.9, 'prop', 0.9, 10, 5, 5, 1, u)
    _check(out['weights'], 'L-AGSF')
    assert bool(jnp.all(jnp.isfinite(out['means']))), 'L-AGSF: non-finite means'


if __name__ == '__main__':
    for test in (test_lgsf_weights_are_finite, test_ugsf_weights_are_finite, test_lagsf_weights_are_finite):
        try:
            test()
            print('PASS', test.__name__)
        except AssertionError as e:
            print('FAIL', test.__name__, '-', e)
