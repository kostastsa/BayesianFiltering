from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr

import gaussfiltax.filters.pf as pf_module
from gaussfiltax.filters.pf import APF


def _leaf(value):
    return SimpleNamespace(value=jnp.asarray(value))


def _base_params():
    return SimpleNamespace(
        initial=SimpleNamespace(
            mean=_leaf(jnp.array([0.0])),
            cov=_leaf(jnp.array([[1e-9]])),
        ),
        dynamics=SimpleNamespace(
            bias=_leaf(jnp.array([0.0])),
            cov=_leaf(jnp.array([[1e-9]])),
        ),
        emissions=SimpleNamespace(
            bias=_leaf(jnp.array([0.0])),
            cov=_leaf(jnp.array([[1.0]])),
        ),
    )


class _DeterministicSSM:
    input_dim = 0

    def dynamics_function(self, params, prev_state, noise_vec, inputs):
        return prev_state + noise_vec

    def dynamics_simulator(self, key, params, state, inputs):
        # Deterministic propagation so ll can be checked exactly in the 1-particle test.
        return state

    def emission_log_prob(self, params, emission, state, inputs):
        diff = emission - state
        return -0.5 * jnp.sum(diff * diff)


def test_apf_no_second_resampling(monkeypatch):
    params = _base_params()
    apf = APF(props=None, ssm=_DeterministicSSM())

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("APF should not call util.resample after ancestor resampling")

    monkeypatch.setattr(pf_module, "resample", _raise_if_called)

    observations = jnp.array([[0.0]])
    outputs, ll = apf.run(
        params=params,
        observations=observations,
        num_particles=8,
        key=jr.PRNGKey(0),
        ess_threshold=10.0,  # would force old implementation's second resample
    )

    assert jnp.isfinite(ll)
    assert jnp.allclose(jnp.sum(outputs["weights"], axis=0), 1.0)


def test_apf_one_particle_loglik_matches_true_loglik():
    params = _base_params()
    apf = APF(props=None, ssm=_DeterministicSSM())

    # With one particle and deterministic dynamics, the APF incremental ll should
    # reduce exactly to log p(y_t | x_t).
    observations = jnp.array([[2.0]])
    _, ll = apf.run(
        params=params,
        observations=observations,
        num_particles=1,
        key=jr.PRNGKey(0),
    )

    expected = -0.5 * (2.0 ** 2)
    assert jnp.allclose(ll, expected, atol=1e-6)
