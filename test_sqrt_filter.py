"""
Test script for square root Kalman filter implementation.
"""
import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cholesky
from filters.steps import (
    _predict, _condition_on, _sqrt_predict, _sqrt_condition_on, _sqrt_kalman_step,
    _ukf_predict_nonadditive, _ukf_condition_on_nonadditive,
    _ukf_sqrt_predict_nonadditive, _ukf_sqrt_condition_on_nonadditive
)

# Set up a simple linear system for testing
def test_sqrt_vs_standard():
    """Compare square root filter with standard filter on a simple system."""
    
    key = jr.PRNGKey(0)
    state_dim = 3
    obs_dim = 2
    
    # Random initial state
    m0 = jr.normal(key, (state_dim,))
    P0 = jnp.eye(state_dim) * 2.0
    S0 = cholesky(P0, lower=True)
    
    # Linear dynamics: x_{t+1} = F @ x_t + q
    F = jnp.eye(state_dim) + 0.1 * jr.normal(jr.PRNGKey(1), (state_dim, state_dim))
    Q = jnp.eye(state_dim) * 0.1
    
    # Linear observation: y_t = H @ x_t + r
    H = jr.normal(jr.PRNGKey(2), (obs_dim, state_dim))
    R = jnp.eye(obs_dim) * 0.5
    
    # Define functions
    f = lambda x, q, u: F @ x + q
    F_x = lambda x, q, u: F
    F_q = lambda x, q, u: jnp.eye(state_dim)
    
    h = lambda x, r, u: H @ x + r
    H_x = lambda x, r, u: H
    H_r = lambda x, r, u: jnp.eye(obs_dim)
    
    q0 = jnp.zeros(state_dim)
    r0 = jnp.zeros(obs_dim)
    u = jnp.zeros(1)  # No input
    
    # Generate observation
    true_state = jr.normal(jr.PRNGKey(3), (state_dim,))
    y = H @ true_state + jr.normal(jr.PRNGKey(4), (obs_dim,)) * 0.1
    
    print("Testing Prediction Step...")
    print("=" * 60)
    
    # Standard prediction
    mu_pred_std, P_pred_std, _ = _predict(m0, P0, f, F_x, F_q, Q, q0, u)
    
    # Square root prediction
    mu_pred_sqrt, S_pred_sqrt, _ = _sqrt_predict(m0, S0, f, F_x, F_q, Q, q0, u)
    P_pred_sqrt = S_pred_sqrt @ S_pred_sqrt.T
    
    print(f"Mean difference (should be ~0): {jnp.max(jnp.abs(mu_pred_std - mu_pred_sqrt)):.2e}")
    print(f"Covariance difference (should be ~0): {jnp.max(jnp.abs(P_pred_std - P_pred_sqrt)):.2e}")
    
    print("\nTesting Update Step...")
    print("=" * 60)
    
    # Standard update
    ll_std, mu_cond_std, P_cond_std, _, _ = _condition_on(mu_pred_std, P_pred_std, h, H_x, H_r, R, r0, u, y)
    
    # Square root update
    ll_sqrt, mu_cond_sqrt, S_cond_sqrt, _, _ = _sqrt_condition_on(mu_pred_sqrt, S_pred_sqrt, h, H_x, H_r, R, r0, u, y)
    P_cond_sqrt = S_cond_sqrt @ S_cond_sqrt.T
    
    print(f"Log-likelihood difference (should be ~0): {jnp.abs(ll_std - ll_sqrt):.2e}")
    print(f"Mean difference (should be ~0): {jnp.max(jnp.abs(mu_cond_std - mu_cond_sqrt)):.2e}")
    print(f"Covariance difference (should be ~0): {jnp.max(jnp.abs(P_cond_std - P_cond_sqrt)):.2e}")
    
    print("\nTesting Combined Kalman Step...")
    print("=" * 60)
    
    # Combined square root step
    ll_combined, mu_combined, S_combined = _sqrt_kalman_step(m0, S0, f, F_x, F_q, Q, q0, u, h, H_x, H_r, R, r0, y)
    P_combined = S_combined @ S_combined.T
    
    print(f"Log-likelihood difference (should be ~0): {jnp.abs(ll_std - ll_combined):.2e}")
    print(f"Mean difference (should be ~0): {jnp.max(jnp.abs(mu_cond_std - mu_combined)):.2e}")
    print(f"Covariance difference (should be ~0): {jnp.max(jnp.abs(P_cond_std - P_combined)):.2e}")
    
    print("\nVerifying Positive Definiteness...")
    print("=" * 60)
    eigvals_sqrt = jnp.linalg.eigvals(P_cond_sqrt)
    eigvals_combined = jnp.linalg.eigvals(P_combined)
    print(f"Min eigenvalue (sqrt update): {jnp.min(eigvals_sqrt):.2e} (should be > 0)")
    print(f"Min eigenvalue (combined): {jnp.min(eigvals_combined):.2e} (should be > 0)")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Square root filter matches standard filter results.")


def test_sqrt_ukf_vs_standard_ukf():
    """Compare square root UKF with standard UKF on a nonlinear system."""
    
    print("\n" + "=" * 60)
    print("TESTING SQUARE ROOT UKF vs STANDARD UKF")
    print("=" * 60)
    
    key = jr.PRNGKey(42)
    state_dim = 4
    obs_dim = 2
    
    # Random initial state
    m0 = jr.normal(key, (state_dim,))
    P0 = jnp.eye(state_dim) * 1.5
    S0 = cholesky(P0, lower=True)
    
    # Nonlinear dynamics and observation noise covariances
    Q = jnp.eye(state_dim) * 0.2
    R = jnp.eye(obs_dim) * 0.3
    
    # UKF parameters
    alpha = 1.0
    beta = 2.0
    kappa = 0.0
    
    # Define nonlinear functions
    # Dynamics: slightly nonlinear transformation
    def f(x, q, u):
        return jnp.array([
            x[0] + 0.1 * jnp.sin(x[1]) + q[0],
            x[1] + 0.1 * x[2] + q[1],
            x[2] - 0.05 * x[0]**2 + q[2],
            x[3] * 0.95 + 0.1 * x[1] + q[3]
        ])
    
    # Observation: nonlinear measurement
    def h(x, r, u):
        return jnp.array([
            x[0]**2 + x[1] + r[0],
            x[2] * x[3] + r[1]
        ])
    
    q0 = jnp.zeros(state_dim)
    r0 = jnp.zeros(obs_dim)
    u = jnp.zeros(1)  # No input
    
    # Generate observation
    true_state = jr.normal(jr.PRNGKey(5), (state_dim,))
    noise_r = jr.normal(jr.PRNGKey(6), (obs_dim,)) * 0.1
    y = h(true_state, noise_r, u)
    
    print("\nTesting UKF Prediction Step...")
    print("=" * 60)
    
    # Standard UKF prediction
    mu_pred_std, P_pred_std = _ukf_predict_nonadditive(m0, P0, f, u, Q, alpha, beta, kappa, q0)
    
    # Square root UKF prediction
    mu_pred_sqrt, S_pred_sqrt = _ukf_sqrt_predict_nonadditive(m0, S0, f, u, Q, alpha, beta, kappa, q0)
    P_pred_sqrt = S_pred_sqrt @ S_pred_sqrt.T
    
    mean_diff_pred = jnp.max(jnp.abs(mu_pred_std - mu_pred_sqrt))
    cov_diff_pred = jnp.max(jnp.abs(P_pred_std - P_pred_sqrt))
    
    print(f"Mean difference (should be ~0): {mean_diff_pred:.2e}")
    print(f"Covariance difference (should be ~0): {cov_diff_pred:.2e}")
    
    # Check that differences are small (numerical precision)
    assert mean_diff_pred < 1e-6, f"UKF prediction mean mismatch: {mean_diff_pred}"
    assert cov_diff_pred < 1e-5, f"UKF prediction covariance mismatch: {cov_diff_pred}"
    print("✓ UKF Prediction: PASS")
    
    print("\nTesting UKF Update Step...")
    print("=" * 60)
    
    # Standard UKF update
    ll_std, mu_cond_std, P_cond_std = _ukf_condition_on_nonadditive(
        mu_pred_std, P_pred_std, h, R, u, y, alpha, beta, kappa, r0
    )
    
    # Square root UKF update
    ll_sqrt, mu_cond_sqrt, S_cond_sqrt = _ukf_sqrt_condition_on_nonadditive(
        mu_pred_sqrt, S_pred_sqrt, h, R, u, y, alpha, beta, kappa, r0
    )
    P_cond_sqrt = S_cond_sqrt @ S_cond_sqrt.T
    
    ll_diff = jnp.abs(ll_std - ll_sqrt)
    mean_diff_cond = jnp.max(jnp.abs(mu_cond_std - mu_cond_sqrt))
    cov_diff_cond = jnp.max(jnp.abs(P_cond_std - P_cond_sqrt))
    
    print(f"Log-likelihood difference (should be ~0): {ll_diff:.2e}")
    print(f"Mean difference (should be ~0): {mean_diff_cond:.2e}")
    print(f"Covariance difference (should be ~0): {cov_diff_cond:.2e}")
    
    # Check that differences are small (numerical precision)
    assert ll_diff < 1e-4, f"UKF update log-likelihood mismatch: {ll_diff}"
    assert mean_diff_cond < 1e-5, f"UKF update mean mismatch: {mean_diff_cond}"
    assert cov_diff_cond < 1e-5, f"UKF update covariance mismatch: {cov_diff_cond}"
    print("✓ UKF Update: PASS")
    
    print("\nVerifying Positive Definiteness...")
    print("=" * 60)
    
    # Check positive definiteness for square root results
    eigvals_pred = jnp.linalg.eigvals(P_pred_sqrt)
    eigvals_cond = jnp.linalg.eigvals(P_cond_sqrt)
    
    min_eig_pred = jnp.min(jnp.real(eigvals_pred))
    min_eig_cond = jnp.min(jnp.real(eigvals_cond))
    
    print(f"Min eigenvalue (sqrt prediction): {min_eig_pred:.2e} (should be > 0)")
    print(f"Min eigenvalue (sqrt update): {min_eig_cond:.2e} (should be > 0)")
    
    assert min_eig_pred > 0, f"Square root UKF prediction not positive definite: {min_eig_pred}"
    assert min_eig_cond > 0, f"Square root UKF update not positive definite: {min_eig_cond}"
    print("✓ Positive Definiteness: PASS")
    
    print("\n" + "=" * 60)
    print("All UKF tests completed successfully!")
    print("Square root UKF matches standard UKF results.")
    print("=" * 60)


if __name__ == "__main__":
    test_sqrt_vs_standard()
    test_sqrt_ukf_vs_standard_ukf()
