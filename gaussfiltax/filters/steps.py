import jax.numpy as jnp #type: ignore
import jax.random as jr #type: ignore
import jax.tree_util as jtu #type: ignore
from functools import partial #type: ignore
from jax import lax, vmap, jacfwd, jacrev, jit #type: ignore
from jax.scipy.linalg import qr, cholesky #type: ignore
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
from util.numerics import _resample, _resample2, psd_solve
from util.misc import MVN_log_prob
from util.numerics import get_sigma_points
from containers import GaussianComponent, GaussianSum, _branches_from_tree1, _branches_from_tree2
import containers
from jaxtyping import Array, Float, Int #type: ignore

from parameters import ParamSSM
import time

import matplotlib.pyplot as plt #type: ignore

def _predict(m, P, f, F_x, F_q, Q, q0, u):
    r"""Predict next mean and covariance using first-order additive EKF
        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)
    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.
    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    F_x = F_x(m, q0, u)
    F_q = F_q(m, q0, u)
    mu_pred = f(m, q0, u)
    Sigma_pred = F_x @ P @ F_x.T + F_q @ Q @ F_q.T

    return mu_pred, Sigma_pred, F_x

def _condition_on(m, P, h, H_x, H_r, R, r0, u, y):
    r"""Condition a Gaussian potential on a new observation.
       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**
    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.
     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    H_x = H_x(m, r0, u)
    H_r = H_r(m, r0, u)
    S = H_r @ R @ H_r.T + H_x @ P @ H_x.T
    K = psd_solve(S, H_x @ P).T
    posterior_cov = P - K @ S @ K.T
    posterior_mean = m + K @ (y - h(m, r0, u))
    ll = MVN_log_prob(h(m, r0, u), S, y)

    return ll, posterior_mean, posterior_cov, H_x, K

def _sqrt_predict(m, S, f, F_x, F_q, Q, q0, u):
    r"""Square root prediction step for EKF using QR decomposition.
    
    Instead of propagating covariance P, propagates square root S where P = S @ S.T
    This provides better numerical stability and guarantees positive definiteness.
    
    Uses QR decomposition: If A = [F_x @ S, F_q @ sqrt(Q)], then A.T = Q @ R
    and the predicted square root is R.T (upper triangular).
    
    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance square root (P = S @ S.T).
        f (Callable): dynamics function.
        F_x (Callable): Jacobian of dynamics function w.r.t. state.
        F_q (Callable): Jacobian of dynamics function w.r.t. process noise.
        Q (D_hid,D_hid): dynamics covariance matrix.
        q0 (D_hid,): nominal process noise.
        u (D_in,): inputs.
    
    Returns:
        mu_pred (D_hid,): predicted mean.
        S_pred (D_hid,D_hid): predicted covariance square root.
        F_x_eval (D_hid,D_hid): evaluated Jacobian matrix.
    """
    F_x_eval = F_x(m, q0, u)
    F_q_eval = F_q(m, q0, u)
    mu_pred = f(m, q0, u)
    
    # Compute square root of Q
    sqrt_Q = cholesky(Q, lower=True)
    
    # Form the augmented matrix [F_x @ S, F_q @ sqrt(Q)]
    # This will be (D_hid, 2*D_hid) in general
    A = jnp.hstack([F_x_eval @ S, F_q_eval @ sqrt_Q])
    
    # QR decomposition: A.T = Q @ R, so we want R.T as our square root
    _, R = qr(A.T, mode='economic')
    S_pred = R.T
    
    return mu_pred, S_pred, F_x_eval

def _sqrt_condition_on(m, S, h, H_x, H_r, R, r0, u, y):
    r"""Square root update step for EKF using QR decomposition.
    
    Instead of propagating covariance P, propagates square root S where P = S @ S.T
    This provides better numerical stability and guarantees positive definiteness.
    
    The update uses QR decomposition and back-substitution to compute the Kalman gain
    and updated square root covariance without explicitly forming P.
    
    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance square root (P = S @ S.T).
        h (Callable): emission function.
        H_x (Callable): Jacobian of emission function w.r.t. state.
        H_r (Callable): Jacobian of emission function w.r.t. measurement noise.
        R (D_obs,D_obs): emission covariance matrix.
        r0 (D_obs,): nominal measurement noise.
        u (D_in,): inputs.
        y (D_obs,): observation.
    
    Returns:
        ll (float): log-likelihood of observation.
        mu_cond (D_hid,): filtered mean.
        S_cond (D_hid,D_hid): filtered covariance square root.
        H_x_eval (D_obs,D_hid): evaluated Jacobian matrix.
        K (D_hid,D_obs): Kalman gain matrix.
    """
    H_x_eval = H_x(m, r0, u)
    H_r_eval = H_r(m, r0, u)
    
    # Compute square root of R
    sqrt_R = cholesky(R, lower=True)
    
    # Form the augmented matrix for innovation covariance square root
    # A = [H_x @ S, H_r @ sqrt(R)] is (D_obs, D_hid + D_obs)
    A = jnp.hstack([H_x_eval @ S, H_r_eval @ sqrt_R])
    
    # QR decomposition: A.T = Q @ R_qr
    Q_qr, R_qr = qr(A.T, mode='economic')
    S_innov = R_qr.T  # Innovation covariance square root (D_obs, D_obs)
    
    # Innovation covariance S = S_innov @ S_innov.T
    innovation = y - h(m, r0, u)
    S_innov_full = S_innov @ S_innov.T
    
    # Compute log-likelihood
    ll = MVN_log_prob(h(m, r0, u), S_innov_full, y)
    
    # Compute Kalman gain: K = P @ H_x.T @ S^{-1}
    # We have P = S @ S.T, so K = S @ S.T @ H_x.T @ (S_innov @ S_innov.T)^{-1}
    # More efficiently: K = S @ (S.T @ H_x.T) @ S_innov^{-1} @ S_innov.T^{-1}
    
    # Solve for Kalman gain using the QR factors
    # K = P @ H_x.T @ S_innov^{-T} @ S_innov^{-1}
    temp = psd_solve(S_innov_full, H_x_eval @ S @ S.T).T
    K = temp
    
    # Updated mean
    mu_cond = m + K @ innovation
    
    # Joseph form for square root update (more stable)
    # P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    # For square root: use QR on [(I - K @ H_x) @ S, K @ sqrt(R)]
    I_KH = jnp.eye(S.shape[0]) - K @ H_x_eval
    B = jnp.hstack([I_KH @ S, K @ sqrt_R])
    _, R_update = qr(B.T, mode='economic')
    S_cond = R_update.T
    
    return ll, mu_cond, S_cond, H_x_eval, K

def _sqrt_kalman_step(m, S, f, F_x, F_q, Q, q0, u, h, H_x, H_r, R, r0, y):
    r"""Combined square root Kalman prediction and update step.
    
    Performs both prediction and update while maintaining square root form.
    This is more numerically stable than the standard Kalman filter.
    
    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance square root.
        f (Callable): dynamics function.
        F_x (Callable): Jacobian of dynamics w.r.t. state.
        F_q (Callable): Jacobian of dynamics w.r.t. process noise.
        Q (D_hid,D_hid): process noise covariance.
        q0 (D_hid,): nominal process noise.
        u (D_in,): inputs.
        h (Callable): emission function.
        H_x (Callable): Jacobian of emission w.r.t. state.
        H_r (Callable): Jacobian of emission w.r.t. measurement noise.
        R (D_obs,D_obs): measurement noise covariance.
        r0 (D_obs,): nominal measurement noise.
        y (D_obs,): observation.
    
    Returns:
        ll (float): log-likelihood of observation.
        posterior_mean (D_hid,): filtered mean.
        posterior_sqrt_cov (D_hid,D_hid): filtered covariance square root.
    """
    # Prediction step
    mu_pred, S_pred, _ = _sqrt_predict(m, S, f, F_x, F_q, Q, q0, u)
    
    # Update step
    ll, posterior_mean, posterior_sqrt_cov, _, _ = _sqrt_condition_on(
        mu_pred, S_pred, h, H_x, H_r, R, r0, u, y
    )
    
    return ll, posterior_mean, posterior_sqrt_cov

def _kalman_step(m, P, f, F_x, F_q, Q, q0, u, h, H_x, H_r, R, r0, y):
    F_x = F_x(m, q0, u)
    F_q = F_q(m, q0, u)
    mu_pred = f(m, q0, u)
    Sigma_pred = F_x @ P @ F_x.T + F_q @ Q @ F_q.T

    H_x = H_x(mu_pred, r0, u)
    H_r = H_r(mu_pred, r0, u)
    S = H_r @ R @ H_r.T + H_x @ Sigma_pred @ H_x.T
    K = psd_solve(S, H_x @ Sigma_pred).T
    posterior_cov = Sigma_pred - K @ S @ K.T
    posterior_mean = mu_pred + K @ (y - h(mu_pred, r0, u))
    ll = MVN_log_prob(h(mu_pred, r0, u), S, y)
    return ll, posterior_mean, posterior_cov

def _ukf_predict_additive(m, P, f, u, Q, uparams, q0):
    r"""Predict next mean and covariance using first-order additive UKF`
        
    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.
    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    state_dim = m.shape[0]
    q0 = jnp.zeros((state_dim,))
    lamda = uparams.alpha**2 * (state_dim + uparams.kappa) - state_dim
    sigma_points = utils._get_sigma_points(m, P, lamda)
    new_sigma_points = vmap(f, in_axes=(0, None, None))(sigma_points, q0, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2*(lamda+state_dim)) + f(m, q0, u) * lamda / (lamda + state_dim)
    Sigma_pred = (new_sigma_points - mu_pred).T @ (new_sigma_points - mu_pred) / ( 2 * (lamda+state_dim)) + \
                (f(m, q0, u)-mu_pred) @ (f(m, q0, u) - mu_pred).T * (lamda / (lamda + state_dim) + 1-uparams.alpha**2+uparams.beta) + Q
    return mu_pred, Sigma_pred

def _ukf_predict_nonadditive(m, P, f, u, Q, alpha, beta, kappa, q0):
    r"""Predict next mean and covariance using first-order additive UKF`
        
    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.
    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    state_dim = m.shape[0]
    noise_dim = Q.shape[0]

    lamda = alpha**2 * (state_dim + noise_dim + kappa) - (state_dim + noise_dim)

    mA = jnp.concatenate((m, q0))
    PA = jnp.block([[P, jnp.zeros((state_dim, noise_dim))], [jnp.zeros((noise_dim, state_dim)), Q]])

    sigma_points = get_sigma_points(mA, PA, lamda)
    fA = lambda xA, u: f(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(fA, in_axes=(0, None))(sigma_points, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + f(m, q0, u) * lamda / (lamda + state_dim + noise_dim)
    Sigma_pred = jnp.einsum('ij,ik->jk',(new_sigma_points - mu_pred), (new_sigma_points - mu_pred)) / ( 2 * (lamda + state_dim + noise_dim)) + \
                        (lamda / (lamda + state_dim + noise_dim) + 1 - alpha**2 + beta) * jnp.einsum('i,j->ij', f(m, q0, u) - mu_pred, f(m, q0, u) - mu_pred)
    return mu_pred, Sigma_pred

def _ukf_condition_on_additive(m, P, h, R, u, y, uparams, r0):
    r"""Condition a Gaussian potential on a new observation using first-order additive UKF.
    """
    state_dim = m.shape[0]
    emission_dim = y.shape[0]
    r0 = jnp.zeros(emission_dim)
    lamda = uparams.alpha**2 * (state_dim + uparams.kappa) - state_dim
    sigma_points = utils._get_sigma_points(m, P, lamda)
    new_sigma_points = vmap(h, in_axes=(0, None, None))(sigma_points, r0, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2*(lamda+state_dim)) + h(m, r0, u) * lamda / (lamda + state_dim)
    S = jnp.einsum('ij,ik->jk',(new_sigma_points - mu_pred), (new_sigma_points - mu_pred)) / ( 2 * (lamda + state_dim )) + \
                        (lamda / (lamda + state_dim) + 1 - uparams.alpha**2 + uparams.beta) * jnp.einsum('i,j->ij', h(m, r0, u) - mu_pred, h(m, r0, u) - mu_pred) + R
    C = jnp.einsum('ij,ik->jk', new_sigma_points - mu_pred, sigma_points[:, :state_dim] - m)  / (2*(lamda+state_dim))
    
    K = psd_solve(S, C).T
    posterior_cov = P - K @ S @ K.T
    posterior_mean = m + K @ (y - mu_pred)
    ll = MVN_log_prob(mu_pred, S, y)

    return ll, posterior_mean, posterior_cov

def _ukf_condition_on_nonadditive(m, P, h, R, u, y, alpha, beta, kappa, r0=None):
    r"""Condition a Gaussian potential on a new observation using first-order additive UKF.
    """
    state_dim = m.shape[0]
    noise_dim = r0.shape[0]
    emission_dim = y.shape[0]

    lamda = alpha**2 * (state_dim + noise_dim + kappa) - (state_dim + noise_dim)

    mA = jnp.concatenate((m, r0))
    PA = jnp.block([[P, jnp.zeros((state_dim, noise_dim))], [jnp.zeros((noise_dim, state_dim)), R]])

    sigma_points = get_sigma_points(mA, PA, lamda)
    hA = lambda xA, u: h(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(hA, in_axes=(0, None))(sigma_points, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + h(m, r0, u) * lamda / (lamda + state_dim + noise_dim)
    S = jnp.einsum('ij,ik->jk',(new_sigma_points - mu_pred), (new_sigma_points - mu_pred)) / ( 2 * (lamda + state_dim + noise_dim)) + \
                        (lamda / (lamda + state_dim + noise_dim) + 1 - alpha**2 + beta) * jnp.einsum('i,j->ij', h(m, r0, u) - mu_pred, h(m, r0, u) - mu_pred)


    C = jnp.einsum('ij,ik->jk', new_sigma_points - mu_pred, sigma_points[:, :state_dim] - m)  / (2*(lamda+state_dim + noise_dim))
    K = psd_solve(S, C).T
    posterior_cov = P - K @ S @ K.T
    posterior_mean = m + K @ (y - mu_pred)
    ll = MVN_log_prob(mu_pred, S, y)
    return ll, posterior_mean, posterior_cov

def _ukf_sqrt_predict_nonadditive(m, S, f, u, Q, alpha, beta, kappa, q0):
    r"""Square root UKF prediction step for non-additive noise.
    
    Instead of propagating covariance P, propagates square root S where P = S @ S.T.
    This provides better numerical stability and guarantees positive definiteness.
    
    Uses QR decomposition to compute the predicted square root covariance.
    
    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance square root (P = S @ S.T).
        f (Callable): dynamics function.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        alpha (float): UKF spread parameter.
        beta (float): UKF parameter for incorporating prior knowledge.
        kappa (float): UKF secondary scaling parameter.
        q0 (D_hid,): nominal process noise.
    
    Returns:
        mu_pred (D_hid,): predicted mean.
        S_pred (D_hid,D_hid): predicted covariance square root.
    """
    state_dim = m.shape[0]
    noise_dim = Q.shape[0]
    
    lamda = alpha**2 * (state_dim + noise_dim + kappa) - (state_dim + noise_dim)
    
    # Augmented mean and square root covariance
    mA = jnp.concatenate((m, q0))
    sqrt_Q = cholesky(Q, lower=True)
    # Augmented square root: SA = [[S, 0], [0, sqrt_Q]]
    SA = jnp.block([[S, jnp.zeros((state_dim, noise_dim))], 
                     [jnp.zeros((noise_dim, state_dim)), sqrt_Q]])
    
    # Reconstruct augmented covariance for sigma points
    PA = SA @ SA.T
    sigma_points = get_sigma_points(mA, PA, lamda)
    
    # Transform sigma points through dynamics
    fA = lambda xA, u: f(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(fA, in_axes=(0, None))(sigma_points, u)
    
    # Compute predicted mean
    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + \
              f(m, q0, u) * lamda / (lamda + state_dim + noise_dim)
    
    # Compute weights
    wc = 1.0 / (2 * (lamda + state_dim + noise_dim))
    wc0 = lamda / (lamda + state_dim + noise_dim) + (1 - alpha**2 + beta)
    
    # Center the sigma points
    centered_sigma_points = new_sigma_points - mu_pred
    center_point = f(m, q0, u) - mu_pred
    
    # Form matrix for QR decomposition: sqrt(wc) * centered_points.T
    # Plus the center point scaled by sqrt(wc0)
    weighted_points = jnp.sqrt(wc) * centered_sigma_points.T
    weighted_center = jnp.sqrt(jnp.abs(wc0)) * center_point.reshape(-1, 1)
    
    # Stack all columns for QR
    if wc0 >= 0:
        A = jnp.hstack([weighted_points, weighted_center])
    else:
        # Cholupdate: downdate if wc0 is negative (shouldn't happen normally)
        A = weighted_points
    
    # QR decomposition to get square root
    _, R = qr(A.T, mode='economic')
    S_pred = R.T
    
    return mu_pred, S_pred

def _ukf_sqrt_condition_on_nonadditive(m, S, h, R, u, y, alpha, beta, kappa, r0):
    r"""Square root UKF update step for non-additive measurement noise.
    
    Instead of propagating covariance P, propagates square root S where P = S @ S.T.
    This provides better numerical stability and guarantees positive definiteness.
    
    Uses QR decomposition and sigma points to compute the updated square root covariance.
    
    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance square root (P = S @ S.T).
        h (Callable): emission function.
        R (D_obs,D_obs): emission covariance matrix.
        u (D_in,): inputs.
        y (D_obs,): observation.
        alpha (float): UKF spread parameter.
        beta (float): UKF parameter for incorporating prior knowledge.
        kappa (float): UKF secondary scaling parameter.
        r0 (D_obs,): nominal measurement noise.
    
    Returns:
        ll (float): log-likelihood of observation.
        mu_cond (D_hid,): filtered mean.
        S_cond (D_hid,D_hid): filtered covariance square root.
    """
    state_dim = m.shape[0]
    noise_dim = r0.shape[0]
    
    lamda = alpha**2 * (state_dim + noise_dim + kappa) - (state_dim + noise_dim)
    
    # Augmented mean and square root covariance
    mA = jnp.concatenate((m, r0))
    sqrt_R = cholesky(R, lower=True)
    # Augmented square root: SA = [[S, 0], [0, sqrt_R]]
    SA = jnp.block([[S, jnp.zeros((state_dim, noise_dim))], 
                     [jnp.zeros((noise_dim, state_dim)), sqrt_R]])
    
    # Reconstruct augmented covariance for sigma points
    PA = SA @ SA.T
    sigma_points = get_sigma_points(mA, PA, lamda)
    
    # Transform sigma points through emission function
    hA = lambda xA, u: h(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(hA, in_axes=(0, None))(sigma_points, u)
    
    # Compute predicted measurement mean
    mu_y = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + \
           h(m, r0, u) * lamda / (lamda + state_dim + noise_dim)
    
    # Compute weights
    wc = 1.0 / (2 * (lamda + state_dim + noise_dim))
    wc0 = lamda / (lamda + state_dim + noise_dim) + (1 - alpha**2 + beta)
    
    # Center the sigma points
    centered_y_points = new_sigma_points - mu_y
    centered_x_points = sigma_points[:, :state_dim] - m
    center_y = h(m, r0, u) - mu_y
    
    # Innovation covariance square root using QR
    weighted_y_points = jnp.sqrt(wc) * centered_y_points.T
    weighted_center_y = jnp.sqrt(jnp.abs(wc0)) * center_y.reshape(-1, 1)
    
    if wc0 >= 0:
        A_y = jnp.hstack([weighted_y_points, weighted_center_y])
    else:
        A_y = weighted_y_points
    
    _, R_y = qr(A_y.T, mode='economic')
    S_yy = R_y.T  # Innovation covariance square root
    
    # Cross-covariance between state and measurement
    # Match the standard UKF computation exactly
    P_xy = jnp.einsum('ij,ik->jk', centered_x_points, centered_y_points) / (2 * (lamda + state_dim + noise_dim))
    
    # Innovation covariance
    P_yy = S_yy @ S_yy.T
    
    # Kalman gain
    K = psd_solve(P_yy, P_xy.T).T
    
    # Innovation
    innovation = y - mu_y
    
    # Updated mean
    mu_cond = m + K @ innovation
    
    # Joseph form for posterior covariance (more stable)
    # P_post = (I - K @ H_eff) @ P @ (I - K @ H_eff).T + K @ R_eff @ K.T
    # For UKF, we use the empirical covariance structure from sigma points
    # This matches the standard UKF which computes: P - K @ P_yy @ K.T
    P_xx = S @ S.T
    P_post = P_xx - K @ P_yy @ K.T
    
    # Ensure symmetry before Cholesky
    P_post = 0.5 * (P_post + P_post.T)
    
    # Take Cholesky decomposition for square root
    S_cond = cholesky(P_post, lower=True)
    
    # Log-likelihood
    ll = MVN_log_prob(mu_y, P_yy, y)
    
    return ll, mu_cond, S_cond

def _autocov1(m, P, jacobian, hessian_tensor, num_particles, bias, u, alpha, eta=0.1, tol=0.1):
    r"""Automatically compute the covariance of the Gaussian particles my minimizing solving a semidefinite program.
    The mean, covariance and hessian are used in the construction of the SDP. Also, potentially the number of particles
    can be automatically determined, but can also be given as an argument.
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        hessian_tensor (Callable): Hessian of the emission function.
        args (tuple): list of arguments for the optimizer.
    Returns:
        cov (D_hid,D_hid): covariance of the Gaussian particles.
        num_particles (int): number of particles.
    """
    # Hessian has shape (emission_dim, state_dim, state_dim), i.e., for each of the emission_dim, we have a
    # (state_dim x state_dim) matrix.
    state_dim = P.shape[0]
    hessian = hessian_tensor(m, bias, u)
    J = jacobian(m, bias, u)

    #1a
    # Delta = utils.sdp_opt(state_dim, num_particles, P, J, hessian, alpha, tol)

    #1b
    # Delta = utils.sdp_opt2(state_dim, num_particles, P, J, hessian, alpha, eta, tol)

    #2
    # Delta = alpha * jnp.eye(state_dim)

    #3
    rho = alpha
    Delta = rho * P

    #4
    # rho = jnp.minimum(.95, alpha * jnp.trace(P @ J.T @ J) / jnp.sum(jnp.trace(hessian @ P, axis1=1, axis2=2)**2))
    # Delta = rho * P

    return Delta, rho

def _autocov2(m, P, jacobian, hessian_tensor, num_particles, bias, u, alpha, eta=0.1, tol=1.0): 
    r"""Automatically compute the covariance of the Gaussian particles my minimizing solving a semidefinite program.
    The mean, covariance and hessian are used in the construction of the SDP. Also, potentially the number of particles
    can be automatically determined, but can also be given as an argument.
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        hessian_tensor (Callable): Hessian of the emission function.
        args (tuple): list of arguments for the optimizer.
    Returns:
        cov (D_hid,D_hid): covariance of the Gaussian particles.
        num_particles (int): number of particles.
    """
    # Hessian has shape (emission_dim, state_dim, state_dim), i.e., for each of the emission_dim, we have a
    # (state_dim x state_dim) matrix.
    state_dim = P.shape[0]
    hessian = hessian_tensor(m, bias, u)
    J = jacobian(m,bias,u)

    
    #1a
    # Lambda = utils.sdp_opt(state_dim, num_particles, P, J, hessian, alpha, tol)
    # Lambda = 0.5 * Lambda/jnp.trace(Lambda) - utils.project_to_psd(0.5 * Lambda/jnp.trace(Lambda) - Lambda)


    #1b 
    # Lambda = utils.sdp_opt2(state_dim, num_particles, P, J, hessian, alpha, tol)
    # rho = 1.0


    #2
    # Lambda = alpha * jnp.eye(state_dim)2

    #3
    # rho = alpha
    # Lambda = alpha * P

    #4
    rho = jnp.minimum(.95, alpha * jnp.trace(P @ J.T @ J) / jnp.sum(jnp.trace((hessian) @ P, axis1=1, axis2=2)**2))
    Lambda = rho * P

    return Lambda, rho
