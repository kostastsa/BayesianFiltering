import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax
from functools import partial
from jax import lax, vmap, pmap, jacfwd, jacrev, debug, device_put, jit
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
import gaussfiltax.utils as utils
from gaussfiltax.utils import _resample
from gaussfiltax.containers import GaussianComponent, GaussianSum, _branches_from_tree1, _branches_from_tree2
from gaussfiltax import containers
from jaxtyping import Array, Float, Int


from dynamax.utils.utils import psd_solve
from gaussfiltax.models import ParamsNLSSM, ParamsBPF
import time

import matplotlib.pyplot as plt

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x
_MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))
def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


class PosteriorGaussianSumFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.
    :param weights: weights of the Gaussian components
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    """
    weights: Optional[Float[Array, "num_components ntime"]] = None
    means: Optional[Float[Array, "num_components ntime state_dim"]] = None
    covariances: Optional[Float[Array, "num_components ntime state_dim state_dim"]] = None
    predicted_means: Optional[Float[Array, "num_components ntime state_dim"]] = None
    predicted_covariances: Optional[Float[Array, "num_components ntime state_dim state_dim"]] = None

class ParamsUKF(NamedTuple):
    r"""Parameters for the Unscented Kalman Filter.
    :param alpha: Spread of the sigma points. Typically 1e-3.
    :param beta: Used to incorporate prior knowledge of the distribution. 2 is optimal for Gaussian distributions.
    :param kappa: Secondary scaling parameter usually set to 0.
    """
    alpha: Float = 1e-3
    beta: Float = 2
    kappa: Float = 0

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
    ll = _MVN_log_prob(h(m, r0, u), S, y)
    return ll, posterior_mean, posterior_cov, H_x, K

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
    ll = _MVN_log_prob(h(mu_pred, r0, u), S, y)
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

def _ukf_predict_nonadditive(m, P, f, u, Q, uparams, q0):
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

    lamda = uparams.alpha**2 * (state_dim + noise_dim + uparams.kappa) - (state_dim + noise_dim)

    mA = jnp.concatenate((m, q0))
    PA = jnp.block([[P, jnp.zeros((state_dim, noise_dim))], [jnp.zeros((noise_dim, state_dim)), Q]])

    sigma_points = utils._get_sigma_points(mA, PA, lamda)
    fA = lambda xA, u: f(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(fA, in_axes=(0, None))(sigma_points, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + f(m, q0, u) * lamda / (lamda + state_dim + noise_dim)
    Sigma_pred = jnp.einsum('ij,ik->jk',(new_sigma_points - mu_pred), (new_sigma_points - mu_pred)) / ( 2 * (lamda + state_dim + noise_dim)) + \
                        (lamda / (lamda + state_dim + noise_dim) + 1 - uparams.alpha**2 + uparams.beta) * jnp.einsum('i,j->ij', f(m, q0, u) - mu_pred, f(m, q0, u) - mu_pred)
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
    ll = _MVN_log_prob(mu_pred, S, y)

    return ll, posterior_mean, posterior_cov

def _ukf_condition_on_nonadditive(m, P, h, R, u, y, uparams, r0=None):
    r"""Condition a Gaussian potential on a new observation using first-order additive UKF.
    """
    state_dim = m.shape[0]
    noise_dim = r0.shape[0]
    emission_dim = y.shape[0]

    lamda = uparams.alpha**2 * (state_dim + noise_dim + uparams.kappa) - (state_dim + noise_dim)

    mA = jnp.concatenate((m, r0))
    PA = jnp.block([[P, jnp.zeros((state_dim, noise_dim))], [jnp.zeros((noise_dim, state_dim)), R]])

    sigma_points = utils._get_sigma_points(mA, PA, lamda)
    hA = lambda xA, u: h(xA[:state_dim], xA[state_dim:], u)
    new_sigma_points = vmap(hA, in_axes=(0, None))(sigma_points, u)

    mu_pred = jnp.sum(new_sigma_points, axis=0) / (2 * (lamda + state_dim + noise_dim)) + h(m, r0, u) * lamda / (lamda + state_dim + noise_dim)
    S = jnp.einsum('ij,ik->jk',(new_sigma_points - mu_pred), (new_sigma_points - mu_pred)) / ( 2 * (lamda + state_dim + noise_dim)) + \
                        (lamda / (lamda + state_dim + noise_dim) + 1 - uparams.alpha**2 + uparams.beta) * jnp.einsum('i,j->ij', h(m, r0, u) - mu_pred, h(m, r0, u) - mu_pred)


    C = jnp.einsum('ij,ik->jk', new_sigma_points - mu_pred, sigma_points[:, :state_dim] - m)  / (2*(lamda+state_dim + noise_dim))
    K = psd_solve(S, C).T
    posterior_cov = P - K @ S @ K.T
    posterior_mean = m + K @ (y - mu_pred)
    ll = _MVN_log_prob(mu_pred, S, y)
    return ll, posterior_mean, posterior_cov

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
    Delta = alpha * P

    #4 
    # Delta = jnp.minimum(1, alpha * jnp.trace(P) / jnp.sum(jnp.trace(_hessian @ P, axis1=1, axis2=2))) * P

    return Delta, num_particles

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
    # Lambda = utils.sdp_opt2(state_dim, num_particles, P, J, hessian, alpha, eta, tol)


    #2
    # Lambda = alpha * jnp.eye(state_dim)

    #3
    Lambda = alpha * P

    #4
    # Lambda = jnp.minimum(1.0, alpha * jnp.trace(P) / jnp.sum(jnp.trace(_hessian @ P, axis1=1, axis2=2)**2)) * P
    # jax.debug.print("🤯 {x} 🤯", x = alpha * jnp.trace(P) / jnp.sum(jnp.trace(_hessian @ P, axis1=1, axis2=2)**2))

    return Lambda, num_particles

def gaussian_sum_filter(
    params: ParamsNLSSM,
    emissions: Float[Array, "ntime emission_dim"],
    num_components: int = 1,
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGaussianSumFiltered:
    r"""Run an Gaussian sum filter, which is a mixture of (iterated) extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F_x, H_x = jacfwd(f, argnums=0), jacfwd(h, argnums=0)
    F_q, H_r = jacfwd(f, argnums=1), jacfwd(h, argnums=1)
#    f, h, F_x, H_x, F_q, H_r = (_process_fn(fn, inputs) for fn in (f, h, F_x, H_x, F_q, H_r))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        weights, pred_means, pred_covs = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on this emission
        lls, filtered_means, filtered_covs, _, _ = vmap(_condition_on, in_axes=(0, 0, None, None, None, None, None, None, None))(pred_means, pred_covs, h, H_x, H_r, R, r0, u, y)
        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights, weights)
        weights /= jnp.sum(weights)

        # Predict the next state
        pred_means, pred_covs, _ = vmap(_predict, in_axes=(0, 0, None, None, None, None, None, None))(filtered_means, filtered_covs, f, F_x, F_q, Q, q0, u)

        # Build carry and output states
        carry = (weights, pred_means, pred_covs)
        outputs = {
            "means": filtered_means,
            "covariances": filtered_covs,
            "predicted_means": pred_means,
            "predicted_covariances": pred_covs,
            "weights": weights
        }

        return carry, outputs

    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components, jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components)])
    carry = (jnp.ones(num_components)/num_components, initial_means, initial_covs)

    _, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       **outputs,
    )

    return posterior_filtered

def unscented_gaussian_sum_filter(
    params: ParamsNLSSM,
    uparams: ParamsUKF,
    emissions: Float[Array, "ntime emission_dim"],
    num_components: int = 1,
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGaussianSumFiltered:
    r"""Run an Gaussian sum filter, which is a mixture of (iterated) extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        weights, pred_means, pred_covs = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]


        # mq = jnp.concatenate([m, q0])
        # PQ = jnp.block([[P, jnp.zeros(state_dim, noise_dim)], [jnp.zeros(noise_dim, state_dim), Q]])        

        # Condition on this emission
        lls, filtered_means, filtered_covs = vmap(_ukf_condition_on_nonadditive, in_axes=(0, 0, None, None, None, None, None, None))(pred_means, pred_covs, h, R, u, y, uparams, r0)
       
       # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights, weights)
        weights /= jnp.sum(weights)

        # Predict the next state
        pred_means, pred_covs = vmap(_ukf_predict_nonadditive, in_axes=(0, 0, None, None, None, None, None))(filtered_means, filtered_covs, f, u, Q, uparams, q0)

        # Build carry and output states
        carry = (weights, pred_means, pred_covs)
        outputs = {
            "means": filtered_means,
            "covariances": filtered_covs,
            "predicted_means": pred_means,
            "predicted_covariances": pred_covs,
            "weights": weights
        }

        return carry, outputs

    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components, jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components)])
    carry = (jnp.ones(num_components)/num_components, initial_means, initial_covs)

    _, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       **outputs,
    )

    return posterior_filtered

def augmented_gaussian_sum_filter(
    params: ParamsNLSSM,
    emissions: Float[Array, "ntime emission dim"],
    num_components: Int[Array, "1 3"],
    rng_key: jr.PRNGKey = jr.PRNGKey(0),
    num_iter: int = 1,
    opt_args : Optional[Float[tuple, "1 2"]] = (0.1, 0.1),
    inputs: Optional[Float[Array, "ntime input dim"]] = None
) -> PosteriorGaussianSumFiltered:
    r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F_x, H_x = jacfwd(f, argnums=0), jacfwd(h, argnums=0)
    F_q, H_r = jacfwd(f, argnums=1), jacfwd(h, argnums=1)
    F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
#    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        filtered_components = carry
        filtered_sum = containers._components_to_gaussian_sum(filtered_components)
        filtered_means = jnp.array(filtered_sum.means)
        filtered_covs = jnp.array(filtered_sum.covariances)
        filtered_weights = jnp.array(filtered_sum.weights)
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]

         # Autocov 1
        tin = time.time()
        nums_to_split = jnp.array([num_components[1]]*num_components[0])
        # Deltas, nums_to_split = vmap(_autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, nums_to_split, q0, u, 1.0)
        Deltas = jnp.array([opt_args[0] * filtered_covs[i] for i in range(num_components[0])])

        # state_dim = filtered_covs[0].shape[0]
        # hessian = F_xx(filtered_means[0], q0, u)
        # J = F_x(filtered_means[0], q0, u)
        # Delta = utils.sdp_opt(state_dim, num_components[0], filtered_covs[0], J, hessian, 1.0)
        # Deltas = jnp.array([Delta for i in range(num_components[0])])

        t_autocov1 = time.time() - tin

        # Branch 1
        key, subkey = jr.split(rng_key)
        tin = time.time()
        _components_to_predict = _branches_from_tree1(filtered_components, list(Deltas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_predict = containers._components_to_gaussian_sum(leaves)
        t_branch1 = time.time() - tin

        # Predict
        tin = time.time()
        predicted_means, predicted_covs, grads_dyn = vmap(_predict, in_axes=(0,0,None,None,None,None,None,None))(jnp.array(_sum_to_predict.means), jnp.array(_sum_to_predict.covariances), f, F_x, F_q, Q, q0, u)
        t_predict = time.time() - tin

        # Recast 1
        tin = time.time()
        predicted_sum = GaussianSum(list(predicted_means), list(predicted_covs), _sum_to_predict.weights)
        predicted_components = containers._gaussian_sum_to_components(predicted_sum)
        t_recast1= time.time() - tin

        # Autocov before update
        tin = time.time()
        nums_to_split = jnp.array([num_components[2]] * num_components[0]* num_components[1])
        # Lambdas, nums_to_split = vmap(gf._autocov2, in_axes=(0, 0, None, None, 0, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, nums_to_split, r0, u, 1.0)
        Lambdas = jnp.array([opt_args[1] * predicted_covs[i] for i in range(num_components[0] * num_components[1])])        
        t_autocov2 = time.time() - tin

        # Branching before update
        key, subkey = jr.split(key)
        tin = time.time()
        _components_to_update = _branches_from_tree2(predicted_components, list(Lambdas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_update, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_update = containers._components_to_gaussian_sum(leaves)
        t_branch2 = time.time() - tin

        # Update
        tin = time.time()
        lls, updated_means, updated_covs, grads_obs, gain = vmap(_condition_on, in_axes=(0,0,None,None,None,None,None,None,None))(jnp.array(_sum_to_update.means), jnp.array(_sum_to_update.covariances), h, H_x, H_r, R, r0, u, y)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, jnp.array(_sum_to_update.weights))
        weights /= jnp.sum(weights)
        pre_weights = weights
        t_update = time.time() - tin

        # Resampling 
        tin = time.time()
        resampled_idx = jr.choice(jr.PRNGKey(0), jnp.arange(weights.shape[0]), shape=(num_components[0], ), p=weights)
        filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
        filtered_covs = jnp.take(updated_covs, resampled_idx, axis=0)
        # filtered_covs = 10*filtered_covs # Covariance re-inflation
        weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
        t_re =  time.time() - tin

        # Deterministic Reduction
        # tin = time.time()
        # idx = jnp.argpartition(weights, -num_components[0])[-num_components[0]:]
        # filtered_means = jnp.take(updated_means, idx, axis=0)
        # filtered_covs = jnp.take(updated_covs, idx, axis=0)
        # weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
        # t_re =  time.time() - tin

        # jax.debug.print("🤯 {x} 🤯", x=t)
        # jax.debug.print("🤯 {x} 🤯", x=updated_means)

        # Build carry and output states
        carry = containers._gaussian_sum_to_components(GaussianSum(list(filtered_means), list(filtered_covs), weights))
        outputs = {
            "weights": weights,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        aux_outputs = {
            "Deltas": Deltas,
            "Lambdas": Lambdas,
            "grads_dyn": grads_dyn,
            "grads_obs": grads_obs,
            "gain": gain,
            "timing": jnp.array([t_autocov1, t_branch1, t_predict, t_recast1, t_autocov2, t_branch2, t_update, t_re]),
            "updated_means": updated_means,
            "pre_weights": pre_weights
        }

        return carry, (outputs, aux_outputs)
    
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    init_components = containers._gaussian_sum_to_components(GaussianSum(initial_means, initial_covs, initial_weights))
    carry = init_components

    carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       outputs["weights"], 
       outputs["means"], 
       outputs["covariances"]
    )
    
    return posterior_filtered, aux_outputs

def speedy_augmented_gaussian_sum_filter(
    params: ParamsNLSSM,
    emissions: Float[Array, "ntime emission dim"],
    num_components: Int[Array, "1 3"],
    rng_key: jr.PRNGKey = jr.PRNGKey(0),
    num_iter: int = 1,
    opt_args : Optional[Float[tuple, "1 2"]] = (0.1, 0.1),
    inputs: Optional[Float[Array, "ntime input dim"]] = None
) -> PosteriorGaussianSumFiltered:
    r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F_x, H_x = jacfwd(f, argnums=0), jacfwd(h, argnums=0)
    F_q, H_r = jacfwd(f, argnums=1), jacfwd(h, argnums=1)
    F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
#    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    # @jit
    def _step(carry, t):
        filtered_means, filtered_covs, weights = carry
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]
        state_dim = filtered_covs[0].shape[0]

         # Autocov 1
        tin = time.time()
        nums_to_split = jnp.array([num_components[1]]*num_components[0])
        # Deltas, nums_to_split = vmap(_autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, nums_to_split, q0, u, 1.0)
        Deltas = jnp.array([opt_args[0] * filtered_covs[i] for i in range(num_components[0])])


        # hessian = F_xx(filtered_means[0], q0, u)
        # J = F_x(filtered_means[0], q0, u)
        # Delta = utils.sdp_opt(state_dim, num_components[0], filtered_covs[0], J, hessian, 1.0)
        # Deltas = jnp.array([Delta for i in range(num_components[0])])

        t_autocov1 = time.time() - tin


        # Compute the z-sample

        key, subkey = jr.split(rng_key)
        tin = time.time()
        iid_sample = jr.normal(key, shape=(num_components[0], state_dim, num_components[1]))
        # print('shape of iid sample', iid_sample.shape)
        tout = time.time() - tin

        mlt_matrices = vmap(jnp.linalg.cholesky)(filtered_covs-Deltas)
        # print('shape of mlt matrices', mlt_matrices.shape)
        centered_sample = vmap(jnp.matmul, in_axes=(0,0))(mlt_matrices, iid_sample)
        # print('shape of centered sample', centered_sample.shape)
        # print('shape of filtered_means', filtered_means.shape)
        z_sample = jnp.expand_dims(filtered_means, axis=2) + centered_sample
        z_sample = jnp.swapaxes(z_sample, 1, 2)
        # means_to_predict = jnp.reshape(z_sample, (num_components[0]*num_components[1], state_dim))
        # print('shape of z sample', z_sample.shape)
        # print('deltas', Deltas.shape)
        # print('shape of means to predict', means_to_predict.shape)

        # # Predict
        tin = time.time()
        map_predict = lambda means, cov: vmap(_predict, in_axes=(0,None,None,None,None,None,None,None))(means, cov, f, F_x, F_q, Q, q0, u)
        predicted_means, predicted_covs, grads_dyn = vmap(map_predict, in_axes=(0,0))(z_sample, Deltas)
        predicted_weights = jnp.tile(weights, (num_components[1],1)).T / num_components[1]

        t_predict = time.time() - tin
        predicted_means = jnp.reshape(predicted_means, (num_components[0]*num_components[1], state_dim))
        predicted_covs = jnp.reshape(predicted_covs, (num_components[0]*num_components[1], state_dim, state_dim))
        predicted_weights = jnp.reshape(predicted_weights, (num_components[0]*num_components[1],))
        # print('shape of predicted means', predicted_means.shape)
        # print('shape of predicted covs', predicted_covs.shape)
        # print('shape of predicted weights', predicted_weights.shape)

        # # Autocov before update
        tin = time.time()
        nums_to_split = jnp.array([num_components[2]] * num_components[0]* num_components[1])
        # Lambdas, nums_to_split = vmap(gf._autocov2, in_axes=(0, 0, None, None, 0, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, nums_to_split, r0, u, 1.0)
        Lambdas = jnp.array([opt_args[1] * predicted_covs[i] for i in range(num_components[0] * num_components[1])])        
        t_autocov2 = time.time() - tin


        # # Compute the s-sample
        key,_ = jr.split(key)
        tin = time.time()
        s_iid_sample = jr.normal(key, shape=(num_components[0]*num_components[1], state_dim, num_components[2]))
        # print('shape of iid sample', iid_sample.shape)
        tout = time.time() - tin

        s_mlt_matrices = vmap(jnp.linalg.cholesky)(predicted_covs-Lambdas)
        # print('shape of mlt matrices', s_mlt_matrices.shape)
        s_centered_sample = vmap(jnp.matmul, in_axes=(0,0))(s_mlt_matrices, s_iid_sample)
        # print('shape of centered sample', centered_sample.shape)
        # print('shape of predicted_means', predicted_means.shape)
        s_sample = jnp.expand_dims(predicted_means, axis=2) + s_centered_sample
        s_sample = jnp.swapaxes(s_sample, 1, 2)
        # means_to_predict = jnp.reshape(z_sample, (num_components[0]*num_components[1], state_dim))
        # print('shape of z sample', s_sample.shape)
        # print('Lambdas', Lambdas.shape)

        # Update
        tin = time.time()
        map_condition_on = lambda means, cov: vmap(_condition_on, in_axes=(0,None,None,None,None,None,None,None,None))(means, cov, h, H_x, H_r, R, r0, u, y)
        lls, updated_means, updated_covs, grads_obs, gain = vmap(map_condition_on, in_axes=(0,0))(s_sample, Lambdas)
        updated_weights = jnp.tile(predicted_weights, (num_components[2],1)).T / num_components[2]

        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, updated_weights)
        weights /= jnp.sum(weights)
        pre_weights = weights
        t_update = time.time() - tin

        updated_means = jnp.reshape(updated_means, (num_components[0]*num_components[1]*num_components[2], state_dim))
        updated_covs = jnp.reshape(updated_covs, (num_components[0]*num_components[1]*num_components[2], state_dim, state_dim))
        weights = jnp.reshape(weights, (num_components[0]*num_components[1]*num_components[2],))


        # print('shape of updated means', updated_means.shape)
        # print('shape of updated covs', updated_covs.shape)
        # print('shape of updated weights', weights.shape)


        # Resampling 
        tin = time.time()
        resampled_idx = jr.choice(jr.PRNGKey(0), jnp.arange(weights.shape[0]), shape=(num_components[0], ), p=weights)
        filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
        filtered_covs = jnp.take(updated_covs, resampled_idx, axis=0)
        # filtered_covs = 10*filtered_covs # Covariance re-inflation
        weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
        t_re =  time.time() - tin

        # print('shape of filtered means', filtered_means.shape)
        # print('shape of filtered covs', filtered_covs.shape)
        # print('shape of weights', weights.shape)

        # Build carry and output states
        carry = [filtered_means, filtered_covs, weights]
        outputs = {
            "weights": weights,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        aux_outputs = {
            "Deltas": Deltas,
            "Lambdas": Lambdas,
            "grads_dyn": grads_dyn,
            "grads_obs": grads_obs,
            "gain": gain,
            # "timing": jnp.array([t_autocov1, t_branch1, t_predict, t_recast1, t_autocov2, t_branch2, t_update, t_re]),
            "updated_means": updated_means,
            "pre_weights": pre_weights
        }

        return carry, (outputs, aux_outputs)

    
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    carry = [initial_means, initial_covs, initial_weights]

    carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       outputs["weights"], 
       outputs["means"], 
       outputs["covariances"]
    )
    
    return posterior_filtered, aux_outputs

def unscented_agsf(
    params: ParamsNLSSM,
    uparams: ParamsUKF,
    emissions: Float[Array, "ntime emission_dim"],
    num_components: Int[Array, "1 3"],
    rng_key: jr.PRNGKey = jr.PRNGKey(0),
    num_iter: int = 1,
    opt_args : Optional[Float[tuple, "1 2"]] = (0.1, 0.1),
    inputs: Optional[Float[Array, "ntime input_dim"]] = None
) -> PosteriorGaussianSumFiltered:

    r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F_x, H_x = jacfwd(f, argnums=0), jacfwd(h, argnums=0)
    F_q, H_r = jacfwd(f, argnums=1), jacfwd(h, argnums=1)
    F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        filtered_components = carry
        filtered_sum = containers._components_to_gaussian_sum(filtered_components)
        filtered_means = jnp.array(filtered_sum.means)
        filtered_covs = jnp.array(filtered_sum.covariances)
        filtered_weights = jnp.array(filtered_sum.weights)
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Autocov 1
        tin = time.time()
        nums_to_split = jnp.array([num_components[1]]*num_components[0])
        Deltas, nums_to_split = vmap(_autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, nums_to_split, q0, u, opt_args[0])
        t_autocov1 = time.time() - tin

        # Branch 1
        key, subkey = jr.split(rng_key)
        tin = time.time()
        _components_to_predict = _branches_from_tree1(filtered_components, list(Deltas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_predict = containers._components_to_gaussian_sum(leaves)
        t_branch1 = time.time() - tin

        # Predict
        tin = time.time()
        predicted_means, predicted_covs = vmap(_ukf_predict_nonadditive, in_axes=(0,0,None,None,None,None,None))(jnp.array(_sum_to_predict.means), jnp.array(_sum_to_predict.covariances), f, u, Q, uparams,q0)
        t_predict = time.time() - tin

        # Recast 1
        tin = time.time()
        predicted_sum = GaussianSum(list(predicted_means), list(predicted_covs), _sum_to_predict.weights)
        predicted_components = containers._gaussian_sum_to_components(predicted_sum)
        t_recast1= time.time() - tin

        # Autocov before update
        tin = time.time()
        nums_to_split = jnp.array([num_components[2]] * num_components[0]*num_components[1])
        Lambdas, nums_to_split = vmap(_autocov2, in_axes=(0, 0, None, None, 0, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, nums_to_split, r0, u, opt_args[1])
        t_autocov2 = time.time() - tin

        # Branching before update
        key, subkey = jr.split(key)
        tin = time.time()
        _components_to_update = _branches_from_tree2(predicted_components, list(Lambdas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_update, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_update = containers._components_to_gaussian_sum(leaves)
        t_branch2 = time.time() - tin

        # Update
        tin = time.time()
        lls, updated_means, updated_covs = vmap(_ukf_condition_on_nonadditive, in_axes=(0,0,None,None,None,None,None,None))(jnp.array(_sum_to_update.means), jnp.array(_sum_to_update.covariances), h, R, u, y, uparams,r0)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, jnp.array(_sum_to_update.weights))
        weights /= jnp.sum(weights)
        pre_weights = weights
        t_update = time.time() - tin

        # Resampling 
        tin = time.time()
        resampled_idx = jr.choice(jr.PRNGKey(0), jnp.arange(weights.shape[0]), shape=(num_components[0], ), p=weights)
        filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
        filtered_covs = jnp.take(updated_covs, resampled_idx, axis=0)
        # filtered_covs = 10*filtered_covs
        weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
        t_re =  time.time() - tin

        # Deterministic Reduction
        # tin = time.time()
        # idx = jnp.argpartition(weights, -num_components[0])[-num_components[0]:]
        # filtered_means = jnp.take(updated_means, idx, axis=0)
        # filtered_covs = jnp.take(updated_covs, idx, axis=0)
        # weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
        # t_re =  time.time() - tin

        # jax.debug.print("🤯 {x} 🤯", x=t)
        # jax.debug.print("🤯 {x} 🤯", x=updated_means)

        # Build carry and output states
        carry = containers._gaussian_sum_to_components(GaussianSum(list(filtered_means), list(filtered_covs), weights))
        outputs = {
            "weights": weights,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        aux_outputs = {
            "Deltas": Deltas,
            "Lambdas": Lambdas,
            "timing": jnp.array([t_autocov1, t_branch1, t_predict, t_recast1, t_autocov2, t_branch2, t_update, t_re]),
            "updated_means": updated_means,
            "pre_weights": pre_weights
        }

        return carry, (outputs, aux_outputs)
    
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    init_components = containers._gaussian_sum_to_components(GaussianSum(initial_means, initial_covs, initial_weights))
    carry = init_components

    carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       outputs["weights"], 
       outputs["means"], 
       outputs["covariances"]
    )
    
    return posterior_filtered, aux_outputs

def augmented_gaussian_sum_filter_optimal(
    params: ParamsNLSSM,
    emissions: Float[Array, "ntime emission dim"],
    num_components: Int[Array, "1 3"],
    rng_key: jr.PRNGKey = jr.PRNGKey(0),
    num_iter: int = 1,
    opt_args : Optional[Float[tuple, "1 2"]] = (0.1, 0.1),
    inputs: Optional[Float[Array, "ntime input dim"]] = None
) -> PosteriorGaussianSumFiltered:
    r"""Augmented Gaussian sum filter, which is a mixture of extended Kalman filters to produce the
    marginal likelihood and filtered state estimates.
    Args:
        params: model parameters.
        emissions: observation sequence.
        num_components: number of components in Gaussian sum approximation.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "weights".
    Returns:
        post: posterior object.
    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    F_x, H_x = jacfwd(f, argnums=0), jacfwd(h, argnums=0)
    F_q, H_r = jacfwd(f, argnums=1), jacfwd(h, argnums=1)
    F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
#    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        filtered_components = carry
        filtered_sum = containers._components_to_gaussian_sum(filtered_components)
        filtered_means = jnp.array(filtered_sum.means)
        filtered_covs = jnp.array(filtered_sum.covariances)
        filtered_weights = jnp.array(filtered_sum.weights)
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Autocov 1
        tin = time.time()
        nums_to_split = jnp.array([num_components[1]]*num_components[0])
        Deltas, nums_to_split = vmap(_autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, nums_to_split, q0, u, opt_args[0])
        t_autocov1 = time.time() - tin

        # Branch 1
        key, subkey = jr.split(rng_key)
        tin = time.time()
        _components_to_predict = _branches_from_tree1(filtered_components, list(Deltas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_predict = containers._components_to_gaussian_sum(leaves)
        t_branch1 = time.time() - tin

        # Predict
        tin = time.time()
        predicted_means, predicted_covs, grads_dyn = vmap(_predict, in_axes=(0,0,None,None,None,None,None,None))(jnp.array(_sum_to_predict.means), jnp.array(_sum_to_predict.covariances), f, F_x, F_q, Q, q0, u)
        t_predict = time.time() - tin

        # Recast 1
        tin = time.time()
        predicted_sum = GaussianSum(list(predicted_means), list(predicted_covs), _sum_to_predict.weights)
        predicted_components = containers._gaussian_sum_to_components(predicted_sum)
        t_recast1= time.time() - tin

        # Autocov before update
        tin = time.time()
        nums_to_split = jnp.array([num_components[2]] * num_components[0]*num_components[1])
        Lambdas, nums_to_split = vmap(_autocov2, in_axes=(0, 0, None, None, 0, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, nums_to_split, r0, u, opt_args[1])
        t_autocov2 = time.time() - tin

        # Branching before update
        key, subkey = jr.split(key)
        tin = time.time()
        _components_to_update = _branches_from_tree2(predicted_components, list(Lambdas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_update, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_update = containers._components_to_gaussian_sum(leaves)
        t_branch2 = time.time() - tin

        # Update
        tin = time.time()
        lls, updated_means, updated_covs, grads_obs, gain = vmap(_condition_on, in_axes=(0,0,None,None,None,None,None,None,None))(jnp.array(_sum_to_update.means), jnp.array(_sum_to_update.covariances), h, H_x, H_r, R, r0, u, y)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, jnp.array(_sum_to_update.weights))
        weights /= jnp.sum(weights)
        pre_weights = weights
        t_update = time.time() - tin

        # Optimal resampling
        tin = time.time()
        res_idx, weights = utils.optimal_resampling(weights, num_components[0], key)
        filtered_means = jnp.take(updated_means, res_idx, axis=0)
        filtered_covs = jnp.take(updated_covs, res_idx, axis=0)
        t_re = time.time() - tin


        # jax.debug.print("🤯 {x} 🤯", x=t)
        # jax.debug.print("🤯 {x} 🤯", x=updated_means)

        # Build carry and output states
        carry = containers._gaussian_sum_to_components(GaussianSum(list(filtered_means), list(filtered_covs), weights))
        outputs = {
            "weights": weights,
            "means": filtered_means,
            "covariances": filtered_covs
        }

        aux_outputs = {
            "Deltas": Deltas,
            "Lambdas": Lambdas,
            "grads_dyn": grads_dyn,
            "grads_obs": grads_obs,
            "gain": gain,
            "timing": jnp.array([t_autocov1, t_branch1, t_predict, t_recast1, t_autocov2, t_branch2, t_update, t_re]),
            "updated_means": updated_means,
            "pre_weights": pre_weights
        }

        return carry, (outputs, aux_outputs)
    
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    init_components = containers._gaussian_sum_to_components(GaussianSum(initial_means, initial_covs, initial_weights))
    carry = init_components

    carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       outputs["weights"], 
       outputs["means"], 
       outputs["covariances"]
    )

    return posterior_filtered, aux_outputs

def bootstrap_particle_filter(
    params: ParamsBPF,
    emissions: Float[Array, "ntime emission dim"],
    num_particles: Int,
    key: jr.PRNGKey = jr.PRNGKey(0),
    inputs: Optional[Float[Array, "ntime input dim"]] = None,
    ess_threshold: float = 0.5
):
    r"""
    Bootstrap particle filter for the nonlinear state space model.
    Args:
        params: Parameters of the nonlinear state space model.
        emissions: Emissions.
        num_particles: Number of particles.
        rng_key: Random number generator key.
        inputs: Inputs. 

    Returns:
        Posterior particle filtered.
    """

    num_timesteps = len(emissions)

    # Dynamics and emission functions
    f, h = params.dynamics_function, params.emission_function
    inputs = _process_input(inputs, num_timesteps)

    
    def _step(carry, t):
        weights, particles, key = carry
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_noise_covariance, 2, t)
        q0 = _get_params(params.dynamics_noise_bias, 2, t)
        R = _get_params(params.emission_noise_covariance, 2, t)
        r0 = _get_params(params.emission_noise_bias, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Sample new particles 
        keys = jr.split(key, num_particles+1)
        next_key = keys[0]
        map_sample_particles = vmap(params.sample_dynamics_distribution, in_axes=(0,0,None))
        new_particles = map_sample_particles(keys[1:], particles, u)

        # Compute weights 
        map_log_prob = vmap(params.emission_distribution_log_prob, in_axes=(0,None,None))
        lls = map_log_prob(new_particles, y, u)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, weights)
        new_weights = weights / jnp.sum(weights)

        # Resample if necessary
        resample_cond = 1.0 / jnp.sum(jnp.square(new_weights)) < ess_threshold * num_particles
        weights, new_particles, next_key = lax.cond(resample_cond, _resample, lambda *args: args, new_weights, new_particles, next_key)

        outputs = {
            'weights':weights,
            'particles':new_particles
        }

        carry = (weights, new_particles, next_key)

        return carry, outputs
    
    # Initialize carry
    keys = jr.split(key, num_particles+1)
    next_key = keys[0]
    weights = jnp.ones(num_particles) / num_particles
    map_sample = vmap(MVN(loc=params.initial_mean, covariance_matrix=params.initial_covariance).sample, in_axes=(None,0))
    particles = map_sample((), keys[1:])
    carry = (weights, particles, next_key)

    # scan
    _, outputs =  lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
 
    return outputs

