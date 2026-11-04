import jax #type: ignore
import jax.scipy.linalg #type: ignore
import jax.numpy as jnp #type: ignore
import jax.random as jr #type: ignore
import jax.tree_util as jtu #type: ignore
from functools import partial #type: ignore
from jax import lax, vmap, jacfwd, jacrev, jit, debug #type: ignore
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN #type: ignore
from typing import List, Optional, NamedTuple, Union
from util.numerics import _resample, _resample2, psd_solve, refit_gmm
from containers import GaussianComponent, GaussianSum, _branches_from_tree1, _branches_from_tree2
import containers
from jaxtyping import Array, Float, Int #type: ignore
from .steps import (
    _sqrt_condition_on,
    _sqrt_predict,
    _ukf_sqrt_condition_on_nonadditive,
    _ukf_sqrt_predict_nonadditive,
)

from parameters import ParamSSM
import time

import matplotlib.pyplot as plt #type: ignore

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y, 0)) if x is None else x
MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))

def swap_axes_on_values(outputs, axis1=0, axis2=1):

    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))

# Selection of the augmentation covariance Delta (before the prediction) and Lambda (before the update).
#
# Each rule has the signature rule(m, P, jacobian, hessian_tensor, bias, u, alpha, noise_jacobian, noise_cov) and
# returns (Delta, rho). P is the covariance Sigma of the component, alpha is r1 or r2 of the descriptor.
#   'prop': Delta = alpha * P.
#   'auto': Eq. (47), Delta = rho* P with rho* = alpha Tr(P J^T J) / sum_i Tr(H_i P)^2, clipped at RHO_MAX.
#           alpha = 2/N gives Eq. (47) of the paper.
#   'sdp':  the full program (45), min Psi(Delta) subject to 0 <= Delta <= Sigma, with the same weight 1/N = alpha/2.
# 'auto' and 'sdp' whiten the outputs by the noise covariance S = G R G^T (G = d output / d noise) when noise_cov is
# given and rescale is True. Then Psi does not depend on the units of the outputs. The filters give the noise only for
# the update step: the process noise of mtt is singular (G Q G^T has rank 2), so S^{-1/2} does not exist there.

RHO_MAX = 0.95   # Sigma - Delta must stay positive definite: the z-sample uses its Cholesky factor
RHO_MIN = 1e-3   # 'sdp' only: Delta must stay positive definite, the EKF/UKF step uses its Cholesky factor
SDP_ITERS = 200  # fixed number of FISTA iterations, so that the solver stays jittable


def prop_cov_selection(m, P, jacobian, hessian_tensor, bias, u, alpha, noise_jacobian=None, noise_cov=None):

    cov = alpha * P

    return cov, alpha


def _inv_sqrt_psd(S, rel_eps=1e-8):
    """S^{-1/2} of a symmetric positive semidefinite S. Eigenvalues below rel_eps * max are raised to that value."""

    lam, V = jnp.linalg.eigh(0.5 * (S + S.T))
    lam = jnp.maximum(lam, rel_eps * jnp.max(lam))
    return (V / jnp.sqrt(lam)) @ V.T


def whiten_outputs(J, hessians, S):
    """Derivatives of S^{-1/2} f from the derivatives of f: J -> S^{-1/2} J, H_i -> sum_j (S^{-1/2})_ij H_j."""

    W = _inv_sqrt_psd(S)
    return W @ J, jnp.einsum('ij,jkl->ikl', W, hessians)


def _output_derivatives(m, jacobian, hessian_tensor, bias, u, noise_jacobian, noise_cov, rescale):
    """J and H_i at the mean, whitened by the output noise covariance if noise_cov is given and rescale is True."""

    J = jacobian(m, bias, u)
    hessians = hessian_tensor(m, bias, u)
    if rescale and noise_cov is not None:
        G = noise_jacobian(m, bias, u)
        J, hessians = whiten_outputs(J, hessians, G @ noise_cov @ G.T)
    return J, hessians


def auto_cov_selection(m, P, jacobian, hessian_tensor, bias, u, alpha, noise_jacobian=None, noise_cov=None, rescale=True):

    J, hessian = _output_derivatives(m, jacobian, hessian_tensor, bias, u, noise_jacobian, noise_cov, rescale)
    rho = jnp.minimum(RHO_MAX, alpha * jnp.trace(P @ J.T @ J) / jnp.sum(jnp.trace((hessian) @ P, axis1=1, axis2=2)**2))
    cov = rho * P

    return cov, rho


# The full program (45) in the whitened form. With Sigma = C C^T and Delta = C P C^T, the constraint
# 0 <= Delta <= Sigma is 0 <= P <= I, and
#     Psi(P) = c Tr((I - P) A) + 1/4 sum_i Tr(P H_i)^2,   A = C^T J^T J C,  H_i = C^T H_i C,  c = 1/N.
# Psi is a convex quadratic. The solver is FISTA (Beck and Teboulle 2009) with the function-value restart of
# O'Donoghue and Candes (2015). The projection clips the eigenvalues of P to [0, 1].

def sdp_objective(P, A, hessians, c):

    t = jnp.einsum('ab,kab->k', P, hessians)
    return c * (jnp.trace(A) - jnp.sum(P * A)) + 0.25 * jnp.dot(t, t)


def sdp_gradient(P, A, hessians, c):

    t = jnp.einsum('ab,kab->k', P, hessians)
    return -c * A + 0.5 * jnp.einsum('k,kab->ab', t, hessians)


def sdp_fw_gap(P, G):
    """Frank-Wolfe gap <G, P> - sum_k min(lambda_k(G), 0). It is an upper bound of Psi(P) - min Psi (Jaggi 2013)."""

    lam = jnp.linalg.eigvalsh(0.5 * (G + G.T))
    return jnp.sum(P * G) - jnp.sum(jnp.minimum(lam, 0.0))


def _proj_unit_box(X):
    """Projection onto {0 <= P <= I}: clip the eigenvalues to [0, 1]."""

    lam, V = jnp.linalg.eigh(0.5 * (X + X.T))
    return (V * jnp.clip(lam, 0.0, 1.0)) @ V.T


def solve_cov_sdp(A, hessians, c, P0=None, num_iters=SDP_ITERS):
    """Minimize sdp_objective over 0 <= P <= I. Returns P.

    A: (d, d), hessians: (d_y, d, d), both in the whitened form. P0: start point (default 0.5 I). The function-value
    restart rejects each step that increases Psi, so Psi(P) <= Psi(P0) for any number of iterations.
    """

    d = A.shape[0]
    A = 0.5 * (A + A.T)
    hessians = 0.5 * (hessians + jnp.swapaxes(hessians, 1, 2))

    # Scale Psi to order 1. The minimizer does not change.
    gram = jnp.einsum('iab,jab->ij', hessians, hessians)
    lip = 0.5 * jnp.max(jnp.linalg.eigvalsh(gram))  # Lipschitz constant of the gradient
    scale = c * jnp.max(jnp.linalg.eigvalsh(A)) + lip
    scale = jnp.where(scale > 0, scale, 1.0)
    A, c_s, hessians = A / scale, c, hessians / jnp.sqrt(scale)
    step = 1.0 / jnp.maximum(lip / scale, 1e-3)

    x = _proj_unit_box(0.5 * jnp.eye(d) if P0 is None else P0)
    f = sdp_objective(x, A, hessians, c_s)

    def body(_, state):
        x, y, tk, f_old = state
        x_new = _proj_unit_box(y - step * sdp_gradient(y, A, hessians, c_s))
        f_new = sdp_objective(x_new, A, hessians, c_s)
        restart = f_new > f_old
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * tk**2))
        y_acc = x_new + ((tk - 1.0) / t_new) * (x_new - x)
        return (jnp.where(restart, x, x_new),
                jnp.where(restart, x, y_acc),
                jnp.where(restart, 1.0, t_new),
                jnp.where(restart, f_old, f_new))

    x, _, _, _ = lax.fori_loop(0, num_iters, body, (x, x, jnp.asarray(1.0, x.dtype), f))
    return x


def sdp_cov_selection(m, P, jacobian, hessian_tensor, bias, u, alpha, noise_jacobian=None, noise_cov=None, rescale=True):
    """Delta from the full program (45). The start point is the rho* of Eq. (47), so Psi(Delta) <= Psi(rho* Sigma).
    The eigenvalues of Sigma^{-1} Delta are clipped to [RHO_MIN, RHO_MAX]. rho is their mean."""

    J, hessians = _output_derivatives(m, jacobian, hessian_tensor, bias, u, noise_jacobian, noise_cov, rescale)
    C = jnp.linalg.cholesky(P)
    A = C.T @ J.T @ J @ C
    Hw = jnp.einsum('ba,kbc,cd->kad', C, hessians, C)
    c = 0.5 * alpha

    a = jnp.trace(A)
    b = jnp.sum(jnp.trace(Hw, axis1=1, axis2=2)**2)
    rho0 = jnp.where(b > 0, alpha * a / jnp.where(b > 0, b, 1.0), 1.0)
    P0 = jnp.clip(rho0, 0.0, 1.0) * jnp.eye(P.shape[0])

    Pw = solve_cov_sdp(A, Hw, c, P0)
    lam, V = jnp.linalg.eigh(Pw)
    lam = jnp.clip(lam, RHO_MIN, RHO_MAX)
    Pw = (V * lam) @ V.T
    cov = C @ Pw @ C.T
    cov = 0.5 * (cov + cov.T)

    return cov, jnp.mean(lam)


def rel_eigvals(Delta, P):
    """Eigenvalues of Sigma^-1 Delta (Sigma = P), in ascending order. They are the eigenvalues of C^-1 Delta C^-T with
    P = C C^T. 'prop' and 'auto' give d equal values (rho). 'sdp' can give different values in different directions."""

    C = jnp.linalg.cholesky(P)
    X = jax.scipy.linalg.solve_triangular(C, Delta, lower=True)
    X = jax.scipy.linalg.solve_triangular(C, X.T, lower=True)
    return jnp.linalg.eigvalsh(0.5 * (X + X.T))


COV_SELECTION_RULES = {
    'prop': prop_cov_selection,
    'auto': auto_cov_selection,
    'sdp': sdp_cov_selection,
}


def get_cov_selection(rule):
    """The selection function of the descriptor value ac1 or ac2."""

    if callable(rule):
        return rule
    if rule not in COV_SELECTION_RULES:
        raise ValueError(f"Unknown covariance rule {rule!r}. Use one of {sorted(COV_SELECTION_RULES)}.")
    return COV_SELECTION_RULES[rule]

class LAGSF():

    def __init__(self, props, ssm):

        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def run(
        self,
        key: jr.PRNGKey,
        params: ParamSSM,
        emissions: Float[Array, '...'],
        autocov1,
        r1,
        autocov2,
        r2,
        M,
        N,
        L,
        num_iter: int = 1,
        inputs: Optional[Float[Array, '...']] = None
        ):
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

        dynamics_fn = partial(self.ssm.dynamics_function, params)
        emission_fn = partial(self.ssm.emission_function, params)

        # Jacobian calculation
        F_x, H_x = jacfwd(dynamics_fn, argnums=0), jacfwd(emission_fn, argnums=0)
        F_q, H_r = jacfwd(dynamics_fn, argnums=1), jacfwd(emission_fn, argnums=1)
        F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
        inputs = _process_input(inputs, num_timesteps)

        # Rules for the augmentation covariances ('prop', 'auto' or 'sdp')
        autocov1 = get_cov_selection(autocov1)
        autocov2 = get_cov_selection(autocov2)

        def _step(carry, t):

            filtered_means, filtered_covs, weights, key = carry
            
            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]
            state_dim = filtered_covs[0].shape[0]

            # Autocov before predict
            filtered_covs_prev = filtered_covs
            Deltas, rho1s = vmap(lambda m, P: autocov1(m, P, F_x, F_xx, q0, u, r1, None, None))(filtered_means, filtered_covs)

            # Compute the z-sample
            key, subkey = jr.split(key)
            iid_sample = jr.normal(subkey, shape=(M, state_dim, N))
            mlt_matrices = vmap(jnp.linalg.cholesky)(filtered_covs-Deltas)
            centered_sample = vmap(jnp.matmul, in_axes=(0,0))(mlt_matrices, iid_sample)
            z_sample = jnp.expand_dims(filtered_means, axis=2) + centered_sample
            z_sample = jnp.swapaxes(z_sample, 1, 2)

            # Predict (square root EKF)
            # Convert Deltas (covariances) to square roots
            delta_sqrts = vmap(jnp.linalg.cholesky)(Deltas)
            map_predict = lambda means, S: vmap(_sqrt_predict, in_axes=(0,None,None,None,None,None,None,None))(means, S, dynamics_fn, F_x, F_q, Q, q0, u)
            predicted_means, predicted_sqrts, _ = vmap(map_predict, in_axes=(0,0))(z_sample, delta_sqrts)
            predicted_weights = jnp.tile(weights, (N,1)).T / N

            predicted_means = jnp.reshape(predicted_means, (M * N, state_dim))
            predicted_sqrts = jnp.reshape(predicted_sqrts, (M * N, state_dim, state_dim))
            predicted_weights = jnp.reshape(predicted_weights, (M * N,))

            # Prior inflation (apply in covariance space then back to sqrt)
            predicted_covs = vmap(lambda S: S @ S.T)(predicted_sqrts)
            predicted_covs = vmap(lambda cov: cov + 0.01 * jnp.eye(state_dim))(predicted_covs)
            predicted_sqrts = vmap(jnp.linalg.cholesky)(predicted_covs)

            # Autocov before update
            # Compute Lambdas using covariance space; predicted_covs reconstructed above
            Lambdas, rho2s = vmap(lambda m, P: autocov2(m, P, H_x, H_xx, r0, u, r2, H_r, R))(predicted_means, predicted_covs)

            # Compute the s-sample
            key, subkey = jr.split(key)
            s_iid_sample = jr.normal(subkey, shape=(M * N, state_dim, L))
            s_mlt_matrices = vmap(jnp.linalg.cholesky)(predicted_covs - Lambdas)
            s_centered_sample = vmap(jnp.matmul, in_axes=(0,0))(s_mlt_matrices, s_iid_sample)
            s_sample = jnp.expand_dims(predicted_means, axis=2) + s_centered_sample
            s_sample = jnp.swapaxes(s_sample, 1, 2)

            # Update (square root EKF)
            lambda_sqrts = vmap(jnp.linalg.cholesky)(Lambdas)
            map_condition_on = lambda means, S: vmap(partial(_sqrt_condition_on, residual=getattr(self.ssm, 'emission_residual', None)), in_axes=(0,None,None,None,None,None,None,None,None))(means, S, emission_fn, H_x, H_r, R, r0, u, y)
            lls, updated_means, updated_sqrts, _, _ = vmap(map_condition_on, in_axes=(0,0))(s_sample, lambda_sqrts)
            updated_weights = jnp.tile(predicted_weights, (L, 1)).T / L

            nans = jnp.isnan(lls)
            lls = jnp.where(nans, -jnp.inf, lls)
            # Weights in the log domain (see gsf.py)
            log_weights = jnp.log(updated_weights) + lls
            weights = jnp.exp(log_weights - jnp.max(log_weights))
            weights /= jnp.sum(weights)

            updated_means = jnp.reshape(updated_means, (M * N * L, state_dim))
            updated_sqrts = jnp.reshape(updated_sqrts, (M * N * L, state_dim, state_dim))
            weights = jnp.reshape(weights, (M * N * L,))

            # Resampling 
            key, subkey = jr.split(key)
            resampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(M, ), p=weights)
            filtered_means =  jnp.take(updated_means, resampled_idx, axis=0)
            filtered_sqrts = jnp.take(updated_sqrts, resampled_idx, axis=0)
            weights = jnp.ones(shape=(M, )) /M

            # Convert to covariances for output and add small inflation
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrts)
            filtered_covs = vmap(lambda cov: cov + 0.01 * jnp.eye(state_dim))(filtered_covs)

            # Build carry and output states
            carry = (filtered_means, filtered_covs, weights, key)
            
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            aux_outputs = {
                "Deltas": Deltas,
                "Lambdas": Lambdas,
                "rho1s": rho1s,
                "rho2s": rho2s,
                "eig1s": vmap(rel_eigvals)(Deltas, filtered_covs_prev),    # eigenvalues of Sigma^-1 Delta, (M, d)
                "eig2s": vmap(rel_eigvals)(Lambdas, predicted_covs)        # eigenvalues of Sigma^-1 Lambda, (M N, d)                
            }

            return carry, (outputs, aux_outputs)
        
        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(M, key)
        initial_covs = jnp.array([params.initial.cov.value for i in range(M)])
        initial_weights = jnp.ones(shape=(M,)) / M

        carry = (initial_means, initial_covs, initial_weights, key)

        carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)

        return outputs, aux_outputs

class UAGSF():

    def __init__(self, props, ssm):
        self.props = props
        self.ssm = ssm
        self.xparam = None
        self.time = None

    def run(
        self,
        key: jr.PRNGKey,
        params: ParamSSM,
        emissions: Float[Array, '...'],
        autocov1,
        r1,
        autocov2,
        r2,
        M,
        N,
        L,
        num_iter: int = 1,
        alpha: Float = 1e-3, 
        beta: Float = 2.0, 
        kappa: Float = 0.0,
        inputs: Optional[Float[Array, '...']] = None
    ):
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

        dynamics_fn = partial(self.ssm.dynamics_function, params)
        emission_fn = partial(self.ssm.emission_function, params)

        # Dynamics and emission functions and their Jacobians
        F_x, H_x = jacfwd(dynamics_fn, argnums=0), jacfwd(emission_fn, argnums=0)
        H_r = jacfwd(emission_fn, argnums=1)
        F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))
        inputs = _process_input(inputs, num_timesteps)

        # Rules for the augmentation covariances ('prop', 'auto' or 'sdp')
        autocov1 = get_cov_selection(autocov1)
        autocov2 = get_cov_selection(autocov2)

        def _step(carry, t):

            filtered_means, filtered_covs, weights, key = carry
            
            # Get parameters and inputs for time index t
            Q = _get_params(params.dynamics.cov.value, 2, t)
            q0 = _get_params(params.dynamics.bias.value, 2, t)
            R = _get_params(params.emissions.cov.value, 2, t)
            r0 = _get_params(params.emissions.bias.value, 2, t)
            u = inputs[t]
            y = emissions[t]
            state_dim = filtered_covs[0].shape[0]

            # Autocov 1
            filtered_covs_prev = filtered_covs
            Deltas, rho1s = vmap(lambda m, P: autocov1(m, P, F_x, F_xx, q0, u, r1, None, None))(filtered_means, filtered_covs)

            # Compute the z-sample
            key, subkey = jr.split(key)
            iid_sample = jr.normal(subkey, shape=(M, state_dim, N))
            mlt_matrices = vmap(jnp.linalg.cholesky)(filtered_covs-Deltas)
            centered_sample = vmap(jnp.matmul, in_axes=(0,0))(mlt_matrices, iid_sample)
            z_sample = jnp.expand_dims(filtered_means, axis=2) + centered_sample
            z_sample = jnp.swapaxes(z_sample, 1, 2)

            # Predict (square root UKF)
            delta_sqrts = vmap(jnp.linalg.cholesky)(Deltas)
            map_predict = lambda means, S: vmap(_ukf_sqrt_predict_nonadditive, in_axes=(0,None,None,None,None,None,None,None,None))(means, S, dynamics_fn, u, Q, alpha, beta, kappa, q0)
            predicted_means, predicted_sqrts = vmap(map_predict, in_axes=(0,0))(z_sample, delta_sqrts)
            predicted_weights = jnp.tile(weights, (N, 1)).T / N

            predicted_means = jnp.reshape(predicted_means, (M * N, state_dim))
            predicted_sqrts = jnp.reshape(predicted_sqrts, (M * N, state_dim, state_dim))
            predicted_weights = jnp.reshape(predicted_weights, (M * N,))

            # Prior inflation
            scale = 0.01
            predicted_covs = vmap(lambda S: S @ S.T)(predicted_sqrts)
            predicted_covs = predicted_covs + scale * jnp.eye(state_dim)[None, :, :]
            predicted_sqrts = vmap(jnp.linalg.cholesky)(predicted_covs)

            # Autocov before update
            Lambdas, rho2s = vmap(lambda m, P: autocov2(m, P, H_x, H_xx, r0, u, r2, H_r, R))(predicted_means, predicted_covs) 

            # Compute the s-sample
            key, subkey = jr.split(key)
            s_iid_sample = jr.normal(subkey, shape=(M*N, state_dim, L))
            s_mlt_matrices = vmap(jnp.linalg.cholesky)(predicted_covs - Lambdas)
            s_centered_sample = vmap(jnp.matmul, in_axes=(0,0))(s_mlt_matrices, s_iid_sample)
            s_sample = jnp.expand_dims(predicted_means, axis=2) + s_centered_sample
            s_sample = jnp.swapaxes(s_sample, 1, 2)

            # Update (square root UKF)
            lambda_sqrts = vmap(jnp.linalg.cholesky)(Lambdas)
            map_condition_on = lambda means, S: vmap(partial(_ukf_sqrt_condition_on_nonadditive, residual=getattr(self.ssm, 'emission_residual', None)), in_axes=(0,None,None,None,None,None,None,None,None,None))(means, S, emission_fn, R, u, y, alpha, beta, kappa, r0)
            lls, updated_means, updated_sqrts = vmap(map_condition_on, in_axes=(0,0))(s_sample, lambda_sqrts)
            updated_weights = jnp.tile(predicted_weights, (L,1)).T / L

            # Weights in the log domain (see gsf.py)
            log_weights = jnp.log(updated_weights) + lls
            weights = jnp.exp(log_weights - jnp.max(log_weights))
            weights /= jnp.sum(weights)

            updated_means = jnp.reshape(updated_means, (M*N*L, state_dim))
            updated_sqrts = jnp.reshape(updated_sqrts, (M*N*L, state_dim, state_dim))
            weights = jnp.reshape(weights, (M*N*L,))

            # Resampling 
            key, subkey = jr.split(key)
            resampled_idx = jr.choice(subkey, jnp.arange(weights.shape[0]), shape=(M, ), p=weights)
            filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
            filtered_sqrts = jnp.take(updated_sqrts, resampled_idx, axis=0)
            weights = jnp.ones(shape=(M,)) / M
            filtered_covs = vmap(lambda S: S @ S.T)(filtered_sqrts)

            # Build carry and output states
            carry = (filtered_means, filtered_covs, weights, key)
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            aux_outputs = {
                "Deltas": Deltas,
                "Lambdas": Lambdas,
                "rho1s": rho1s,
                "rho2s": rho2s,
                "eig1s": vmap(rel_eigvals)(Deltas, filtered_covs_prev),    # eigenvalues of Sigma^-1 Delta, (M, d)
                "eig2s": vmap(rel_eigvals)(Lambdas, predicted_covs)        # eigenvalues of Sigma^-1 Lambda, (M N, d)
            }   

            return carry, (outputs, aux_outputs)

        
        initial_means = MVN(params.initial.mean.value, params.initial.cov.value).sample(M, key)
        initial_covs = jnp.array([params.initial.cov.value for i in range(M)])
        initial_weights = jnp.ones(shape=(M,)) / M

        carry = (initial_means, initial_covs, initial_weights, key)

        carry, (outputs, aux_outputs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
        outputs = swap_axes_on_values(outputs)
        
        return outputs, aux_outputs

