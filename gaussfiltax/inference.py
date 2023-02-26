import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from functools import partial
from jax import lax, vmap, jacfwd, jacrev, debug, device_put, jit
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
import gaussfiltax.utils as utils
from gaussfiltax.containers import GaussianComponent, GaussianSum, _branches_from_tree
from gaussfiltax import containers
from jaxtyping import Array, Float, Int


from dynamax.utils.utils import psd_solve
from dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
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

def _predict(m, P, f, F, Q, u):
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
    F_x = F(m, u)
    mu_pred = f(m, u)
    Sigma_pred = F_x @ P @ F_x.T + Q
    # print("mu_pred", mu_pred)
    # print("Sigma_pred", Sigma_pred)
    return mu_pred, Sigma_pred

def _condition_on(m, P, h, H, R, u, y):
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
    H_x = H(m, u)
    S = R + H_x @ P @ H_x.T
    K = psd_solve(S, H_x @ P).T
    posterior_cov = P - K @ S @ K.T
    posterior_mean = m + K @ (y - h(m, u))
    ll = _MVN_log_prob(h(m, u), S, y)
    return ll, posterior_mean, posterior_cov

def _split_single_component(m, P, Delta, N, key=jr.PRNGKey(0)):
    r"""Split a Gaussian component into N Gaussian particles.
    Args:
        m (D_hid,): mean.
        P (D_hid,D_hid): covariance.
        Delta (D_hid,D_hid): covariance of Gaussian particles.
        N (int): number of particles.

    Returns:
        particles (N,D_hid): array of particles.
    """
    latent_dist = MVN(m, P - Delta)
    particles = latent_dist.sample((N,), seed=key)
    return particles

def _split_multiple_components(mus, covs, Deltas, n_split, key=jr.PRNGKey(0)):
    num_components = covs.shape[0]
    keys = jr.split(key, num_components)
    all_particles = []
    for m in range(num_components):
        particles = _split_single_component(mus[m], covs[m], Deltas[m], n_split[m], keys[m])
        all_particles.append(particles)
    return all_particles

def _autocov(m, P, hessian_tensor, num_particles, u, args):
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
    # state_dim x state_dim matrix.
    state_dim = P.shape[0]
    _hessian = hessian_tensor(m, u)
    emission_dim = _hessian.shape[0]
    cov_init = P                               # This is something that should be specified by the user / adapted
    cov_cutoff = 0.5 * P      # This is something that should be specified by the user / adapted
    nsteps_gd = args[0]
    eta_gd = args[1]
    alpha = args[2]
    alpha = alpha / num_particles # alpha = lipschitz ** 2 / number of particles    
    Delta = utils.sdp_opt_test(state_dim, emission_dim, alpha, cov_init, cov_cutoff, _hessian, nsteps_gd, eta_gd)
#    Delta = 0.5 * jnp.eye(state_dim)
    return Delta, num_particles

def gaussian_sum_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    num_components: int = 1,
    num_iter: int = 1,
    inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    output_fields: Optional[List[str]]=["weights", "means", "covariances", "predicted_means", "predicted_covariances"],
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
    F, H = jacfwd(f), jacfwd(h)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    f_vec, h_vec, F_vec, H_vec = (vmap(fn, in_axes=(0, None)) for fn in (f, h, F, H)) # vmap over components, component axis is 0
    inputs = _process_input(inputs, num_timesteps)
    MVN_log_prob_vec = vmap(_MVN_log_prob, in_axes=(0, 0, None))

    def _step(carry, t):
        weights, pred_means, pred_covs = carry

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on this emission
        lls, filtered_means, filtered_covs = vmap(_condition_on, in_axes=(0, 0, None, None, None, None, None))(pred_means, pred_covs, h, H, R, u, y)
        # Compute weights
        lls -= jnp.max(lls)
        loglik_weights = jnp.exp(lls)
        weights = jnp.multiply(loglik_weights, weights)
        weights /= jnp.sum(weights)

        # Predict the next state
        pred_means, pred_covs = vmap(_predict, in_axes=(0, 0, None, None, None, None))(filtered_means, filtered_covs, f, F, Q, u)

        # Build carry and output states
        carry = (weights, pred_means, pred_covs)
        outputs = {
            "means": filtered_means,
            "covariances": filtered_covs,
            "predicted_means": pred_means,
            "predicted_covariances": pred_covs,
            "weights": weights
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

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
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission dim"],
    num_components: Int[Array, "1 3"],
    rng_key: jr.PRNGKey = jr.PRNGKey(0),
    num_iter: int = 1,
    opt_args : Optional[Float[tuple, "1 3"]] = (20, 0.1, 0.1),
    inputs: Optional[Float[Array, "ntime input dim"]] = None,
    output_fields: Optional[List[str]]=["weights", "means", "covariances", "Deltas", "Lambdas"]
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
    F, H = jit(jacfwd(f)), jit(jacfwd(h))
    FH, HH = jit(jacrev(F)), jit(jacrev(H))
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    f_vec, h_vec, F_vec, H_vec = (vmap(fn, in_axes=(0, None)) for fn in (f, h, F, H)) # vmap over components, component axis is 0
    inputs = _process_input(inputs, num_timesteps)
    MVN_log_prob_vec = vmap(_MVN_log_prob, in_axes=(0, 0, None))

    def _step(carry, t):
        filtered_components = carry
        filtered_sum = containers._components_to_gaussian_sum(filtered_components)
        filtered_means = jnp.array(filtered_sum.means)
        filtered_covs = jnp.array(filtered_sum.covariances)
        filtered_weights = jnp.array(filtered_sum.weights)
        
        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        y = emissions[t]

        # Autocov 1
        tin = time.time()
        nums_to_split = jnp.array([num_components[1]]*num_components[0])
        Deltas, nums_to_split = vmap(_autocov, in_axes=(0, 0, None, 0, None, None))(filtered_means, filtered_covs, FH, nums_to_split, u, opt_args)
#        print("Autocov #1 time: ", time.time() - tin)

        # Branch 1
        key, subkey = jr.split(rng_key)
        tin = time.time()
        _components_to_predict = _branches_from_tree(filtered_components, list(Deltas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_predict = containers._components_to_gaussian_sum(leaves)
#        print("Branch #1 time: ", time.time() - tin)

        # Predict
        tin = time.time()
        predicted_means, predicted_covs = vmap(_predict, in_axes=(0,0,None,None,None,None))(jnp.array(_sum_to_predict.means), jnp.array(_sum_to_predict.covariances), f, F, Q, u)
#        print("Predict time: ", time.time() - tin)

        # Recast 1
        tin = time.time()
        predicted_sum = GaussianSum(list(predicted_means), list(predicted_covs), _sum_to_predict.weights)
        predicted_components = containers._gaussian_sum_to_components(predicted_sum)
#        print("Recast time: ", time.time() - tin)

        # Autocov before update
        tin = time.time()
        nums_to_split = jnp.array([num_components[2]] * num_components[0]*num_components[1])
        Lambdas, nums_to_split = vmap(_autocov, in_axes=(0, 0, None, 0, None, None))(predicted_means, predicted_covs, FH, nums_to_split, u, opt_args)
#        print("Autocov #2 time: ", time.time() - tin)


        # Branching before update
        key, subkey = jr.split(key)
        tin = time.time()
        _components_to_update = _branches_from_tree(predicted_components, list(Lambdas), list(nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_update, is_leaf=lambda x: isinstance(x, GaussianComponent))
        _sum_to_update = containers._components_to_gaussian_sum(leaves)
#        print("Branch #2 time: ", time.time() - tin)

        # Update
        tin = time.time()
        lls, updated_means, updated_covs = vmap(_condition_on, in_axes=(0,0,None,None,None,None,None))(jnp.array(_sum_to_update.means), jnp.array(_sum_to_update.covariances), h, H, R, u, y)
        lls -= jnp.max(lls)
        ls = jnp.exp(lls)
        weights = jnp.multiply(ls, jnp.array(_sum_to_update.weights))
        weights /= jnp.sum(weights)
#        print("Update time: ", time.time() - tin)

        # Recast 2
        tin = time.time()
        updated_sum = GaussianSum(list(updated_means), list(updated_covs), weights)
        updated_components = containers._gaussian_sum_to_components(updated_sum)
#        print("Recast #2 time: ", time.time() - tin)

        # Resampling 
        tin - time.time()
        resampled_idx = jr.choice(jr.PRNGKey(0), jnp.arange(weights.shape[0]), shape=(num_components[0], ), p=weights)
        filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
        filtered_covs = jnp.take(updated_covs, resampled_idx, axis=0)
        weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
#        print("Resampling time: ", time.time() - tin)

        # Build carry and output states
        carry = filtered_components
        outputs = {
            "weights": weights,
            "means": filtered_means,
            "covariances": filtered_covs,
            "Deltas": Deltas,
            "Lambdas": Lambdas
        }

        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs
        
    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    init_components = containers._gaussian_sum_to_components(GaussianSum(initial_means, initial_covs, initial_weights))
    carry = init_components

    carry, outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = swap_axes_on_values(outputs)
    posterior_filtered = PosteriorGaussianSumFiltered(
       outputs["weights"], outputs["means"], outputs["covariances"]
    )
    aux_outputs = {'Deltas': outputs['Deltas'], 'Lambdas': outputs['Lambdas']}
    
    return posterior_filtered, aux_outputs

    # posterior_filtered = _step(carry, 0)
    # return posterior_filtered