import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore
from jax import vmap, debug, lax, random as jr # type: ignore
import optax
import jax
import jax.scipy.stats as jss # type: ignore
from jax.scipy.special import logsumexp as lse # type: ignore
import jax.scipy.stats as stats # type: ignore
import jax.scipy as scipy # type: ignore
from jax import jit # type: ignore
from functools import partial 
from typing import Optional, Callable
from jax import tree_map # type: ignore
import random

def kde_error(particles, true_cps, sigma=1.0):
        
    num_prt = particles.shape[0]
    dim = particles.shape[1]
    dps = particles - true_cps
    ds = - jnp.linalg.norm(dps, axis=1)**2 / 2 / sigma**2
    logp = lse(ds) #- jnp.log(num_prt) - (dim/2) * jnp.log(2 * jnp.pi * sigma ** 2)

    return -logp


def min_error(data, inf, true_states):

    means, _ = get_means_covs(data, inf, with_covs=False)
    me_t = jnp.min(jnp.linalg.norm(means - true_states[None, :, :], axis=2), axis=0)
    
    return jnp.mean(me_t), me_t


def log_prob_t(means, covs, true_state, weights=None, eps: float = 1e-6):

    n_comps = means.shape[0]
    log_w = -jnp.log(n_comps) if weights is None else jnp.log(weights)
    ds = means - true_state
    covs = 0.5 * (covs + jnp.swapaxes(covs, -1, -2))
    covs = covs + eps * jnp.eye(covs.shape[-1])
    inv_covs = jnp.linalg.inv(covs)
    ds = jnp.einsum('ij, ijk, ik -> i', ds, inv_covs, ds)
    ds = -0.5 * ds
    ds = ds - 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * covs))
    logp = lse(ds + log_w)

    return logp

def get_weights(data, inf):
    """Normalized weights of the particles or mixture components, shape (n_comps, T).
    Filters that resample at every step give equal weights."""

    outputs = data[0] if inf == 'bpf' or inf == 'apf' else data
    weights = outputs['weights']
    w_sum = jnp.sum(weights, axis=0, keepdims=True)

    return weights / w_sum

def mse(data, inf, true_states):

    means, _ = get_means_covs(data, inf, with_covs=False)
    weights = get_weights(data, inf)
    mean_est = jnp.einsum('kt,ktd->td', weights, means)
    mse_t = vmap(jnp.linalg.norm)(mean_est - true_states)**2

    return jnp.mean(mse_t), mse_t

def get_means_covs(data, inf, eps: float = 1e-6, with_covs: bool = True):
    """Means and covariances of the mixture components. With with_covs=False the covariances of BPF and APF
    (n_prt copies of the same matrix, large for many particles) are not built and None is returned for them."""

    if inf == 'bpf' or inf == 'apf':

        means = data[0]['particles']
        weights = get_weights(data, inf)
        n_prt, _, _ = means.shape

        if not with_covs:

            return means, None

        def _kde_cov(particles_t, weights_t):
            d = particles_t.shape[1]
            mean = weights_t @ particles_t
            std = jnp.sqrt(weights_t @ (particles_t - mean) ** 2)
            std = jnp.where(std > 0.0, std, eps)
            h = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * (n_prt ** (-1.0 / (d + 4.0)))
            diag = (h * std) ** 2
            return jnp.diag(diag)

        def _cov_or_kde(t):
            # Weighted sample covariance of the particle cloud at time t.
            cov_t = jnp.atleast_2d(jnp.cov(means[:, t, :], rowvar=False, aweights=weights[:, t]))
            cov_t = 0.5 * (cov_t + cov_t.T)
            eigvals = jnp.linalg.eigvalsh(cov_t)
            collapse = jnp.any(jnp.isnan(cov_t)) | jnp.any(jnp.isinf(cov_t)) | (jnp.min(eigvals) <= eps)
            cov_t = jnp.where(collapse, _kde_cov(means[:, t, :], weights[:, t]), cov_t)
            cov_t = cov_t + eps * jnp.eye(cov_t.shape[0])
            return cov_t

        e_covs = vmap(_cov_or_kde)(jnp.arange(means.shape[1]))
        covs = jnp.tile(e_covs / n_prt, (n_prt, 1, 1, 1))

    else:

        means = data['means']
        covs = data['covariances']

    return means, covs


def log_prob_error(data, inf, true_states):

    means, covs = get_means_covs(data, inf)
    weights = get_weights(data, inf)
    n_steps = true_states.shape[0]
    lp_errs = lax.map(lambda t: -log_prob_t(means[:, t], covs[:, t], true_states[t], weights[:, t]), jnp.arange(n_steps))

    return jnp.mean(lp_errs), lp_errs


def bias(data, inf, true_states):

    means, _ = get_means_covs(data, inf, with_covs=False)
    weights = get_weights(data, inf)
    bias_t = vmap(jnp.linalg.norm)(jnp.einsum('kt,ktd->td', weights, means) - true_states)

    return jnp.mean(bias_t), bias_t

def get_num_sims(exp_desc):

    sim_desc = exp_desc.sim
    inf_desc = exp_desc.inf

    if inf_desc.name == 'lgsf':

        n_sims = 4 * inf_desc.n_comps * sim_desc.num_timesteps

    elif inf_desc.name == 'ugsf':

        n_sims = 2 * (2 * sim_desc.state_dim + 1) * inf_desc.n_comps * sim_desc.num_timesteps

    elif inf_desc.name == 'lagsf':

        n_sims = 3 * inf_desc.M * inf_desc.N * (inf_desc.L + 1) * sim_desc.num_timesteps

    elif inf_desc.name == 'uagsf':

        n_sims = (2 * sim_desc.state_dim) * inf_desc.M * inf_desc.N * (inf_desc.L + 1) * sim_desc.num_timesteps

    elif inf_desc.name == 'bpf':

        n_sims = inf_desc.num_prt * sim_desc.num_timesteps

    elif inf_desc.name == 'apf':

        n_sims = inf_desc.num_prt * sim_desc.num_timesteps

    return n_sims

def sdev(data, inf):

    means, _ = get_means_covs(data, inf)

    dps = particles - true_cps
    nsq = jnp.linalg.norm(dps, axis=1) ** 2
    mse = jnp.mean(nsq)

    return jnp.sqrt(mse)


def bootstrap(key, rmse_array, B):

    N = rmse_array.shape[0]
    rmse_boot = jnp.zeros((B,))
    boot_samples = []

    for b in range(B):
        key, subkey = jr.split(key)
        ids = jr.randint(subkey, (N,), 0, N)
        boot = rmse_array[ids]
        rmse_boot = rmse_boot.at[b].set(jnp.mean(boot))
        boot_samples.append(boot)
    boot_samples = jnp.stack(boot_samples, axis=0)

    return rmse_boot, boot_samples


def compute_distances(emissions, observations, num_timesteps, emission_dim):

    # ensure emissions has a leading dimension
    if emissions.ndim == 2:
        emissions = emissions[None, ...]
    
    distances = vmap(lambda sim_emissions: jnp.linalg.norm(observations - sim_emissions) / jnp.sqrt(num_timesteps * emission_dim))(emissions)

    return distances


def clear_nans(errors):

    means = errors[:, 0]
    stds = errors[:, 1]
    nsims = errors[:, 2]
    nans_infs = jnp.isnan(means) + jnp.isinf(means)
    means = means[~nans_infs]
    stds = stds[~nans_infs]
    nsims = jnp.log(nsims[~nans_infs])
    
    nfail = jnp.sum(nans_infs)
    success_pct = (1 - nfail / errors.size)
    errors = errors[~jnp.isnan(errors)]
    errors = errors[~jnp.isinf(errors)]

    out = jnp.array([means, stds, nsims]).T

    return out, nfail, success_pct


def compute_acf(y, max_lag):

    """Computes the autocorrelation function (ACF) for each dimension and the autocorrelation matrix."""
    T, _ = y.shape
    y_mean = jnp.mean(y, axis=0)
    L = jnp.tril(jnp.ones((T, T)))
    y = y - y_mean
    
    if max_lag >= T:
    
        raise ValueError("max_lag should be smaller than the number of time steps T to avoid excessive computation.")
    
    map_acf_frobenius = lambda k: jnp.linalg.norm(jnp.abs(jnp.einsum('i,ij,ik -> jk', L[T-k], y, jnp.roll(y, -k, axis=0)) / jnp.sum(L[T-k])), 'fro')
    
    return vmap(map_acf_frobenius)(jnp.arange(max_lag + 1))


def kmeans(key, data, num_clusters, tol=1e-4):
    dx = data.shape[1]

    def cond_fun(carry):
        old_centroids, new_centroids = carry
        any_nans = jnp.isnan(new_centroids).any()
        return jnp.logical_and(jnp.linalg.norm(old_centroids - new_centroids) > tol, any_nans)

    def _step(carry):
        _, centroids = carry
        distances = vmap(lambda x: vmap(lambda x, mu: jnp.linalg.norm(x - mu), in_axes=(None, 0))(x, centroids))(data)
        cluster_assignments = jnp.argmin(distances, axis=1)
        new_centroids = vmap(lambda i: jnp.sum(jnp.where(jnp.tile(cluster_assignments==i, (dx,1)).T, data, 0.0), axis=0) / jnp.sum(cluster_assignments==i))(jnp.arange(num_clusters))
        carry = (centroids, new_centroids)
        return carry

    init_carry = (jnp.zeros((num_clusters, dx)), jr.choice(key, data, (num_clusters,)))
    out = lax.while_loop(cond_fun, _step, init_carry)
    
    return out[1]


def ratio(x, alpha, exp_matrix_gen, sample, sigma):

    exp_values = vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, sample, sigma)
    ratio = alpha @ exp_values

    return ratio, exp_values


def exp_matrix_gen(x, y, sigma): 
    return jnp.exp(-(0.5 / sigma**2) * jnp.linalg.norm(x - y)**2)


def find_quantile(prev_sample, new_sample, sigma=1.0):
    '''Implements the method from the paper Adaptive Approximate Bayesian Computation
    Tolerance Selection by Simola et al. to find the quantile parameter for the
    '''

    n_samples = prev_sample.shape[0]
    
    exp_matrix = vmap(lambda x: vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, prev_sample, sigma))(new_sample)
    exp_matrix_0 = vmap(lambda x: vmap(exp_matrix_gen, in_axes=(None, 0, None))(x, prev_sample, sigma))(prev_sample)
    e0 = jnp.sum(exp_matrix_0, axis=0) / n_samples

    def cond_fn(carry):
        _, err = carry
        return err > .1

    def step(carry):

        prev_alpha, err = carry
        b = 1 / (exp_matrix @ prev_alpha)
        new_alpha = prev_alpha * (exp_matrix.T @ b) / e0 / n_samples
        err = jnp.linalg.norm(new_alpha - prev_alpha)
        carry = new_alpha, err
    
        return carry

    alpha_star, _ = lax.while_loop(cond_fn, step, (jnp.ones(n_samples), 2.0))

    def _cond_fn(carry):
        _, err = carry
        return err > 0.01

    def _step(carry):

        prev_x, err = carry
        r, exp_values = ratio(prev_x, alpha_star, exp_matrix_gen, prev_sample, sigma)
        new_x = jnp.einsum('i,i,ij->j', alpha_star, exp_values, prev_sample) / r
        err = jnp.linalg.norm(new_x - prev_x)
        carry = new_x, err
    
        return carry

    i_star = jnp.argmax(alpha_star)
    x_star, _ = lax.while_loop(_cond_fn, _step, (prev_sample[i_star], 2.0))
    c, _ = ratio(x_star, alpha_star, exp_matrix_gen, prev_sample, sigma)
    q = 1 / c

    return q


def average_min_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    mod_ds = ds + jnp.diag(jnp.full((ds.shape[0],), jnp.inf))
    min_ds = jnp.min(mod_ds, axis=1)
    avg_min_distance = jnp.mean(min_ds)

    return avg_min_distance


def average_pair_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    mod_ds = jnp.triu(ds)[jnp.triu(ds)>0]
    mod_ds = mod_ds.flatten()

    return jnp.mean(mod_ds)


def average_max_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)
    max_ds = jnp.max(ds, axis=1)

    return jnp.mean(max_ds)


def max_distance(particles):

    particles = jnp.unique(particles, axis=0)
    ds = vmap(lambda y: vmap(lambda x: jnp.linalg.norm(x-y))(particles))(particles)

    return jnp.max(ds)


def compute_all_errors(cps, true_cps):

    kderr = kde_error(cps, true_cps)
    minerr = min_error(cps, true_cps)
    pmerr = post_mean_error(cps, true_cps)
    rmserr = rmse(cps, true_cps)
    
    amd = average_min_distance(cps)
    apd = average_pair_distance(cps)
    amxd = average_max_distance(cps)
    mxd = max_distance(cps)

    return kderr, minerr, pmerr, rmserr, amd, apd, amxd, mxd


def collapse(mean_mat, covariance_tens, weight_vec):
    M, dx = np.shape(mean_mat)
    mean_out = np.matmul(weight_vec, mean_mat)
    cov_out = np.zeros([dx, dx])
    for m in range(M):
        diff = mean_mat[m] - mean_out
        cov_out = cov_out + weight_vec[m] * (covariance_tens[m] +
                                             np.tensordot(diff, diff, axes=0))
    return mean_out, cov_out


def dec_to_base(num, base):  # Maximum base - 36
    base_num = ""
    while num > 0:
        dig = int(num % base)
        if dig < 10:
            base_num += str(dig)
        else:
            base_num += chr(ord('A') + dig - 10)  # Using uppercase letters
        num //= base
    base_num = base_num[::-1]  # To reverse the string
    return base_num


def normal_KL_div(mean1, mean2, cov1, cov2):
    d = np.shape(cov1)[0]
    Omega = np.linalg.inv(cov2)
    KL = np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) - d + \
         np.matmul(np.matmul(np.transpose(mean1 - mean2), Omega), (mean1 - mean2)) + np.trace(Omega * cov1)
    return KL / 2


def split_by_sampling(mean, cov, new_cov, num_comp):
    dcov = cov - new_cov
    dx = np.shape(mean)[0]
    if dx == 1:
        new_means = np.random.normal(mean, dcov, (num_comp, 1))
    else:
        new_means = np.random.multivariate_normal(mean, dcov, num_comp)
    return new_means


def split_to_sigma_points(mean, cov, lamda):
    dx = np.shape(mean)[0]
    sigma_points = np.zeros((2 * dx + 1, dx))
    sigma_points[0] = mean
    if dx == 1:
        sqrtCov = np.sqrt(cov)
        sigma_points[1] = mean + np.sqrt(dx + lamda) * sqrtCov
        sigma_points[2] = mean - np.sqrt(dx + lamda) * sqrtCov
    else:
        sqrtCov = np.linalg.cholesky(cov)
        for i in range(dx):
            sigma_points[i + 1] = mean + np.sqrt(dx + lamda) * sqrtCov.T[i]
            sigma_points[dx + i + 1] = mean - np.sqrt(dx + lamda) * sqrtCov.T[i]
    return sigma_points


def gm(x, means, sigma, num_comp):
    out = 0
    for mean in means:
        out += stats.norm.pdf(x, mean, sigma) / num_comp
    return out


def gaussian_logpdf(y, m, S):
    D = m.shape[0]
    L = np.linalg.cholesky(S)
    x = np.linalg.solve(L, np.transpose(y - m))
    return -0.5 * D * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(x ** 2)


def loss(D, Pv, L, Nv, H):
    return (2 * L ** 2 / Nv) * np.trace(Pv - D) + (1 / 4) * np.trace(np.matmul(D, H)) ** 2


def matrix_projection(A, B):
    return (np.trace(np.matmul(A.T, B)) / np.trace(np.matmul(B.transpose(), B))) * B

@jit
def project_to_psd(Delta):
    evals, evec = jnp.linalg.eigh(Delta)
    new_evals = jnp.multiply(evals > 0, evals)
    new_Delta = evec @ jnp.diag(new_evals) @ evec.T
    Delta = (new_Delta + new_Delta.T) / 2
    return Delta.astype(jnp.float32)


def gradient_descent(dim, N, L, X0, P, H, Nsteps, eta):
    X = X0
    for i in range(Nsteps):
        X = X - eta * (-(2 * L ** 2 / N) * np.eye(dim) + (1 / 2) * np.trace(np.matmul(H, X)) * H)
    return X

_vec = lambda x, n: jnp.reshape(x, (n**2, ))
_mat = lambda x, n: jnp.reshape(x, (n, n))
_matrices_to_vectors = lambda matrix_array, n: jnp.array(list(vmap(lambda x : jnp.reshape(x, (n**2, )))(matrix_array)))
_vectors_to_matrices = lambda vector_array, n: jnp.array(list(vmap(lambda x : jnp.reshape(x, (n, n)))(vector_array)))

@partial(jit, static_argnums=(0,))
def sdp_opt(state_dim, N, P, jacobian, hessian, beta, tol=0.1):
    """
    This works by blancing the two terms using alpha
    """
    # construct 2nd order term
    vec_P = _vec(P, state_dim)
    vec_hessians = _matrices_to_vectors(hessian, state_dim)
    low_rank = jnp.zeros((state_dim**2, state_dim**2))
    for i in range(state_dim):
        low_rank += vec_hessians[i] * vec_hessians[i].T
    lhs = (1/4) * low_rank + jnp.eye(state_dim**2)
    vec_J = _vec(jacobian.T @ jacobian, state_dim)
    alpha = beta * (N/4) * jnp.dot(vec_P.T, low_rank @ vec_P) / jnp.dot(vec_P.T, vec_J)
    aid = alpha * vec_J / N 

    # looping step
    def _step(val):
        vec_delta = val[0]
        rhs = aid + vec_delta
        new_vec_delta = jnp.linalg.solve(lhs, rhs)
        Delta = _mat(new_vec_delta, state_dim)
        Delta = project_to_psd(Delta)
        Delta = 0.99 * P - project_to_psd(0.99 * P - Delta)
        Delta = project_to_psd(Delta) + 0.01 * jnp.eye(state_dim)
        new_vec_delta = _vec(Delta, state_dim)
        new_diff = jnp.linalg.norm(new_vec_delta-vec_delta) / state_dim ** 2
        return (new_vec_delta, new_diff)

    delta_init = jnp.zeros((state_dim, state_dim))
    vec_delta_init = _vec(delta_init, state_dim)
    diff_init = 1.
    val_init = (vec_delta_init, diff_init)
    out = lax.while_loop(lambda x: x[1]>tol, _step, val_init)
    return _mat(out[0], state_dim)

def sdp_opt2(state_dim, N, P, jacobian, hessian, alpha, tol=0.1):
    """
    original vanilla sdt_opt
    """
    tol = 0.1
    # construct 2nd order term
    vec_hessians = _matrices_to_vectors(hessian, state_dim)
    low_rank = jnp.zeros((state_dim**2, state_dim**2))
    for i in range(state_dim):
        low_rank += vec_hessians[i] * vec_hessians[i].T
    lhs = (1/4) * low_rank + jnp.eye(state_dim**2)
    aid = alpha * _vec(jacobian.T @ jacobian, state_dim) / N 

    # looping step
    def _step(val):
        vec_delta = val[0]
        rhs = aid + vec_delta
        new_vec_delta = jnp.linalg.solve(lhs, rhs)
        Delta = _mat(new_vec_delta, state_dim)
        Delta = project_to_psd(Delta)
        Delta = P - project_to_psd(P - Delta)
        Delta = project_to_psd(Delta)
        new_vec_delta = _vec(Delta, state_dim)
        new_diff = jnp.linalg.norm(new_vec_delta-vec_delta) / state_dim ** 2
        return (new_vec_delta, new_diff)

    delta_init = jnp.zeros((state_dim, state_dim))
    vec_delta_init = _vec(delta_init, state_dim)
    diff_init = 1.
    val_init = (vec_delta_init, diff_init)
    out = lax.while_loop(lambda x: x[1]>tol, _step, val_init)
    return _mat(out[0], state_dim)

def rmse(x_est, x_base):

    T = x_est.shape[0]
    sum_sq = jnp.sum((x_est - x_base) ** 2)

    return jnp.sqrt(sum_sq / T)

def resample(weights, num_samples):
    _flattened_weights = weights.flatten()
    M, N, L = weights.shape
    ind_mat = [[[(m, n, l) for l in range(L)] for n in range(N)] for m in range(M)]
    ind_arr = np.array(ind_mat)
    _flat_ind_mat = np.array(ind_arr).reshape((M * N * L, 3))
    sample_flat_ind = random.choice(np.arange(M * N * L), num_samples, p=_flattened_weights)
    return _flat_ind_mat[sample_flat_ind]

def retain(weights, num_retained):
    _flattened_weights = weights.flatten()
    M, N, L = weights.shape
    ind_mat = [[[(m, n, l) for l in range(L)] for n in range(N)] for m in range(M)]
    ind_arr = np.array(ind_mat)
    _flat_ind_mat = np.array(ind_arr).reshape((M * N * L, 3))
    sorted_ind = np.argsort(_flattened_weights)[-num_retained:]
    return _flat_ind_mat[sorted_ind]

def _resample(weights, particles, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, resampled_particles, next_key

def _resample2(weights, particles, key):                                                                  
    keys = jr.split(key, 2)
    num_particles = weights.shape[0]
    resampled_idx = jr.choice(keys[0], jnp.arange(weights.shape[0]), shape=(num_particles,), p=weights)
    resampled_particles = jnp.take(particles, resampled_idx, axis=0)
    weights = jnp.ones(shape=(num_particles,)) / num_particles
    next_key = keys[1]
    return weights, resampled_particles, next_key, resampled_idx

@partial(jit, static_argnums=(1,))
def optimal_resampling(weights, N, key):
    """Find the threshold for resampling particles using the optimal resampling algorithm of Fearnhead and Clifford (2003).
    Returns inidices of resampled particles and their weights.
    """
    # sort weights
    M = weights.shape[0]
    sorted_weights = jnp.sort(weights)
    sorted_idx = jnp.argsort(weights)

    # compute threshold p and value L of particles to retain
    lower_diag = jnp.triu(jnp.ones((M, M)), k=0).T
    ps = lax.dynamic_slice(lower_diag, (M-N, 0), (N-1, M)) @ sorted_weights / jnp.arange(1,N)
    ps = jnp.flip(ps)
    bounds = vmap(lambda ind: (sorted_weights[M-ind-1], sorted_weights[M-ind]))(jnp.arange(1, N))
    preds = vmap(lambda y,x,z : jnp.logical_and(y < x, x < z))(bounds[0], ps, bounds[1])
    L = jnp.where(preds, jnp.arange(1, N), 0).sum()
    p = jnp.where(L==0, 1/N, ps[L-1])
    
    # resample
    res_weights = jnp.where(sorted_weights < p, sorted_weights, 0.0)
    res_weights = res_weights / res_weights.sum()
    res_idx = jr.choice(key, M, shape=(M,), replace=True, p=res_weights)
    unsort_res_idx = sorted_idx[res_idx]

    final_idx = jnp.where(sorted_weights < p, unsort_res_idx, sorted_idx)
    final_weights = jnp.where(sorted_weights < p, p, sorted_weights)

    return final_idx[M-N:], final_weights[M-N:] / final_weights[M-N:].sum() 

# @partial(jit, static_argnums=(1,))
def get_sigma_points(m, P, ulambda):

    """Get sigma points for the unscented transform."""
    dx = m.shape[0]
    L = jnp.real(scipy.linalg.sqrtm(P))
    sigma_plus = jnp.stack([m]*dx, axis = 0) + jnp.sqrt(dx+ulambda)*L.T
    sigma_minus = jnp.stack([m]*dx, axis = 0) - jnp.sqrt(dx+ulambda)*L.T
    sigma_points = jnp.concatenate([sigma_plus, sigma_minus], axis=0)

    return sigma_points


def psd_solve(A,b):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = A + 1e-6
    return jnp.linalg.solve(A,b)    


def pcrlb(key, model, params, seq_length, nsim, inputs):
    # sample nsim state trajectories
    keys = jr.split(key, nsim)
    states, _ = vmap(model.sample, in_axes=(None, 0, None, None))(params, key, seq_length, inputs = inputs)
    return states


def bootstrap(key, rmse_array, B):
    N = rmse_array.shape[0]
    rmse_boot = jnp.zeros((B,))
    for b in range(B):
        key, subkey = jr.split(key)
        ind = jr.randint(subkey, (N,), 0, N)
        rmse_boot = rmse_boot.at[b].set(jnp.mean(rmse_array[ind]))
    return rmse_boot


def vec_clean_and_rmse(array, key, B, Nsim):
    success_pct = []
    rmse_array = []
    std_array = []
    keys = jr.split(key, array.shape[1])    
    for i in range(array.shape[1]):
        ind = jnp.argwhere(jnp.isnan(array[:,i]))
        clean_array = jnp.delete(array[:,i], ind)
        clean_array = bootstrap(keys[i], clean_array, B)
        rmse_array.append(jnp.mean(clean_array))
        std_array.append(jnp.std(clean_array))
        success_pct.append(int(100 - 100*len(ind) / Nsim))
    return success_pct, rmse_array, std_array


# # KL divergence between two Gaussians
# def kl_gaussian(mu1, sigma1, mu2, sigma2):

#     d = mu1.shape[-1]
#     sigma2_inv = jnp.linalg.inv(sigma2 @ sigma2.T)
#     sigma1_cov = sigma1 @ sigma1.T
#     diff = mu2 - mu1
#     trace_term = jnp.trace(sigma2_inv @ sigma1_cov)
#     mahalanobis = diff.T @ sigma2_inv @ diff
#     log_det_ratio = 2 * (jnp.linalg.slogdet(sigma2)[1] - jnp.linalg.slogdet(sigma1)[1])

#     return 0.5 * (trace_term + mahalanobis - d + log_det_ratio)


# # Variational KL approximation
# def d_var(f_params, g_params):

#     '''
#     Computes the variational KL divergence D_var between two Gaussian mixtures (only the 
#     term that depends on the parameters of the second mixture).
#     '''

#     pi, mu_f, sigma_f = f_params
#     w, mu_g, sigma_g = g_params

#     def inner(a):

#         def log_sum(b):

#             D_ab = kl_gaussian(mu_f[a], sigma_f[a], mu_g[b], sigma_g[b])

#             return jnp.log(w[b] + 1e-8) - D_ab

#         logsumexp = jax.scipy.special.logsumexp(jax.vmap(log_sum)(jnp.arange(len(w))))

#         return pi[a] * logsumexp

#     return jnp.sum(jax.vmap(inner)(jnp.arange(len(pi))))


def reduce_gmm_via_subselection(key, gmm, B):

    weights, means, covs_or_chol = gmm
    # if method == 'top':
    indices = jnp.argsort(weights)[-B:][::-1]
    # elif method == 'random':
    # indices = jax.random.choice(key, len(weights), shape=(B,), replace=False)
    # else:
        # raise ValueError("method must be 'top' or 'random'")
    
    new_weights = weights[indices]
    new_weights /= jnp.sum(new_weights)
    new_means = means[indices]
    new_covs_or_chol = covs_or_chol[indices]

    return new_weights, new_means, new_covs_or_chol


# def refit_gmm(key, gmm_init, B):

#     target_gmm_params = reduce_gmm_via_subselection(key, gmm_init, B)
#     g_flat, g_unravel = jax.flatten_util.ravel_pytree(target_gmm_params)

#     def loss_fn(g_flat):

#         g_params = g_unravel(g_flat)

#         return -d_var(gmm_init, g_params)  # Minimize negative D_var

#     opt = optax.adam(1e-2)
#     opt_state = opt.init(g_flat)

#     @jax.jit
#     def step(g_flat, opt_state):

#         loss, grads = jax.value_and_grad(loss_fn)(g_flat)
#         updates, opt_state = opt.update(grads, opt_state)
#         g_flat = optax.apply_updates(g_flat, updates)

#         return g_flat, opt_state, loss

#     def scan_step(carry, _):

#         g_flat, opt_state = carry
#         g_flat, opt_state, loss = step(g_flat, opt_state)

#         return (g_flat, opt_state), loss

#     (g_flat, opt_state), losses = jax.lax.scan(
#         scan_step,
#         (g_flat, opt_state),
#         None,
#         length=100
#     )

#     weights, means, covs = g_unravel(g_flat)
#     weights = weights / weights.sum()  

#     return (weights, means, covs)

import tensorflow_probability.substrates.jax.bijectors as tfb #type: ignore

real_to_chol = tfb.Chain([
    tfb.TransformDiagonal(tfb.Exp()),
    tfb.FillTriangular(),
])


chol_to_real = tfb.Chain([
    tfb.Invert(tfb.FillTriangular()),
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
])

# KL divergence between Gaussians
def kl_gaussian(mu1, L1, mu2, L2):
    d = mu1.shape[0]
    Sigma1 = L1 @ L1.T
    Sigma2 = L2 @ L2.T
    inv_Sigma2 = jnp.linalg.inv(Sigma2)
    diff = mu2 - mu1
    tr = jnp.trace(inv_Sigma2 @ Sigma1)
    mahal = diff.T @ inv_Sigma2 @ diff
    logdet = 2 * (jnp.linalg.slogdet(L2)[1] - jnp.linalg.slogdet(L1)[1])
    return 0.5 * (tr + mahal - d + logdet)

# Variational KL lower bound
def d_var(f_params, g_params_vec):

    pi, mu_f, L_f = f_params
    w, mu_g, L_g_vec = g_params_vec
    L_g = jax.vmap(real_to_chol.forward)(L_g_vec)

    def component_term(i):

        def kl_term(j):

            return jnp.log(w[j] + 1e-8) - kl_gaussian(mu_f[i], L_f[i], mu_g[j], L_g[j])

        return pi[i] * jax.scipy.special.logsumexp(jax.vmap(kl_term)(jnp.arange(w.shape[0])))

    return jnp.sum(jax.vmap(component_term)(jnp.arange(pi.shape[0])))

# Optimization wrapper
def refit_gmm(key, given_gmm, B):

    f_w, f_mu, f_cov = given_gmm
    f_L = jax.vmap(jnp.linalg.cholesky)(f_cov)
    f_params = (f_w, f_mu, f_L)

    # Target GMM g
    g_w, g_mu, g_L = reduce_gmm_via_subselection(key, given_gmm, B)
    g_Lvec = jax.vmap(chol_to_real.forward)(g_L)
    g_params_init = (g_w, g_mu, g_Lvec)
    flat_g, g_unravel = jax.flatten_util.ravel_pytree(g_params_init)

    def loss_fn(flat_g):
        
        g_params = g_unravel(flat_g)
        
        return -d_var(f_params, g_params)

    optimizer = optax.adam(1e-2)
    state = optimizer.init(flat_g)

    @jax.jit
    def step(g_flat, opt_state):

        loss, grad = jax.value_and_grad(loss_fn)(g_flat)
        updates, opt_state = optimizer.update(grad, opt_state)
        g_flat = optax.apply_updates(g_flat, updates)

        return g_flat, opt_state, loss

    def scan_step(carry, _):
        g_flat, opt_state = carry
        g_flat, opt_state, loss = step(g_flat, opt_state)
        return (g_flat, opt_state), -loss

    (flat_g, state), _ = lax.scan(scan_step, (flat_g, state), None, length=100)

    g_W, g_M, g_L_vec = g_unravel(flat_g)
    g_L = jax.vmap(real_to_chol.forward)(g_L_vec)
    g_C = jax.vmap(lambda L: L @ L.T)(g_L)  
    g_W = g_W / jnp.sum(g_W) 
    g_params = (g_W, g_M, g_C)

    return g_params
