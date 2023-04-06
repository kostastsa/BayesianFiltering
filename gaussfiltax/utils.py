from numpy import random
import numpy as np
import scipy.stats as stats
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial


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
    Omega = np.linalg.inv(cov2);
    KL = np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) - d + \
         np.matmul(np.matmul(np.transpose(mean1 - mean2), Omega), (mean1 - mean2)) + np.trace(Omega * cov1);
    return KL / 2;


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

def sdp_opt(state_dim, P, hessian, alpha, tol=0.1):
    tol = 0.1
    # construct 2nd order term
    vec_hessians = _matrices_to_vectors(hessian, state_dim)
    low_rank = jnp.zeros((state_dim**2, state_dim**2))
    for i in range(state_dim):
        low_rank += vec_hessians[i] * vec_hessians[i].T
    lhs = low_rank + jnp.eye(state_dim**2)
    aid = alpha * _vec(jnp.eye(state_dim), state_dim)

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
    return out[0]

def mse(x_est, x_base):
    T = x_est.shape[0]
    sum_sq = np.sum((x_est - x_base) ** 2)
    return sum_sq / T


def rmse(x_est, x_base):
    T = x_est.shape[0]
    sum_sq = np.sum((x_est - x_base) ** 2)
    return np.sqrt(sum_sq / T)


def W_distance(means, covs, particles, weights):
    dist = 0.0
    N = means.shape[0]
    num_prt = particles.shape[0]
    for n in range(N):
        for i in range(num_prt):
            dist += weights[n] * (covs[n] + (means[n]-particles[i])**2)
    return dist / num_prt


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






