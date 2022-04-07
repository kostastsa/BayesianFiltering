import numpy as np
import scipy.stats as stats


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
        new_means = np.random.multivariate_normal(mean, cov, num_comp)
    return new_means

def split_to_sigma_points(mean, cov, alpha, kappa):
    dx = np.shape(mean)[0]
    lam = alpha ** 2 * (dx + kappa) - dx
    sigma_points = np.zeros((2 * dx + 1, dx))
    sigma_points[0] = mean
    if dx == 1:
        sqrtCov = np.sqrt(cov)
        sigma_points[1] = mean + np.sqrt(dx + lam) * sqrtCov
        sigma_points[2] = mean - np.sqrt(dx + lam) * sqrtCov
    else:
        sqrtCov = np.linalg.cholesky(cov)
        for i in range(dx):
            sigma_points[i+1] = mean + np.sqrt(dx + lam) * sqrtCov.T[i]
            sigma_points[dx+i+1] = mean - np.sqrt(dx + lam) * sqrtCov.T[i]
    return sigma_points

def gm(x, means, sigma, num_comp):
    out = 0
    for mean in means:
        out += stats.norm.pdf(x, mean, sigma) / num_comp
    return out

