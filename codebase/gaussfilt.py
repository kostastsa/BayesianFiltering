import numpy as np
from numpy import random
from jax import grad, jit, vmap
from jax import  jacfwd, jacrev

import utils


class SSM:

    def __init__(self, dx, dy, c, Q, d, R, f=None, g=None):
        self.dx = dx
        self.dy = dy
        self.f = f
        self.g = g
        self.Q = Q
        self.R = R
        self.c = c
        self.d = d

    def simulate(self, T, x0):
        """
        :param T:
        :param init_state:
        :return:
        """
        self.T = T
        xs = np.zeros([T, self.dx])
        ys = np.zeros([T, self.dy])
        old_x = x0
        for t in range(T):
            new_x, new_y = self.propagate(old_x)
            xs[t, :] = new_x
            ys[t, :] = new_y
            old_x = new_x
        return xs, ys

    def propagate(self, old_x):
        """
        :param old_x:
        :return:
        """
        new_x = self.f(old_x) + random.multivariate_normal(self.c, self.Q)
        new_y = self.g(new_x) + random.multivariate_normal(self.d, self.R)
        return new_x, new_y


class GaussFilt:
    """
    The GaussFilt class is the parent class of all Gaussian filtering subclasses that follow. Each subclass corresponds
    to a filtering algorithm, respectively the Unscented Kalman filter (UKF) the Monte Carlo filter (MCF) the Extended
    Kalman filter (EKF) and the oMCLA filter.
    The parent class is defined in terms of a corresponding SSM that is used to define the implicit model, used by the
    filter. The moment_approx method is used to make the moment approximations in each filter subclass, the run method is
    shared  by all Gaussian filters and is a recursive application of the moment_approx method for prediction and update.
    """

    def __init__(self, ssm):
        self.f = ssm.f
        self.g = ssm.g
        self.Q = ssm.Q
        self.R = ssm.R
        self.dx = ssm.dx
        self.dy = ssm.dy

    def moment_approx(self, m, P, kw):
        raise NotImplementedError

    def run(self, ys, m0, P0):
        # Initialize arrays
        seq_length = np.shape(ys)[0]
        filtered_means = np.zeros((seq_length + 1, self.dx))
        filtered_covs = np.zeros((seq_length + 1, self.dx, self.dx))
        predicted_means = np.zeros((seq_length, self.dx))
        predicted_covs = np.zeros((seq_length, self.dx, self.dx))
        ll = np.zeros(seq_length)

        # Initial conditions in last entry of the array (seq_length)
        filtered_means[seq_length] = m0
        filtered_covs[seq_length] = P0

        for t in range(seq_length):
            #print('t:', t)
            # Prediction
            predicted_means[t], predicted_covs[t] = self.moment_approx(filtered_means[t - 1], filtered_covs[t - 1],
                                                                       'pred')[0:2]
            # Update
            mu_y, Sy, Cxy = self.moment_approx(predicted_means[t], predicted_covs[t], 'upd')
            gain_matrix = Cxy @ np.linalg.inv(Sy)  # TODO: replace inv with more efficient implementation
            filtered_means[t] = predicted_means[t] + (mu_y - ys[t]) @ gain_matrix.transpose()
            filtered_covs[t] = predicted_covs[t] - gain_matrix @ Sy @ gain_matrix.transpose()
            ll[t] = utils.gaussian_logpdf(np.reshape(ys[t], [1, self.dy]), mu_y, Sy)
        return ll, filtered_means, filtered_covs

    def which_step(self, kw):
        if kw == 'pred':
            func = self.f
            cov = self.Q
            dim_in = self.dx
            dim_out = self.dx
        if kw == 'upd':
            func = self.g
            cov = self.R
            dim_in = self.dx
            dim_out = self.dy
        return func, cov, dim_in, dim_out


class UKF(GaussFilt):

    def __init__(self, ssm, alpha, beta, kappa):
        GaussFilt.__init__(self, ssm)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lamda = alpha ** 2 * (self.dx + kappa) - self.dx

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        sigma_points = utils.split_to_sigma_points(m, P, self.lamda)
        new_sigma_points = np.array(list(map(func, sigma_points)))
        new_sigma_points = np.squeeze(new_sigma_points) #squeeze fix
        mean_out = (self.lamda / (self.dx + self.lamda)) * new_sigma_points[0] + 1 / (
                    2 * (self.dx + self.lamda)) * np.sum(
            new_sigma_points[1:], axis=0)
        var_out = cov + (self.lamda / (self.dx + self.lamda) + 1 - self.alpha ** 2 + self.beta) * np.outer(
            new_sigma_points[0] - mean_out, new_sigma_points[0] - mean_out) \
                  + 1 / (2 * (self.dx + self.lamda)) * np.transpose(new_sigma_points[1:] - mean_out) @ \
                  (new_sigma_points[1:] - mean_out)
        cov_out = (self.lamda / (self.dx + self.lamda) + 1 - self.alpha ** 2 + self.beta) * np.squeeze(np.outer(
            np.transpose(sigma_points[0] - m),
            new_sigma_points[0] - mean_out)) + 1 / (2 * (self.dx + self.lamda)) * \
                  np.squeeze((sigma_points[1:] - m).T @ (new_sigma_points[1:] - mean_out)) #squeeze fix
        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])


class MCF(GaussFilt):

    def __init__(self, ssm, num_particles):
        GaussFilt.__init__(self, ssm)
        self.num_particles = num_particles

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        particles = random.multivariate_normal(m, P, self.num_particles)
        trans_particles = np.array(list(map(func, particles)))
        mean_out = np.sum(trans_particles, axis=0) / self.num_particles
        var_out = cov + (trans_particles-mean_out).transpose() @ (trans_particles-mean_out) / (self.num_particles-1)
        cov_out = (particles-m).transpose() @ (trans_particles-mean_out) / (self.num_particles-1)
        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])


class EKF(GaussFilt):

    def __init__(self, ssm, order=2):
        GaussFilt.__init__(self, ssm)
        self.f_jacobian = jacfwd(self.f)
        self.g_jacobian = jacfwd(self.g)
        if order == 2:
            self.f_hessian = jacfwd(jacrev(self.f))
            self.g_hessian = jacfwd(jacrev(self.g))
        else:
            self.f_hessian = np.array([0])
            self.g_hessian = np.array([0])

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        if kw == 'pred':
            jacobian = self.f_jacobian
            hessian = self.f_hessian
        if kw == 'upd':
            jacobian = self.g_jacobian
            hessian = self.g_hessian

        ## Check dimensions for trace ax (mean_out)
        if dim_in == 1 or dim_out == 1:
            ax1 = 0
            ax2 = 1
        else:
            ax1 = 1
            ax2 = 2

        H = hessian(m)
        if dim_out == 1:
            H = np.squeeze(H, axis=0)

        ## for debugging
        # print('1', )
        # print('2', )
        # print('3', )
        # print('4', )

        ## Compute moments
        mean_out = func(m) + (1 / 2) * np.trace(H @ P, axis1=ax1, axis2=ax2)
        var_out = cov + jacobian(m) @ P @ jacobian(m).T + (1 / 2) * \
                  np.trace(((H @ P).reshape(dim_in * dim_out, dim_in) @
                            (H @ P).reshape(dim_in * dim_out, dim_in).T)
                           .reshape(dim_in, dim_in * dim_out * dim_out).T
                           .reshape(dim_out * dim_out, dim_in, dim_in), axis1=1, axis2=2).reshape(dim_out, dim_out)
        cov_out = P @ jacobian(m).T

        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])




