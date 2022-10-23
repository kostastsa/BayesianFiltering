import numpy as np
from numpy import random
from jax import numpy as jnp
from jax import jacfwd, jacrev
import time
import pandas as pd
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
        assert self.f(old_x).shape[0] == self.dx
        assert self.g(old_x).shape[0] == self.dy

        for t in range(T):
            new_x, new_y = self.propagate(old_x)
            # print(ys[t, :].shape)
            # print(new_y.shape)
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
        self.time = 0.0

    def moment_approx(self, m, P, kw):
        raise NotImplementedError

    def _init_dataframe(self, seq_length):
        columns = ['t', 'means', 'covs', 'gain', 'mu_y', 'Sy']
        self.df = pd.DataFrame(columns=columns)
        self.df = pd.DataFrame()
        self.df['t'] = np.zeros(seq_length)
        self.df['means'] = np.zeros(seq_length)
        self.df['covs'] = np.zeros(seq_length)
        self.df['gain'] = np.zeros(seq_length)
        self.df['mu_y'] = np.zeros(seq_length)
        self.df['Sy'] = np.zeros(seq_length)

    def run(self, ys, m0, P0, verbose=False):
        tin = time.time()
        # Initialize arrays
        seq_length = np.shape(ys)[0]
        filtered_means = np.zeros((seq_length + 1, self.dx))
        filtered_covs = np.zeros((seq_length + 1, self.dx, self.dx))
        predicted_means = np.zeros((seq_length, self.dx))
        predicted_covs = np.zeros((seq_length, self.dx, self.dx))
        ll = np.zeros(seq_length)

        # for debugging
        if self.dx == 1:
            self._init_dataframe(seq_length)

        # Initial conditions in last entry of the array (seq_length)
        filtered_means[seq_length] = m0
        filtered_covs[seq_length] = P0

        for t in range(seq_length):
            if verbose:
                print('{}.run | t='.format(self), t)
            # Prediction
            predicted_means[t], predicted_covs[t] = self.moment_approx(filtered_means[t - 1], filtered_covs[t - 1],
                                                                       'pred')[0:2]
            # Update
            mu_y, Sy, Cxy = self.moment_approx(predicted_means[t], predicted_covs[t], 'upd')
            gain_matrix = Cxy @ np.linalg.inv(Sy)  # TODO: replace inv with more efficient implementation
            filtered_means[t] = predicted_means[t] + (ys[t] - mu_y) @ gain_matrix.T
            filtered_covs[t] = predicted_covs[t] - gain_matrix @ Sy @ gain_matrix.T
            ll[t] = utils.gaussian_logpdf(np.reshape(ys[t], [1, self.dy]), mu_y, Sy)

            # For numerical debugging
            if self.dx == 1:
                self.df['t'][t] = t
                self.df['means'][t] = filtered_means[t]
                self.df['covs'][t] = filtered_covs[t]
                self.df['gain'][t] = gain_matrix
                self.df['mu_y'][t] = mu_y
                self.df['Sy'][t] = Sy

        self.time = time.time() - tin
        return ll, filtered_means[0:seq_length], filtered_covs[0:seq_length]

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

    def __init__(self, ssm, alpha=1e-3, beta=2, kappa=0):
        GaussFilt.__init__(self, ssm)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lamda = alpha ** 2 * (self.dx + kappa) - self.dx

    def __str__(self):
        return 'UKF'

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        sigma_points = utils.split_to_sigma_points(m, P, self.lamda)
        new_sigma_points = np.array(list(map(func, sigma_points)))
        new_sigma_points = np.squeeze(new_sigma_points)  # squeeze fix
        mean_out = (self.lamda / (self.dx + self.lamda)) * new_sigma_points[0] + (1 / (
                2 * (self.dx + self.lamda))) * np.sum(
            new_sigma_points[1:], axis=0)
        var_out = cov + (self.lamda / (self.dx + self.lamda) + 1 - self.alpha ** 2 + self.beta) * np.outer(
            new_sigma_points[0] - mean_out, new_sigma_points[0] - mean_out) \
                  + 1 / (2 * (self.dx + self.lamda)) * (new_sigma_points[1:] - mean_out).T @ \
                  (new_sigma_points[1:] - mean_out)
        cov_out = (self.lamda / (self.dx + self.lamda) + 1 - self.alpha ** 2 + self.beta) * np.squeeze(np.outer(
            np.transpose(sigma_points[0] - m),
            new_sigma_points[0] - mean_out)) + 1 / (2 * (self.dx + self.lamda)) * \
                  np.squeeze((sigma_points[1:] - m).T @ (new_sigma_points[1:] - mean_out))  # squeeze fix
        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])


class MCF(GaussFilt):

    def __init__(self, ssm, num_particles):
        GaussFilt.__init__(self, ssm)
        self.num_particles = num_particles

    def __str__(self):
        return 'MCF'

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        particles = random.multivariate_normal(m, P, self.num_particles)
        trans_particles = np.array(list(map(func, particles)))
        mean_out = np.sum(trans_particles, axis=0) / self.num_particles
        var_out = cov + (trans_particles - mean_out).T @ (trans_particles - mean_out) / self.num_particles
        cov_out = (particles - m).T @ (trans_particles - mean_out) / self.num_particles

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
            self.f_hessian = lambda x: np.zeros((self.dx, self.dx, self.dx))
            self.g_hessian = lambda x: np.zeros((self.dy, self.dx, self.dx))

    def __str__(self):
        return 'EKF'

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        if kw == 'pred':
            jacobian = self.f_jacobian
            hessian = self.f_hessian
        if kw == 'upd':
            jacobian = self.g_jacobian
            hessian = self.g_hessian

        ## Check dimensions for trace ax (mean_out)
        if dim_in == 1:
            ax1 = 0
            ax2 = 1
        else:
            ax1 = 1
            ax2 = 2

        H = hessian(m)

        if dim_out == 1:
            H = np.squeeze(H, axis=0)

        ## Compute moments
        mean_out = func(m) + (1 / 2) * np.trace(np.reshape(H @ P, (dim_out, dim_in, dim_in)), axis1=ax1, axis2=ax2)
        var_out = cov + jacobian(m) @ P @ jacobian(m).T + (1 / 2) * \
                  np.trace(((H @ P).reshape(dim_in * dim_out, dim_in) @
                            (H @ P).reshape(dim_in * dim_out, dim_in).T)
                           .reshape(dim_in, dim_in * dim_out * dim_out).T
                           .reshape(dim_out * dim_out, dim_in, dim_in), axis1=1, axis2=2).reshape(dim_out, dim_out)
        cov_out = P @ jacobian(m).T

        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])


class MCLAF(GaussFilt):

    def __init__(self, ssm, num_particles):
        GaussFilt.__init__(self, ssm)
        self.num_particles = num_particles
        self.f_jacobian = jacfwd(self.f)
        self.g_jacobian = jacfwd(self.g)
        self.f_hessian = jacfwd(jacrev(self.f))
        self.g_hessian = jacfwd(jacrev(self.g))

    def __str__(self):
        return 'MCLAF'

    def moment_approx(self, m, P, kw):
        func, cov, dim_in, dim_out = self.which_step(kw)
        if kw == 'pred':
            jacobian = self.f_jacobian
            hessian = self.f_hessian
            lip = 1
        if kw == 'upd':
            jacobian = self.g_jacobian
            hessian = self.g_hessian
            lip = 1

        H = hessian(m)
        if dim_out == 1:
            H = jnp.squeeze(H, axis=0)
            avg_hessian = H
        else:
            avg_hessian = jnp.sum(H, axis=0)

        # Choose Delta
        Delta = utils.sdp_opt(self.dx, self.num_particles, lip, P, P, avg_hessian, 10, 0.01)

        # Sample latent particles + compute Jacobians at particles
        particles = random.multivariate_normal(m, P - Delta, self.num_particles)
        trans_particles = np.array(list(map(func, particles)))
        grads_at_particles = np.array(list(map(jacobian, particles)))

        # Compute moments
        mean_out = np.sum(trans_particles, axis=0) / self.num_particles
        var_out = cov + (trans_particles - mean_out).T @ (trans_particles - mean_out) / self.num_particles \
                  + grads_at_particles.reshape(self.num_particles * dim_out, dim_in).reshape(dim_out,
                                                                                             self.num_particles * dim_in) @ \
                  (grads_at_particles @ Delta) \
                      .reshape(self.num_particles * dim_out, dim_in) \
                      .reshape(dim_out, self.num_particles * dim_in).T / self.num_particles
        cov_out = (particles - m).T @ (trans_particles - mean_out) / self.num_particles \
                  + Delta @ np.sum(grads_at_particles, 0).T / self.num_particles
        # TODO: check the normalizations (1/N) vs (1/N-1), also the .T in cov_out, second term

        return np.reshape(mean_out, [1, dim_out]), \
               np.reshape(var_out, [dim_out, dim_out]), \
               np.reshape(cov_out, [dim_in, dim_out])


class GaussSumFilt():

    def __init__(self, gauss_filt, num_models):
        # Initialize GaussFilt objects
        self.M = num_models
        self.dx = gauss_filt.dx
        self.dy = gauss_filt.dy
        self.num_models = num_models
        self.gf = gauss_filt
        self.time = 0.0

    def __str__(self):
        return 'gf.GSF'

    def run(self, ys, m0, P0, verbose=False):
        tin = time.time()
        seq_length = np.shape(ys)[0]
        filtered_component_means = np.zeros((seq_length + 1, self.dx, self.M))
        filtered_component_covs = np.zeros((seq_length + 1, self.dx, self.dx, self.M))
        component_weights = np.zeros((seq_length + 1, self.M))

        # Initial conditions in last entry of the array (seq_length)
        component_weights[seq_length] = np.ones(self.M) / self.M

        for m in range(self.M):
            filtered_component_means[seq_length, :, m] = m0 + random.multivariate_normal(np.zeros(self.dx),
                                                                                         np.eye(self.dx))
            filtered_component_covs[seq_length, :, :, m] = P0

        for t in range(seq_length):
            if verbose:
                print('{}.run | t='.format(self), t)
            for m in range(self.M):
                # prediction
                mean = filtered_component_means[t - 1, :, m]
                cov = filtered_component_covs[t - 1, :, :, m]

                mean_pred, cov_pred = self.gf.moment_approx(mean, cov, 'pred')[0:2]
                # Update
                mu_y, Sy, Cxy = self.gf.moment_approx(mean_pred, cov_pred, 'upd')
                gain_matrix = Cxy @ np.linalg.inv(Sy)  # TODO: replace inv with more efficient implementation
                filtered_component_means[t, :, m] = mean_pred + (ys[t] - mu_y) @ gain_matrix.T
                filtered_component_covs[t, :, :, m] = cov_pred - gain_matrix @ Sy @ gain_matrix.T
                loglik = utils.gaussian_logpdf(np.reshape(ys[t], [1, self.dy]), mu_y, Sy)
                component_weights[t, m] = np.exp(loglik) * component_weights[t-1, m]
            component_weights[t] /= np.sum(component_weights[t])
        self.time = time.time() - tin

        return filtered_component_means, filtered_component_covs, component_weights

