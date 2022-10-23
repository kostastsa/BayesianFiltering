import numpy as np
from numpy import random
from jax import numpy as jnp
from jax import jacfwd, jacrev
from scipy.stats import multivariate_normal
import time
import pandas as pd
import utils
import gaussfilt as gf

class GaussSumFilt:

    def __init__(self, ssm, M):
        self.f = ssm.f
        self.g = ssm.g
        self.Q = ssm.Q
        self.R = ssm.R
        self.dx = ssm.dx
        self.dy = ssm.dy
        self.M = M
        self.f_jacobian = jacfwd(self.f)
        self.f_hessian = jacfwd(jacrev(self.f))
        self.g_jacobian = jacfwd(self.g)
        self.g_hessian = jacfwd(jacrev(self.g))
        self.time = 0.0

    def __str__(self):
        return 'GSF'

    def run(self, ys, m0, P0, verbose = False):
        tin = time.time()
        # Initialize arrays
        seq_length = np.shape(ys)[0]
        filtered_component_means = np.zeros((seq_length + 1, self.dx, self.M))
        filtered_component_covs = np.zeros((seq_length + 1, self.dx, self.dx, self.M))
        point_est = np.zeros((seq_length, self.dx))
        component_weights = np.zeros((seq_length + 1, self.M))

        predicted_component_means = np.zeros((self.M, self.dx))
        predicted_component_covs = np.zeros((self.M, self.dx, self.dx))

        # Initial conditions in last entry of the array (seq_length)
        component_weights[seq_length] = np.ones(self.M) / self.M

        for m in range(self.M):
            filtered_component_means[seq_length, :, m] = m0 + random.multivariate_normal(np.zeros(self.dx), np.eye(self.dx))
            filtered_component_covs[seq_length, :, :, m] = P0

        for t in range(seq_length):
            if verbose:
                print('{}.run | t='.format(self), t)
            for m in range(self.M):
                # prediction
                mean = filtered_component_means[t - 1, :, m]
                cov = filtered_component_covs[t - 1, :, :, m]
                predicted_component_means[m] = self.f(mean)
                predicted_component_covs[m] = cov + self.f_jacobian(mean) @ cov @ self.f_jacobian(mean).T

                # update
                mean = predicted_component_means[m]
                cov = predicted_component_covs[m]
                mu_y = self.g(mean)
                Sy = self.R + self.g_jacobian(mean) @ cov @ self.g_jacobian(mean).T
                Cxy = cov @ self.g_jacobian(mean).T
                gain_matrix = Cxy @ np.linalg.inv(Sy)  # TODO: replace inv with more efficient implementation
                filtered_component_means[t, :, m] = mean + (ys[t] - mu_y) @ gain_matrix.T
                filtered_component_covs[t, :, :, m] = cov - gain_matrix @ Sy @ gain_matrix.T
                loglik = utils.gaussian_logpdf(np.reshape(ys[t], [1, self.dy]), mu_y, Sy)
                component_weights[t, m] = np.exp(loglik) * component_weights[t - 1, m]
            component_weights[t] /= np.sum(component_weights[t])
            point_est[t] = np.sum(np.multiply(filtered_component_means[t, :], component_weights[t]))
        self.time = time.time() - tin
        return filtered_component_means, filtered_component_covs, component_weights, point_est


class AugGaussSumFilt:

    def __init__(self, ssm, M, N, L):
        self.f = ssm.f
        self.g = ssm.g
        self.Q = ssm.Q
        self.R = ssm.R
        self.dx = ssm.dx
        self.dy = ssm.dy
        self.M = M
        self.N = N
        self.L = L
        self.f_jacobian = jacfwd(self.f)
        self.f_hessian = jacfwd(jacrev(self.f))
        self.g_jacobian = jacfwd(self.g)
        self.g_hessian = jacfwd(jacrev(self.g))
        self.set = False
        self.lf = 'auto'
        self.df = 'auto'
        self.time = 0.0

    def __str__(self):
        return 'AGSF'

    def set_aug_selection_params(self, *args, **selection_mode):
        if list(selection_mode.values())[0] == 'prop':
            self.aug_param_select_pred = 'prop'
            self.prop_pred = args[0]
        elif list(selection_mode.values())[0] == 'opt_lip':
            self.aug_param_select_pred = 'opt_lip'
            self.lip_pred = args[0]
        elif list(selection_mode.values())[0] == 'opt_max_grad':
            self.aug_param_select_pred = 'opt_max_grad'
            self.lip_pred_fac = args[0]
        elif list(selection_mode.values())[0] == 'input':
            self.aug_param_select_pred = 'input'
            self.Delta = args[0]



        if list(selection_mode.values())[1] == 'prop':
            self.aug_param_select_upd = 'prop'
            self.prop_upd = args[1]
        elif list(selection_mode.values())[1] == 'opt_lip':
            self.aug_param_select_upd = 'opt_lip'
            self.lip_upd = args[1]
        elif list(selection_mode.values())[1] == 'opt_max_grad':
            self.aug_param_select_upd = 'opt_max_grad'
            self.lip_upd_fac = args[1]
        elif list(selection_mode.values())[1] == 'input':
            self.aug_param_select_upd = 'input'
            self.Lambda = args[1]

    def run(self, ys, m0, P0, verbose = False):
        tin = time.time()
        # Initialize arrays
        seq_length = np.shape(ys)[0]
        filtered_component_means = np.zeros((seq_length + 1, self.dx, self.M))
        filtered_component_covs = np.zeros((seq_length + 1, self.dx, self.dx, self.M))
        point_est = np.zeros((seq_length, self.dx))
        component_weights = np.zeros((seq_length + 1, self.M))
        _interm_weights = np.zeros((self.M, self.N, self.L))

        predicted_component_means = np.zeros((self.M, self.N, self.dx))
        predicted_component_covs = np.zeros((self.M, self.N, self.dx, self.dx))

        _filtered_component_means = np.zeros((self.M, self.N, self.L, self.dx))
        _filtered_component_covs = np.zeros((self.M, self.N, self.L, self.dx, self.dx))

        # Initial conditions in last entry of the array (seq_length)
        component_weights[seq_length] = np.ones(self.M) / self.M
        for m in range(self.M):
            filtered_component_means[seq_length, :, m] = m0
            filtered_component_covs[seq_length, :, :, m] = P0
        max_grad_u = 1
        max_grad_p = 1
        for t in range(seq_length):
            if verbose:
                print('{}.run | t='.format(self), t)
            # prediction
            for m in range(self.M):
                mean = filtered_component_means[t - 1, :, m]
                cov = filtered_component_covs[t - 1, :, :, m]
                H = self.f_hessian(mean)
                if self.dx == 1:
                    H = jnp.squeeze(H, axis=0)
                    avg_hessian = H
                else:
                    avg_hessian = jnp.sum(H, axis=0)

                if self.aug_param_select_pred == 'prop':
                    self.Delta = self.prop_pred * cov
                elif self.aug_param_select_pred == 'opt_lip':
                    self.Delta = utils.sdp_opt(self.dx, self.N, self.lip_pred, cov, cov, avg_hessian, 10, 0.01)
                elif self.aug_param_select_pred == 'opt_max_grad':
                    self.Delta = utils.sdp_opt(self.dx, self.N, self.lip_pred_fac * max_grad_p, cov, cov, avg_hessian, 10, 0.01)
                elif self.aug_param_select_pred == 'input':
                    self.Delta = self.Delta if self.Delta < cov else cov
                    self.Delta = np.array(self.Delta).reshape(self.dx, self.dx)

                # Sample latent particles + compute Jacobians at particles
                _particles_to_predict = random.multivariate_normal(mean, cov - self.Delta, self.N)
                predicted_component_means[m] = np.array(list(map(self.f, _particles_to_predict)))
                _grads_at_particles = np.array(list(map(self.f_jacobian, _particles_to_predict)))
                max_grad_p = np.abs(np.max(_grads_at_particles).squeeze())
                for n in range(self.N):  # TODO: get rid of this for loop
                    predicted_component_covs[m, n] = _grads_at_particles[n] @ self.Delta @ _grads_at_particles[n].T + self.Q

            # update
            for m in range(self.M):
                for n in range(self.N):
                    mean = predicted_component_means[m, n]
                    cov = predicted_component_covs[m, n]
                    H = self.g_hessian(mean)
                    if self.dx == 1:
                        H = jnp.squeeze(H, axis=0)
                        avg_hessian = H
                    else:
                        avg_hessian = jnp.sum(H, axis=0)

                    if self.aug_param_select_upd == 'prop':
                        self.Lambda = self.prop_upd * cov
                    elif self.aug_param_select_upd == 'opt_lip':
                        self.Lambda = utils.sdp_opt(self.dx, self.L, self.lip_upd, cov, cov, avg_hessian, 10, 0.01)
                    elif self.aug_param_select_upd == 'opt_max_grad':
                        self.Lambda = utils.sdp_opt(self.dx, self.L, self.lip_upd_fac * max_grad_u, cov, cov, avg_hessian, 10, 0.01)
                    elif self.aug_param_select_upd == 'input':
                        self.Lambda = self.Lambda if self.Lambda < cov else cov
                        self.Lambda = np.array(self.Lambda).reshape(self.dx, self.dx)

                    # Sample latent particles + compute Jacobians at particles
                    _particles_to_update = random.multivariate_normal(mean, cov - self.Lambda, self.L)
                    y_means = np.array(list(map(self.g, _particles_to_update)))
                    _grads_at_particles = np.array(list(map(self.g_jacobian, _particles_to_update)))
                    max_grad_u = np.abs(np.max(_grads_at_particles).squeeze())
                    for l in range(self.L):  # TODO: get rid of this for loop
                        y_cov = _grads_at_particles[l] @ self.Lambda @ _grads_at_particles[l].T + self.R
                        Cxy = self.Lambda @ _grads_at_particles[l].T
                        gain_matrix = Cxy @ np.linalg.inv(y_cov)  # TODO: replace inv with more efficient implementation
                        _filtered_component_means[m, n, l] = predicted_component_means[m, n] + (
                                ys[t] - y_means[l]) @ gain_matrix.T
                        _filtered_component_covs[m, n, l] = predicted_component_covs[
                                                                m, n] - gain_matrix @ y_cov @ gain_matrix.T
                        _interm_weights[m, n, l] = component_weights[t - 1, m] * \
                                                   multivariate_normal.pdf(np.reshape(ys[t], [1, self.dy]),
                                                                           mean=y_means[l],
                                                                           cov=y_cov)

            _interm_weights /= np.sum(_interm_weights)

            # Resampling
            resampled_indices = utils.resample(_interm_weights, self.M)
            for m in range(self.M):
                filtered_component_means[t, :, m] = _filtered_component_means[tuple(resampled_indices[m])]
                filtered_component_covs[t, :, :, m] = _filtered_component_covs[tuple(resampled_indices[m])]
            component_weights[t] = component_weights[t-1]
            point_est[t] = np.sum(filtered_component_means[t, :, :], 1) / self.M
        self.time = time.time() - tin
        return filtered_component_means, filtered_component_covs, point_est


