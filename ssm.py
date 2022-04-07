import numpy as np
from scipy import stats as st
import time

import utils


class LinearModelParameters:
    """

    """

    def __init__(self, A, H, Q, R):
        """

        :param A:
        :param H:
        :param Q:
        :param R:
        """
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def __str__(self):
        return 'A : \n' + str(self.A) + '\n' + 'H : \n' + str(self.H) + '\n' + 'Q : \n' + str(
            self.Q) + '\n' + 'R : \n' + str(self.R)


class StateSpaceModel:
    """
    Generative Model class.
    
    Attributes
    ----------
    """

    def __init__(self, dx, dy, f=None, g=None, params=None, descr=None):
        """Initialization method.

        :param dx:
        :param dy:
        :param f:
        :param g:
        :param descr:
        """
        self.dx = dx
        self.dy = dy
        self.f = f
        self.g = g
        self.descr = descr

    def __str__(self):
        if self.descr == "LG":
            return str(self.params)

    def simulate(self, T, init_state):
        """
        :param T:
        :param init_state:
        :return:
        """
        self.T = T
        states = np.zeros([T, self.dx])
        observs = np.zeros([T, self.dy])
        prev_state = init_state
        for t in range(T):
            new_state = self.f(prev_state)
            new_obs = self.g(new_state)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return states, observs

    def propagate(self, prev_state):
        """
        :param prev_state:
        :return:
        """
        if self.dx == 1:
            new_state = self.f(prev_state) + np.random.normal(0, self.params.Q)
        else:
            new_state = self.f(prev_state) + np.random.multivariate_normal(np.zeros([self.dx]), self.params.Q)
        if self.dy == 1:
            new_obs = self.g(new_state) + np.random.normal(0, self.params.R)
        else:
            new_obs = self.g(new_state) + np.random.multivariate_normal(np.zeros([self.dy]), self.params.R)
        return new_state, new_obs

    def extended_kalman_step(self, jacob_dyn, jacob_obs, new_obs, mean_prev, cov_prev, params):
        """

         :param new_obs:
         :param mean_prev:
         :param cov_prev:
         :return:
         """
        global mean_new, cov_new, lf
        params.A = jacob_dyn(mean_prev)
        if self.dx == 1:
            m_ = self.f(mean_prev)
            P_ = params.A * cov_prev * params.A + params.Q
            params.H = jacob_obs(m_)
            if np.isnan(new_obs).any():
                lf = np.nan
                return m_, P_, lf
            if self.dy == 1:
                v = new_obs - self.g(m_)
                S = params.H * P_ * params.H + params.R
                K = P_ * params.H / S

                mean_new = m_ + K * v
                cov_new = P_ - K * S * K
                lf = st.norm(m_ * params.H, S).pdf(new_obs)
            else:
                v = new_obs - self.g(m_)
                S = P_ * np.outer(params.H, params.H.T) + params.R
                K = P_ * np.matmul(params.H.T, np.linalg.inv(S))

                mean_new = m_ + np.dot(K, v.T)
                cov_new = P_ - np.dot(np.matmul(K, S), K.T)
                lf = st.multivariate_normal(np.squeeze(params.H.T * m_), S).pdf(new_obs)
        else:
            m_ = self.f(mean_prev)
            P_ = np.matmul(np.matmul(params.A, cov_prev), params.A.T) + params.Q
            params.H = jacob_obs(m_)
            if np.isnan(new_obs).any():
                lf = np.nan
                return m_, P_, lf
            v = new_obs - np.matmul(m_, params.H.T)
            S = np.matmul(np.matmul(params.H, P_), params.H.T) + params.R
            K = np.matmul(np.matmul(P_, params.H.T), np.linalg.inv(S))

            mean_new = m_ + np.matmul(v, K.T)
            cov_new = P_ - np.matmul(np.matmul(K, S), K)
            if self.dy == 1:
                lf = st.norm(np.matmul(m_, params.H.T), S).pdf(new_obs)
            else:
                lf = st.multivariate_normal(np.matmul(m_, params.H.T), S).pdf(new_obs)
        return mean_new, cov_new, lf

    def extended_kalman_filter(self, observs, jacob_dyn, jacob_obs, params, init):
        tin = time.time()
        T = np.shape(observs)[0]
        means = np.zeros([T, self.dx])
        covs = np.zeros([T, self.dx, self.dx])
        means[0] = init[0]
        covs[0] = init[1]
        for t in range(T - 1):
            means[t + 1], covs[t + 1], lf = self.extended_kalman_step(jacob_dyn, jacob_obs, observs[t], means[t],
                                                                      covs[t], params)
        print('EKF:',time.time() - tin)
        return means, covs

    def latent_ekf(self, observs, num_comp, latent_cov, jacob_dyn, jacob_obs, params, init):
        tin = time.time()
        T = np.shape(observs)[0]
        means = np.zeros([T, self.dx])
        covs = np.zeros([T, self.dx, self.dx])
        prop_means = np.zeros((num_comp, self.dx))
        prop_covs = np.zeros((num_comp, self.dx, self.dx))
        means[0] = init[0]
        covs[0] = init[1]
        for t in range(T - 1):
            split_means = utils.split_by_sampling(means[t], covs[t], latent_cov, num_comp)
            for i in range(num_comp):
                prop_means[i], prop_covs[i], lf = self.extended_kalman_step(jacob_dyn, jacob_obs, observs[t],
                                                                            split_means[i],
                                                                            latent_cov, params)
            means[t+1], covs[t+1] = utils.collapse(prop_means, prop_covs, np.ones(num_comp)/num_comp)
        print('LEKF:', time.time() - tin)
        return means, covs

    def unscented_kalman_filter(self, observs, init, params, alpha, beta, kappa):
        tin = time.time()
        T = np.shape(observs)[0]
        lam = alpha**2 * (self.dx + kappa) - self.dx
        means = np.zeros((T, self.dx))
        covs = np.zeros((T, self.dx, self.dx))
        pred_sigma_points = np.zeros((2 * self.dx + 1, self.dx))
        upd_sigma_points = np.zeros((2 * self.dx + 1, self.dy))
        means[0] = init[0]
        covs[0] = init[1]
        for t in range(1, T):
            #Prediction
            sigma_points = utils.split_to_sigma_points(means[t-1], covs[t-1], alpha, kappa)
            i = 0
            for point in sigma_points:
                pred_sigma_points[i] = self.f(point)
                i += 1
            W0m = (lam / (lam + self.dx))
            Wi = 1 / (2 * (lam + self.dx))
            W0c = W0m + 1 - alpha**2 + beta
            pred_mean = W0m * pred_sigma_points[0]
            pred_mean += Wi * np.mean(pred_sigma_points[1:], axis=0)
            if self.dx == 1:
                pred_cov = W0c * (pred_sigma_points[0] - pred_mean) ** 2 + params.Q
                for i in range(1, 2 * self.dx):
                    pred_cov += Wi * (pred_sigma_points[i] - pred_mean) ** 2
            else:
                pred_cov = W0c * np.outer(pred_sigma_points[0] - pred_mean, pred_sigma_points[0] - pred_mean) + params.Q
                for i in range(1, 2*self.dx):
                    pred_cov += Wi * np.outer(pred_sigma_points[i] - pred_mean, pred_sigma_points[i] - pred_mean)
            # Update
            i = 0
            for point in pred_sigma_points:
                upd_sigma_points[i] = self.g(point)
                i += 1
            obs_mean = W0m * upd_sigma_points[0]
            obs_mean += Wi * np.mean(upd_sigma_points[1:], axis=0)
            if self.dx == 1:
                S_matrix = W0c * (upd_sigma_points[0] - obs_mean) ** 2 + params.R
                C_matrix = W0c * (upd_sigma_points[0] - obs_mean) ** 2
                for i in range(1, 2 * self.dx):
                    S_matrix += Wi * (upd_sigma_points[i] - obs_mean) ** 2
                    C_matrix += Wi * (upd_sigma_points[i] - obs_mean) ** 2

                K_gain = C_matrix / S_matrix
                means[t] = pred_mean + K_gain * (observs[t] - obs_mean)
                covs[t] = pred_cov - K_gain * S_matrix * K_gain.T
            else:
                S_matrix = W0c * np.outer(upd_sigma_points[0] - obs_mean, upd_sigma_points[0] - obs_mean) + params.R
                C_matrix = W0c * np.outer(pred_sigma_points[0] - pred_mean, upd_sigma_points[0] - obs_mean)
                for i in range(1, 2 * self.dx):
                    S_matrix += Wi * np.outer(upd_sigma_points[i] - obs_mean, upd_sigma_points[i] - obs_mean)
                    C_matrix += Wi * np.outer(pred_sigma_points[i] - pred_mean, upd_sigma_points[i] - obs_mean)
                K_gain = np.matmul(C_matrix, np.linalg.inv(S_matrix))
                means[t] = pred_mean + np.matmul(K_gain, (observs[t] - obs_mean))
                covs[t] = pred_cov - np.matmul(np.matmul(K_gain, S_matrix), K_gain.T)
        print('UKF:', time.time() - tin)
        return means, covs


class LGSSM(StateSpaceModel):

    def __init__(self, dx, dy, parameters: LinearModelParameters):
        """
        :param dx:
        :param dy:
        :param parameters:
        """
        self.dx = dx
        self.dy = dy
        self.descr = "LG"
        self.params = parameters
        if self.dx == 1:
            self.f = lambda prev_state: prev_state * self.params.A
        else:
            self.f = lambda prev_state: np.matmul(prev_state, self.params.A.T)
        if self.dy == 1:
            self.g = lambda state: np.dot(state, self.params.H.T)
        else:
            self.g = lambda state: np.matmul(state, self.params.H.T)

    def simulate(self, T, init_state):
        """

        :param T:
        :param init_state:
        :return:
        """
        self.T = T
        states = np.zeros([T, self.dx])
        observs = np.zeros([T, self.dy])
        prev_state = init_state
        for t in range(T):
            if self.dx == 1:
                new_state = self.f(prev_state) + np.random.normal(0, self.params.Q)
            else:
                new_state = self.f(prev_state) + np.random.multivariate_normal(np.zeros([self.dx]), self.params.Q)
            if self.dy == 1:
                new_obs = self.g(new_state) + np.random.normal(0, self.params.R)
            else:
                new_obs = self.g(new_state) + np.random.multivariate_normal(np.zeros([self.dy]), self.params.R)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return states, observs

    def kalman_step(self, new_obs, mean_prev, cov_prev):
        """

         :param new_obs:
         :param mean_prev:
         :param cov_prev:
         :return:
         """
        global mean_new, cov_new, lf
        if self.dx == 1:
            m_ = mean_prev * self.params.A
            P_ = self.params.A * cov_prev * self.params.A + self.params.Q
            if np.isnan(new_obs).any():
                lf = np.nan
                return m_, P_, lf
            if self.dy == 1:
                v = new_obs - m_ * self.params.H
                S = self.params.H * P_ * self.params.H + self.params.R
                K = P_ * self.params.H / S

                mean_new = m_ + K * v
                cov_new = P_ - K * S * K
                lf = st.norm(m_ * self.params.H, S).pdf(new_obs)
            else:
                v = new_obs - np.matmul(m_, self.params.H.T)
                S = P_ * np.outer(self.params.H, self.params.H.T) + self.params.R
                K = P_ * np.matmul(self.params.H.T, np.linalg.inv(S))

                mean_new = m_ + np.dot(K, v.T)
                cov_new = P_ - np.dot(np.matmul(K, S), K.T)
                lf = st.multivariate_normal(np.squeeze(self.params.H.T * m_), S).pdf(new_obs)
        else:
            m_ = np.matmul(mean_prev, self.params.A.T)
            P_ = np.matmul(np.matmul(self.params.A, cov_prev), self.params.A.T) + self.params.Q
            if np.isnan(new_obs).any():
                lf = np.nan
                return m_, P_, lf
            v = new_obs - np.matmul(m_, self.params.H.T)
            S = np.matmul(np.matmul(self.params.H, P_), self.params.H.T) + self.params.R
            K = np.matmul(np.matmul(P_, self.params.H.T), np.linalg.inv(S))

            mean_new = m_ + np.matmul(v, K.T)
            cov_new = P_ - np.matmul(np.matmul(K, S), K)
            if self.dy == 1:
                lf = st.norm(np.matmul(m_, self.params.H.T), S).pdf(new_obs)
            else:
                lf = st.multivariate_normal(np.matmul(m_, self.params.H.T), S).pdf(new_obs)
        return mean_new, cov_new, lf

    def kalman_filter(self, observs, init):
        """

        :param observs:
        :param init:
        :return:
        """
        mean_array = np.zeros([self.T, self.dx])
        cov_array = np.zeros([self.T, self.dx, self.dx])
        lf_array = np.zeros([self.T - 1])
        mean_array[0, :] = init[0]
        cov_array[0, :, :] = init[1]
        if self.dx == 1:
            for t in range(self.T - 1):
                m_ = mean_array[t, :] * self.params.A
                P_ = self.params.A * cov_array[t, :] * self.params.A.T + self.params.Q

                if self.dy == 1:
                    v = observs[t] - m_ * self.params.H
                    S = self.params.H * P_ * self.params.H + self.params.R
                    K = P_ * self.params.H / S

                    mean_array[t + 1, :] = m_ + K * v
                    cov_array[t + 1, :] = P_ - K * S * K
                    lf = st.norm(m_ * self.params.H, S).pdf(observs[t])
                else:
                    v = observs[t] - np.matmul(m_, self.params.H.T)
                    S = P_ * np.matmul(self.params.H, self.params.H.T) + self.params.R
                    K = P_ * np.matmul(self.params.H.T, np.linalg.inv(S))

                    mean_array[t + 1, :] = m_ + np.matmul(K, v)
                    cov_array[t + 1, :] = P_ - np.dot(np.matmul(K, S), K.T)
                    lf = st.multivariate_normal(np.matmul(m_, self.params.H.T), S).pdf(observs[t])
                lf_array[t] = lf
        else:
            for t in range(self.T - 1):
                m_ = np.matmul(mean_array[t, :], self.params.A.T)
                P_ = np.matmul(np.matmul(self.params.A, cov_array[t, :]), self.params.A.T) + self.params.Q

                v = observs[t] - np.matmul(m_, self.params.H.T)
                S = np.matmul(np.matmul(self.params.H, P_), self.params.H.T) + self.params.R
                K = np.matmul(np.matmul(P_, self.params.H.T), np.linalg.inv(S))

                mean_array[t + 1, :] = m_ + np.matmul(v, K.T)
                cov_array[t + 1, :] = P_ - np.matmul(np.matmul(K, S), K)
                if self.dy == 1:
                    lf = st.norm(np.matmul(m_, self.params.H.T), S).pdf(observs[t])
                else:
                    lf = st.multivariate_normal(np.matmul(m_, self.params.H.T), S).pdf(observs[t])
                lf_array[t] = lf
        return mean_array, cov_array, lf_array
