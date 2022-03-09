import numpy as np
from scipy import stats as st


class LinearModelParameters:

    def __init__(self, A, H, Q, R):
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

    def __init__(self, dx, dy, f=None, g=None, descr=None):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.f = f
        self.g = g
        self.descr = descr

    def __str__(self):
        if self.descr == "LG":
            return str(self.params)

    def simulate(self, T, init_state):
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
        if self.dx == 1:
            Q = np.array([self.params.Q])
        else:
            Q = self.params.Q
        if self.dy == 1:
            R = np.array([self.params.R])
        else:
            R = self.params.R
        new_state = self.f(prev_state) + np.random.multivariate_normal(np.zeros([self.dx]), Q)
        new_obs = self.g(new_state) + np.random.multivariate_normal(np.zeros([self.dy]), R)
        return new_state, new_obs


class LGSSM(StateSpaceModel):

    def __init__(self, dx, dy, parameters: LinearModelParameters):
        self.dx = dx
        self.dy = dy
        self.descr = "LG"
        self.params = parameters
        self.f = lambda prev_state: np.matmul(prev_state, self.params.A.T)
        self.g = lambda state: np.matmul(state, self.params.H.T)

    def simulate(self, T, init_state):
        self.T = T
        states = np.zeros([T, self.dx])
        observs = np.zeros([T, self.dy])
        prev_state = init_state
        if self.dx == 1:
            Q = np.array([self.params.Q])
        else:
            Q = self.params.Q
        if self.dy == 1:
            R = np.array([self.params.R])
        else:
            R = self.params.R
        for t in range(T):
            new_state = self.f(prev_state) + np.random.multivariate_normal(np.zeros([self.dx]), Q)
            new_obs = self.g(new_state) + np.random.multivariate_normal(np.zeros([self.dy]), R)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return states, observs

    def kalman_step(self, new_obs, mean_prev, cov_prev):
        global mean_new, cov_new, lf
        m_ = np.matmul(mean_prev, self.params.A.T)
        P_ = np.matmul(np.matmul(self.params.A, cov_prev), self.params.A.T) + self.params.Q
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
        # TODO: include functionality for 1D
        mean_array = np.zeros([self.T, self.dx])
        cov_array = np.zeros([self.T, self.dx, self.dx])
        lf_array = np.zeros([self.T-1])
        mean_array[0, :] = init[0]
        cov_array[0, :, :] = init[1]
        for t in range(self.T-1):
            m_ = np.matmul(mean_array[t, :], self.params.A.T)
            P_ = np.matmul(np.matmul(self.params.A, cov_array[t, :]), self.params.A.T) + self.params.Q

            v = observs[t] - np.matmul(m_, self.params.H.T)
            S = np.matmul(np.matmul(self.params.H, P_), self.params.H.T) + self.params.R
            K = np.matmul(np.matmul(P_, self.params.H.T), np.linalg.inv(S))

            mean_array[t+1, :] = m_ + np.matmul(v, K.T)
            cov_array[t+1, :] = P_ - np.matmul(np.matmul(K, S), K)
            if self.dy == 1:
                lf = st.norm(np.matmul(m_, self.params.H.T), S).pdf(observs[t])
            else:
                lf = st.multivariate_normal(np.matmul(m_, self.params.H.T), S).pdf(observs[t])
            lf_array[t] = lf
        return mean_array, cov_array, lf_array




