from ssm import LGSSM
from ssm import StateSpaceModel
import numpy as np


class SLDS:
    """
    Regime Switching State Space Model class.

    Attributes
    ----------
    """

    def __init__(self, dx, dy, model_parameter_array):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.num_models = len(model_parameter_array)
        self.transition_matrix = np.zeros([self.num_models, self.num_models])
        self.models = np.empty([self.num_models], dtype=StateSpaceModel)
        for m in range(self.num_models):
            self.models[m] = LGSSM(dx, dy, model_parameter_array[m])

    def set_transition_matrix(self, mat):
        self.transition_matrix = mat

    def simulate(self, T, init_state):
        model_history = np.zeros([T], dtype=int)
        states = np.zeros([T, self.dx], dtype=float)
        observs = np.zeros([T, self.dy], dtype=float)

        prev_model = init_state[0]
        prev_state = init_state[1]

        for t in range(T):
            new_model = np.random.choice(range(self.num_models), p=self.transition_matrix[prev_model, :])
            model_history[t] = new_model
            prev_model = new_model
            new_state, new_obs = self.models[new_model].propagate(prev_state)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return [model_history, states], observs

    def conditional_kalman_filter(self, model_history, observs, init):
        # TODO: include functionality for 1D
        t_final = len(model_history)
        mean_array = np.zeros([t_final, self.dx])
        cov_array = np.zeros([t_final, self.dx, self.dx])
        lf_array = np.zeros([t_final-1])
        mean_array[0, :] = init[0]
        cov_array[0, :, :] = init[1]
        for t in range(t_final-2):
            current_model_ind = model_history[t+1]
            model = self.models[current_model_ind]
            mean_array[t+1], cov_array[t+1], lf_array[t+1] = model.kalman_step(observs[t], mean_array[t], cov_array[t])
        return mean_array, cov_array, lf_array
