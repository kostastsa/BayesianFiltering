import copy

from numpy import ndarray

from ssm import LGSSM
from ssm import StateSpaceModel
import numpy as np
from utils import collapse, dec_to_base


class SLDS:
    """
    Regime Switching State Space Model class. This model class represents
    a SDLS as a set of models together with a transition matrix. 
    -The simulate metasdod takes as input the model (it is a class method), a
     final time T and a initial condition and generates data which are not part
     of the object. The method returns all data, i.e., discrete and continuous
     states as well as observations.
    -The conditional KF method takes as input a model history, observations and 
     a filter initialization and returns means, covariances and the likelihood
     factors of the filter.
    -The IMM method

    Attributes
    ----------
    """

    def __init__(self, dx, dy, model_parameter_array):
        """Initialization method.

        :param dx:
        :param dy:
        :param model_parameter_array:
        """

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
        """

        :param T:
        :param init_state:
        :return:
        """

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
        """

        :param model_history:
        :param observs:
        :param init:
        :return:
        """
        # TODO: include functionality for 1D
        t_final = len(model_history)
        mean_array = np.zeros([t_final, self.dx])
        cov_array = np.zeros([t_final, self.dx, self.dx])
        lf_array = np.zeros([t_final - 1])
        mean_array[0, :] = init[0]
        cov_array[0, :, :] = init[1]
        for t in range(t_final - 2):
            current_model_ind = model_history[t + 1]
            model = self.models[current_model_ind]
            mean_array[t + 1], cov_array[t + 1], lf_array[t + 1] = model.kalman_step(observs[t], mean_array[t],
                                                                                     cov_array[t])
        return mean_array, cov_array, lf_array

    def IMM(self, observs, init):
        """

        :param observs:
        :param init:
        :return:
        """
        t_final = np.shape(observs)[0]
        mean_mat_array = np.zeros([t_final, self.num_models, self.dx])
        cov_tens_array = np.zeros([t_final, self.num_models, self.dx, self.dx])
        lik_fac_vec_array = np.zeros([t_final, self.num_models])
        mean_out_array: ndarray = np.zeros([t_final, self.dx])
        cov_out_array: ndarray = np.zeros([t_final, self.dx, self.dx])
        weight_vec_array = np.ones([t_final, self.num_models]) / self.num_models
        _interm_mean_mat = np.zeros([self.num_models, self.dx])
        _interm_cov_tens = np.zeros([self.num_models, self.dx, self.dx])
        _interm_mix_weight_mat = np.zeros([self.num_models, self.num_models])
        # Initialize means and covariances
        for m in range(self.num_models):
            mean_mat_array[0, m] = init[0]
            cov_tens_array[0, m] = init[1]

        for t in range(1, t_final):
            for m in range(self.num_models):
                # Mixing
                _interm_mix_weight_mat[:, m] = np.multiply(self.transition_matrix[:, m],
                                                           weight_vec_array[t - 1, :])
                norm = np.sum(_interm_mix_weight_mat[:, m])
                _interm_mix_weight_mat[:, m] = _interm_mix_weight_mat[:, m] / norm  # Normalize
                _interm_mean_mat[m], _interm_cov_tens[m] = collapse(mean_mat_array[t - 1],
                                                                    cov_tens_array[t - 1],
                                                                    _interm_mix_weight_mat[:, m])

                # Kalman step
                mean_mat_array[t, m], cov_tens_array[t, m], lik_fac_vec_array[t, m] = self.models[m].kalman_step(
                    observs[t],
                    _interm_mean_mat[m],
                    _interm_cov_tens[m])

                # Weight update
                weight_vec_array[t, m] = lik_fac_vec_array[t, m] * norm

            weight_vec_array[t] = weight_vec_array[t] / np.sum(weight_vec_array[t])
            mean_out_array[t], cov_out_array[t] = collapse(mean_mat_array[t],
                                                           cov_tens_array[t],
                                                           weight_vec_array[t])
        return mean_out_array, cov_out_array, weight_vec_array

    def GPB(self, r, observs, init):
        """

        :param r:
        :param observs:
        :param init:
        :return:
        """
        t_final = np.shape(observs)[0]
        tailShape = self.num_models * np.ones([1, r], dtype=int)
        redTailShape = self.num_models * np.ones([1, r - 1], dtype=int)
        _mean_tens = np.zeros(np.concatenate([tailShape[0], np.array([self.dx])], axis=0))
        _cov_tens = np.zeros(np.concatenate([tailShape[0], np.array([self.dx]), np.array([self.dx])], axis=0))
        _lik_tens = np.zeros(tailShape[0])
        _norm = np.zeros(redTailShape[0])
        _red_mean_tens = np.zeros(np.concatenate([redTailShape[0], np.array([self.dx])], axis=0))
        _red_cov_tens = np.zeros(np.concatenate([redTailShape[0], np.array([self.dx]), np.array([self.dx])], axis=0))
        _weight_tens = np.ones(tailShape[0]) / (self.num_models ** r)

        mean_out_array = np.zeros([t_final, self.dx])
        cov_out_array = np.zeros([t_final, self.dx, self.dx])
        weights_out = np.zeros([t_final, self.num_models ** r])

        # Initialize means and covariances
        for idx in range(self.num_models ** r):
            tail_list = list(map(int, list(dec_to_base(idx, self.num_models).zfill(r))))
            _mean_tens[tail_list] = init[0]
            _cov_tens[tail_list] = init[1]

        for t in range(1, t_final):
            # Mixing + collapse
            for red_tail_idx in range(self.num_models ** (r - 1)):
                _mean_list = np.zeros([self.num_models, self.dx])
                _cov_list = np.zeros([self.num_models, self.dx, self.dx])
                _weight_list = np.zeros(self.num_models)
                red_tail_list = list(map(int, list(dec_to_base(red_tail_idx, self.num_models).zfill(r - 1))))
                for i_0 in range(self.num_models):
                    tail_list = copy.copy(red_tail_list)
                    tail_list.insert(0, i_0)
                    _mean_list[i_0, :] = _mean_tens[tuple(tail_list)]
                    _cov_list[i_0, :, :] = _cov_tens[tuple(tail_list)]
                    _weight_list[i_0] = _weight_tens[tuple(tail_list)]

                _norm[tuple(red_tail_list)] = sum(_weight_list)
                _weight_list = _weight_list / _norm[tuple(red_tail_list)]
                _red_mean_tens[tuple(red_tail_list)], _red_cov_tens[tuple(red_tail_list)] \
                    = collapse(_mean_list, _cov_list, _weight_list)

            # Propagate
            for red_tail_idx in range(self.num_models ** (r - 1)):
                red_tail_list = list(map(int, list(dec_to_base(red_tail_idx, self.num_models).zfill(r - 1))))
                for i_r in range(self.num_models):
                    tail_list = copy.copy(red_tail_list)
                    tail_list.append(i_r)
                    _mean_tens[tuple(tail_list)], _cov_tens[tuple(tail_list)], _lik_tens[tuple(tail_list)] \
                        = self.models[i_r].kalman_step(observs[t],
                                                       _red_mean_tens[tuple(red_tail_list)],
                                                       _red_cov_tens[tuple(red_tail_list)])
                    # Update Weights
                    i = red_tail_list[r-2]
                    _weight_tens[tuple(tail_list)] = _lik_tens[tuple(tail_list)] * \
                                                     self.transition_matrix[i, i_r] * \
                                                     _norm[tuple(red_tail_list)]
            _weight_tens = _weight_tens / np.sum(_weight_tens)
            weights_out[t, :] = np.reshape(_weight_tens, [1, self.num_models ** r])
            # Output
            _mean_list_out = np.reshape(_mean_tens, (self.num_models ** r, self.dx))
            _cov_list_out = np.reshape(_cov_tens, (self.num_models ** r, self.dx, self.dx))
            _weight_list_out = np.reshape(_weight_tens, (self.num_models ** r, 1)).T[0]
            mean_out_array[t, :], cov_out_array[t, :, :] = collapse(_mean_list_out, _cov_list_out, _weight_list_out)

        return mean_out_array, cov_out_array, weights_out
