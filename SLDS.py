from ssm import LGSSM
from ssm import StateSpaceModel
import numpy as np
from utils import Utils as u


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
            mean_array[t+1], cov_array[t+1], lf_array[t+1] =  model.kalman_step(observs[t], mean_array[t], cov_array[t])
        return mean_array, cov_array, lf_array


    def IMM(self, observs, init):
        t_final = np.shape(observs)[0]
        mean_mat_array = np.zeros([t_final,self.num_models , self.dx])
        cov_tens_array = np.zeros([t_final, self.num_models, self.dx, self.dx])
        lik_fac_vec_array = np.zeros([t_final, self.num_models])
        mean_out_array = np.zeros([t_final, self.dx])
        cov_out_array = np.zeros([t_final, self.dx, self.dx])
        weight_vec_array = np.ones([t_final, self.num_models]) / self.num_models
        _interm_mean_mat = np.zeros([self.num_models , self.dx])
        _interm_cov_tens = np.zeros([self.num_models , self.dx, self.dx])
        _interm_mix_weight_mat = np.zeros([self.num_models, self.num_models])
        # Initialize means and covariances
        for m in range(self.num_models):
            mean_mat_array[0, m] = init[0]
            cov_tens_array[0, m] = init[1]
        
        for t in range(1,t_final):
            for m in range(self.num_models):
                # Mixing
                _interm_mix_weight_mat[:, m] = np.multiply(self.transition_matrix[:,m],
                                                           weight_vec_array[t-1, :])
                norm = np.sum( _interm_mix_weight_mat[:, m])
                _interm_mix_weight_mat[:, m] = _interm_mix_weight_mat[:, m] / norm  # Normalize
                _interm_mean_mat[m], _interm_cov_tens[m] = u.collapse(mean_mat_array[t-1],
                                                                      cov_tens_array[t-1],
                                                                      _interm_mix_weight_mat[:,m])
                
                # Kalman step
                mean_mat_array[t,m], cov_tens_array[t,m], lik_fac_vec_array[t,m] =  self.models[m].kalman_step(observs[t],  
                                                                     _interm_mean_mat[m],
                                                                     _interm_cov_tens[m])
                
                # Weight update
                weight_vec_array[t,m] = lik_fac_vec_array[t,m] * norm
            
            weight_vec_array[t] = weight_vec_array[t] / np.sum(weight_vec_array[t])
            mean_out_array[t], cov_out_array[t] = u.collapse(mean_mat_array[t],
                                                             cov_tens_array[t],
                                                             weight_vec_array[t])
        return mean_out_array, cov_out_array
        
            