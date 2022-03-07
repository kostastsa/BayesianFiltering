# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


class StateSpaceModel:
    """
    Generative Model class.
    
    Attributes
    ----------
    """

    def __init__(self, dx, dy, f = None, g = None, descr = None):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.f = f
        self.g = g
        self.descr = descr

    def __str__(self):
        if self.descr == "LG":
            return str(self.params)

    def set_as_LG(self, lin_mod_params):
        self.descr = "LG"
        self.params = lin_mod_params
        if self.dx == 1:
            self.f = lambda prev_state: self.params.A * prev_state + \
                                        np.random.normal(np.zeros(1), self.params.Q)
        else:
            self.f = lambda prev_state: np.matmul(prev_state, self.params.A.T) + \
                                        np.random.multivariate_normal(np.zeros([self.dx]), self.params.Q)
        if self.dy == 1:
            self.g = lambda prev_state: self.params.H * prev_state + \
                                        np.random.normal(np.zeros(1), self.params.R)
        else:
            self.g = lambda state: np.matmul(state, self.params.H.T) + \
                               np.random.multivariate_normal(np.zeros([self.dy]), self.params.R)

    def simulate(self, T, init_state):
        self.T = T
        self.states = np.zeros([T+1, self.dx])
        self.observs = np.zeros([T, self.dy])
        self.states[0, :] = init_state
        prev_state = init_state
        for t in range(T):
            new_state = self.f(prev_state)
            new_obs = self.g(new_state)
            self.states[t+1, :] = new_state
            self.observs[t, :] = new_obs
            prev_state = new_state

    def propagate(self, prev_state):
        new_state = self.f(prev_state)
        new_obs = self.g(new_state)
        return new_state, new_obs

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
        self.models = np.empty([self.num_models], dtype = StateSpaceModel)
        for m in range(self.num_models):
            self.models[m] = StateSpaceModel(dx,dy)
            self.models[m].set_as_LG(model_parameter_array[m])

    def set_transition_matrix(self, mat):
        self.transition_matrix = mat

    def generate_model_history(self, T, init_model):
        self.T = T
        self.model_history = np.zeros([T], dtype = int)
        self.model_history[0] = init_model
        prev_model = init_model
        if self.transition_matrix.any():
            for t in range(T-1):
                new_model = np.random.choice(range(self.num_models), p=self.transition_matrix[prev_model, :])
                self.model_history[t + 1] = new_model
                prev_model = new_model

    def simulate(self, init_state):
        self.states = np.zeros([self.T + 1, self.dx])
        self.observs = np.zeros([self.T, self.dy])
        self.states[0, :] = init_state
        prev_state = init_state
        for t in range(self.T):
            curr_mod_ind = self.model_history[t]
            new_state, new_obs = self.models[curr_mod_ind].propagate(prev_state)
            self.states[t + 1, :] = new_state
            self.observs[t, :] = new_obs
            prev_state = new_state


        


class LinearModelParameters:

    def __init__(self, A ,H, Q, R):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def __str__(self):

        return 'A : \n' + str(self.A) + '\n' + 'H : \n' + str(self.H) + '\n' + 'Q : \n' + str(self.Q) + '\n' + 'R : \n' + str(self.R)
