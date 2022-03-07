# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


class GenerativeModel:
    """
    Generative Model class.
    
    Attributes
    ----------
    
    """

    def __init__(self, dx, dy, T):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.T = T
             
class StateSpaceModel:
    """
    State Space Model class.
    
    Attributes
    ----------
    """
    
    def __init__(self, dx, dy):
        """Initialization method."""
        self.f = None
        self.g = None
        self.dx = dx
        self.dy = dy
        
    def generate(self, prev_state):
        new_state = self.f(prev_state)
        new_obs = self.g(new_state)
        return np.array([new_state, new_obs])
   
    
class LinearGaussianSSM(StateSpaceModel):
    """
    Lineatr Gaussian State Space Model class.
    
    Attributes
    ----------
    """
    
    def __init__(self, A, H, Q, R):
        self.dx = np.shape(A)[0]
        self.dy = np.shape(H)[0]
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.f = lambda prev_state : np.matmul(prev_state, A.T) +  \
                  np.random.multivariate_normal(np.zeros([self.dx]), self.Q)
        self.g = lambda state : np.matmul(state, H.T) + \
                  np.random.multivariate_normal(np.zeros([self.dy]), self.R)
    

    @property
    def get_parameters(self):
        """Parameter getter"""
        return {"A":self.A, "H":self.H, "Q":self.Q, "R":self.R}
  
class RegimeSwitchingSSM:
    """
    Regime Switching State Space Model class.

    Attributes
    ----------
    """
    
    def __init__(self, models):
        """Initialization method."""
        self.num_models = len(models)
        self.models = models
        self.p = None
        
    def generate_model_indices(self, length, initial_index):
        self.length = length
        model_indices = np.zeros([1, length], dtype = int)
        model_indices[0] = initial_index
        for i in range(length):
            None
        return
                       
    def generate(self, prev_model_ind, prev_state):
        return
        
        