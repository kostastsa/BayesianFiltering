import numpy as np
from generator_classes import StateSpaceModel
from generator_classes import LinearModelParameters
from generator_classes import SLDS

class KalmanFilter:

    def __init__(self, dx, model_set, observs, model_history = 0):
        self.dx = dx
        (self.T, self.dy) = np.shape(observs)
        self.model_set = model_set
        self.observs = observs
        self.model_history = model_history

    def run(self, init_cond):
        mean_array = np.zeros([self.T, self.dx])
        cov_array = np.zeros([self.T, self.dx, self.dx])
        mean_array[0, :] = init_cond[0]
        cov_array[0, :, :] = init_cond[1]
        for t in range(self.T):
            pass

    def step:
