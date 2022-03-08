import numpy as np
from generator_classes import StateSpaceModel
from generator_classes import LinearModelParameters
from generator_classes import SLDS

class KalmanFilter:

    def __init__(self, model_set):
        self.model_set = model_set


    def run(self, T, model_history = None):
