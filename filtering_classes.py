import numpy as np
from generator_classes import StateSpaceModel
from generator_classes import LinearModelParameters
from generator_classes import SLDS

class KalmanFilter:

    def __init__(self, model_set):
        if len(model_set)==1:
            self.model = model_set
