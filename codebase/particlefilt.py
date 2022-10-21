import numpy as np
from numpy import random
from jax import numpy as jnp
from jax import jacfwd, jacrev
from scipy.stats import multivariate_normal
import time
import pandas as pd
import utils


class BootstrapPF:

    def __int__(self, ssm, N):
        self.f = ssm.f
        self.g = ssm.g
        self.Q = ssm.Q
        self.R = ssm.R
        self.dx = ssm.dx
        self.dy = ssm.dy
        self.N = N

    def __str__(self):
        return 'BPF'

    def run(self, ys, m0, P0):
        seq_length = np.shape(ys)[0]
        particles = np.zeros(seq_length, self.N, self.dx)
        particles[seq_length] = random.multivariate_normal(m0, P0, self.N)

        for t in range(seq_length):



