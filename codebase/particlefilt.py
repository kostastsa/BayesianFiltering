import numpy as np
from numpy import random
from jax import numpy as jnp
from jax import jacfwd, jacrev
from scipy.stats import multivariate_normal
import time
import pandas as pd
import utils


class BootstrapPF:

    def __init__(self, ssm, N):
        self.f = ssm.f
        self.g = ssm.g
        self.Q = ssm.Q
        self.R = ssm.R
        self.dx = ssm.dx
        self.dy = ssm.dy
        self.N = N
        self.time = 0.0

    def __str__(self):
        return 'BPF'

    def run(self, ys, m0, P0, verbose = False):
        tin = time.time()
        seq_length = np.shape(ys)[0]
        particles = np.zeros((seq_length+1, self.N, self.dx))
        particles[seq_length] = random.multivariate_normal(m0, P0, self.N)
        new_particles = np.zeros((self.N, self.dx))
        weights = np.ones(self.N) / self.N

        for t in range(seq_length):
            if verbose:
                print('{}.run | t='.format(self), t)
            for p in range(self.N):
                new_particles[p] = random.multivariate_normal(self.f(particles[t-1, p]), self.Q)
                loglik = utils.gaussian_logpdf(ys[t], self.g(new_particles[p]), self.R)
                weights[p] = np.exp(loglik)
            weights /= np.sum(weights)
            # Resampling
            idx_count = random.multinomial(self.N, pvals=weights)
            _particles = []
            for p in range(self.N):
                for count in range(idx_count[p]):
                    _particles.append(new_particles[p])
            particles[t] = np.array(_particles)
        self.time = time.time() - tin
        return particles





