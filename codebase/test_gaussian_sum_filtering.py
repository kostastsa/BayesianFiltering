import utils
import gaussfilt as gf
import numpy as np
from jax import numpy as jnp
from numpy import random
import matplotlib.pyplot as plt
import gausssumfilt as gsf
import time

dx = 1
dy = 1
seq_length = 100
m0 = 0.1 * np.ones(dx)
P0 = np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 1 * np.eye(dx)
R = 1 * np.eye(dy)

## Define nonlinearity
##############################################################  1
A = 0.8 * np.eye(dx)
#f = lambda x: 0.5 * x + 25 * x / (1+ x**2)
f = lambda x: jnp.cos(x)
g = lambda x: jnp.array([jnp.sin(x) / x])

##############################################################  4
# A = 0.8 * np.eye(dx)
# B = 0.5 * np.eye(dy, dx)
# f = lambda x: A @ x
# g = lambda x: B @ x


# Generate Data

np.random.seed(seed=2)

ssm = gf.SSM(dx, dy, c=c, Q=P0, d=d, R=R, f=f, g=g)
xs, ys = ssm.simulate(seq_length, m0)

# Gaussian Sum filter
M0 = 10
GSF = gsf.GaussSumFilt(ssm, M=M0)
gsf_out = GSF.run(ys, m0, P0)
gsf_mean = np.sum(np.multiply(gsf_out[0].squeeze(), gsf_out[2]), 1)

# Extended Kalman Filter
ekf = gf.EKF(ssm, order=1)
ekf_out = ekf.run(ys, m0, P0)

for i in range(1):

    # Augmented Gaussian Sum filter
    M = 5
    N = 2
    L = 10
    AGSF = gsf.AugGaussSumFilt(ssm, M, N, L)
    AGSF.set_aug_selection_params(1, 0.1, a='prop', b='prop')
    agsf_out = AGSF.run(ys, m0, P0)
    agsf_mean = np.sum(agsf_out[0], 2) / M

    # Plots

    fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
    p11 = axes1[0].plot(xs[:, 0], alpha=1, label="xs")
    p12 = axes1[0].plot(agsf_mean, alpha=0.7, label="agsf")
    p13 = axes1[0].plot(gsf_mean, alpha=0.7, label="gsf")
    p14 = axes1[0].plot(ekf_out[1][:, 0], alpha=0.6, label="ekf")
    # p14 = axes1[0].plot(mcf_out[1][:, 0], alpha=0.6, label="mcf")
    axes1[0].set_ylabel("X")
    #axes1[0].set_xlabel("time")
    axes1[0].set_title("Delta_fac = {}, Lambda_fac = {}".format(AGSF.df, AGSF.lf))
    axes1[0].legend(['x', 'AGSF', 'GSF', 'EKF'])

    for m in range(M):
        p22 = axes1[1].plot(agsf_out[0][:, :, m], alpha=0.7)
    axes1[1].set_ylabel("X")
    #axes1[1].set_xlabel("time")
    axes1[1].set_title("AGSF Components")

    for m in range(M0):
        p22 = axes1[2].plot(gsf_out[0][:, :, m], alpha=0.7)
    axes1[2].set_ylabel("X")
    axes1[2].set_xlabel("time")
    axes1[2].set_title("GSF Components")


    plt.show()

