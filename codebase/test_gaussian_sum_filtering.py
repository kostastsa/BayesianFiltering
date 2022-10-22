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
seq_length = 10
m0 = 10 * np.ones(dx)
P0 = np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 0.1 * np.eye(dx)
R = 1 * np.eye(dy)

dx = 1
dy = 1
seq_length = 100
m0 = 0.1 * np.ones(dx)
P0 =10 * np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 4 * np.eye(dx)
R = 1 * np.eye(dy)

## Define nonlinearity
##############################################################  1 Polynomial
# A = 0.8 * np.eye(dx)
# f = lambda x: 1 * (-1/2 + 1 / (1 + jnp.exp(-4*x)))
# p = 5
# coeff = jnp.array([0.0, -0.0, 1.4 ,0.0001, -0.004])
# g = lambda x: jnp.dot(coeff, jnp.array([x**i for i in range(p)]))

##############################################################  4
A = 0.8 * np.eye(dx)
f = lambda x: jnp.sin(x)
g = lambda x: x**2

Nsim = 1
ekf_rmse = np.zeros(Nsim)
gsf_rmse = np.zeros(Nsim)
agsf_rmse = np.zeros(Nsim)
for i in range(Nsim):
    print('sim {}/{}'.format(i, Nsim))
    # Generate Data
    ssm = gf.SSM(dx, dy, c=c, Q=P0, d=d, R=R, f=f, g=g)
    xs, ys = ssm.simulate(seq_length, m0)

    # Gaussian Sum filter
    M0 = 10
    GSF = gsf.GaussSumFilt(ssm, M=M0)
    gsf_out = GSF.run(ys, m0, P0, verbose=False)
    gsf_mean = np.sum(np.multiply(gsf_out[0].squeeze(), gsf_out[2]), 1)

    # Extended Kalman Filter
    ekf = gf.EKF(ssm, order=1)
    ekf_out = ekf.run(ys, m0, P0, verbose=False)

    # Augmented Gaussian Sum filter
    M = 5
    N = 2
    L = 10
    AGSF = gsf.AugGaussSumFilt(ssm, M, N, L)
    AGSF.set_aug_selection_params(1, 0.7, a='prop', b='prop') # options are ['prop', 'opt_lip', 'opt_max_grad'], a='opt_max_grad', b='opt_max_grad')

    agsf_out = AGSF.run(ys, m0, P0, verbose=False)
    agsf_mean = np.sum(agsf_out[0], 2) / M

    # Computation of errors
    ekf_rmse[i] = utils.rmse(ekf_out[0], xs)
    gsf_rmse[i] = utils.rmse(gsf_mean[:seq_length], xs)
    agsf_rmse[i] = utils.rmse(agsf_mean[:seq_length], xs)

# Plots
# # means
# fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
# p11 = axes1[0].plot(xs[:, 0], alpha=1, label="xs")
# p12 = axes1[0].plot(agsf_mean, alpha=0.7, label="agsf")
# p13 = axes1[0].plot(gsf_mean, alpha=0.7, label="gsf")
# p14 = axes1[0].plot(ekf_out[1][:, 0], alpha=0.6, label="ekf")
# # p14 = axes1[0].plot(mcf_out[1][:, 0], alpha=0.6, label="mcf")
# axes1[0].set_ylabel("X")
# #axes1[0].set_xlabel("time")
# axes1[0].set_title("Delta_fac = {}, Lambda_fac = {}".format(AGSF.df, AGSF.lf))
# axes1[0].legend(['x', 'AGSF', 'GSF', 'EKF'])
#
# for m in range(M):
#     p22 = axes1[1].plot(agsf_out[0][:, :, m], alpha=0.7)
# axes1[1].set_ylabel("X")
# #axes1[1].set_xlabel("time")
# axes1[1].set_title("AGSF Components")
#
# for m in range(M0):
#     p22 = axes1[2].plot(gsf_out[0][:, :, m], alpha=0.7)
# axes1[2].set_ylabel("X")
# axes1[2].set_xlabel("time")
# axes1[2].set_title("GSF Components")
#
# plt.show()
#
# # covs
# fig2, axes2 = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
# axes2[0].plot(ekf_out[2].squeeze(), alpha=0.6, label="ekf")
# axes2[0].set_ylabel("X")
# axes2[0].set_title("EKF cov")
#
# for m in range(M):
#     axes2[1].plot(agsf_out[1].squeeze()[:, m], alpha=0.7)
# axes2[1].set_ylabel("X")
# axes2[1].set_title("AGSF covariances")
#
# for m in range(M0):
#     p22 = axes1[2].plot(gsf_out[1].squeeze()[:, m], alpha=0.7)
# axes2[2].set_ylabel("X")
# axes2[2].set_xlabel("time")
# axes2[2].set_title("GSF covariances")
#
#
#
# plt.show()

