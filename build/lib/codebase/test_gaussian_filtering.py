import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
import matplotlib.pyplot as plt
import time

dx = 1
dy = 1
num_particles = 10
seq_length = 100
m0 = np.zeros(dx)
P0 = np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 1 * np.eye(dx)
R = 0.1 * np.eye(dy)

## Define nonlinearity
##############################################################  1
f = lambda x: jnp.array([x[0] + jnp.sin(x[1]), 0.9 * x[0]])
H = random.random((dy, dx))
g = lambda x: jnp.array(H @ x) / np.sum(H)
##############################################################  2
# depth = 3
# weights_tensor = np.random.random([depth, dx, dx])
# H = random.random((dy, dx))
# g = lambda x: jnp.array(H @ x) / np.sum(H)
#
# def f(x):
#     for layer in range(depth):
#         W = weights_tensor[layer]
#         out_layer = jnp.array(list(map(sigmoid, W @ x)))
#         x = out_layer
#     return x
#
# def sigmoid(x, alpha=1):
#     return 1 / (1+jnp.exp(-alpha*x))

dx = 10
dy = 1
seq_length = 100
m0 = 0.1 * np.ones(dx)
P0 = 10 * np.eye(dx)
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


# Generate Data

np.random.seed(seed=1)

ssm = gf.SSM(dx, dy, c=c, Q=P0, d=d, R=R, f=f, g=g)
xs, ys = ssm.simulate(seq_length, m0)


# # Test Gaussian filters
# # Filtering
# ## Kalman filter
# # params = s.LinearModelParameters(A, c, B, Q, R)
# # model1 = s.LGSSM(dx, dy, params)
# # model1.T = seq_length
# # kf_out = model1.kalman_filter(ys, [m0, P0])
#
# ## Unscented Kalman Filter
# # ukf = gf.UKF(ssm, 1e-3, 2, 0)
# # ukf_out = ukf.run(ys, m0, P0)
#
# np.random.seed(seed=15)
#
# # Monte Carlo Filter
# mcf = gf.MCF(ssm, num_particles)
# mcf_out = mcf.run(ys, m0, P0)
#
# # Extended Kalman Filter
# ekf = gf.EKF(ssm, order=1)
# ekf_out = ekf.run(ys, m0, P0)
#
# # Monte Carlo Linear Approximation Filter
# mclaf = gf.MCLAF(ssm, num_particles)
# mclaf_out = mclaf.run(ys, m0, P0)
#
#
# print('MCF RMSE:', utils.root_mse(mcf_out[1], xs))
# print('EKF RMSE:', utils.root_mse(ekf_out[1], xs))
# print('MCLAF RMSE:', utils.root_mse(mclaf_out[1], xs))
#
# # Plots
#
#
# fig1, axes1 = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
# p11 = axes1[0].plot(xs[:, 0], alpha=1, label="xs")
# p12 = axes1[0].plot(mclaf_out[1][:, 0], alpha=0.7, label="mclaf")
# p13 = axes1[0].plot(ekf_out[1][:, 0], alpha=0.6, label="ekf")
# p14 = axes1[0].plot(mcf_out[1][:, 0], alpha=0.6, label="mcf")
# axes1[0].set_ylabel("X")
# axes1[0].set_xlabel("time")
# axes1[0].set_title("True States VS filter")
# axes1[0].legend(['x', 'MCLA', 'EKF', 'MC'])
#
# p21 = axes1[1].plot(xs[:, 1], alpha=1, label="xs")
# p22 = axes1[1].plot(mclaf_out[1][:, 1], alpha=0.7, label="mclaf")
# p23 = axes1[1].plot(ekf_out[1][:, 1], alpha=0.6, label="ekf")
# p24 = axes1[1].plot(mcf_out[1][:, 1], alpha=0.6, label="mcf")
# axes1[1].set_ylabel("X")
# axes1[1].set_xlabel("time")
# axes1[1].set_title("True States VS UKF")
# axes1[1].legend(['x', 'MCLA', 'EKF', 'MC'])
#
# # fig2, axes2 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
# # axes2.plot(ekf_out[1]-mclaf_out[1])
#
# plt.show()

# Test Gaussian sum filters
ekf = gf.EKF(ssm, 1)
gsf = gf.GaussSumFilt(ekf, 10)
gsf_out = gsf.run(ys, m0, P0, verbose=True)