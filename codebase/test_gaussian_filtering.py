import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
import matplotlib.pyplot as plt
import time

dx = 1
dy = 1
num_particles = 100
seq_length = 100
m0 = np.zeros(dx)
P0 = np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 1 * np.eye(dx)
R = 0.1 * np.eye(dy)

## Define nonlinearity
##############################################################  1
# f = lambda x: jnp.array([x[0] + jnp.sin(x[1]), 0.9 * x[0]])
# A1 = jnp.array([[2, 0], [0, -1]])
# g = lambda x: jnp.array([jnp.dot(x, A1 @ x) / 2])
##############################################################  2
# depth = 3
# weights_tensor = np.random.random([depth, dx, dx])
# g = lambda x: jnp.array([jnp.dot(x, x)])
#
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

##############################################################  3
# f = lambda x: jnp.array([(1 + jnp.dot(x,x))**(3/2), 1])
# g = lambda x: jnp.array([(1 + jnp.dot(x, x))**(-1/2)])
##############################################################  4
f = lambda x: 0.0 * x
g = lambda x: x**2
##############################################################  5
# A = 0.8 * np.eye(dx)
# B = 0.5 * np.eye(dy, dx)
# f = lambda x: A @ x
# g = lambda x: B @ x

# Generate Data

np.random.seed(seed=1)
m = np.array([1])
P = np.eye(dx)
ssm = gf.SSM(dx, dy, c=m, Q=P, d=d, R=R, f=f, g=g)
xs, ys = ssm.simulate(seq_length, m)

xs = random.multivariate_normal(m, P, seq_length)
ys = np.array(list(map(g, xs)))

# Filtering
## Kalman filter
# params = s.LinearModelParameters(A, c, B, Q, R)
# model1 = s.LGSSM(dx, dy, params)
# model1.T = seq_length
# kf_out = model1.kalman_filter(ys, [m0, P0])

## Unscented Kalman Filter
# ukf = gf.UKF(ssm, 1e-3, 2, 0)
# ukf_out = ukf.run(ys, m0, P0)

np.random.seed(seed=15)

## Monte Carlo Filter
mcf = gf.MCF(ssm, num_particles)
#mcf_out = mcf.run(ys, m0, P0)
filtered_means = np.zeros((seq_length, dx))
num_gain_positive = 0
for t in range(seq_length):
    print(t)
    mu_y, Sy, Cxy = mcf.moment_approx(m0, P0, 'upd')
    gain_matrix = Cxy @ np.linalg.inv(Sy)  # TODO: replace inv with more efficient implementation
    filtered_means[t] = m + (ys[t] - mu_y) @ gain_matrix.T
    num_gain_positive += (gain_matrix>0)
    print('y:', ys[t])
    print('mu_y:', mu_y)
    print('gain:', gain_matrix)
    print('Cxy:', Cxy)



# ## Extended Kalman Filter
# ekf = gf.EKF(ssm, order=1)
# ekf_out = ekf.run(ys, m0, P0)

## test Monte Carlo Linear Approximation Filter
# mclaf = gf.MCLAF(ssm, num_particles)
# mclaf_out = mclaf.run(ys, m0, P0)
#
#
# print('MCF RMSE:', utils.root_mse(mcf_out[1], xs))
# print('EKF RMSE:', utils.root_mse(ekf_out[1], xs))
# print('MCLAF RMSE:', utils.root_mse(mclaf_out[1], xs))

# Plots

fig1, axes1 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes1.plot(xs[:, 0], alpha=1, label="xs")
p2 = axes1.plot(ys[:, 0], alpha=0.6, label="ekf")
p3 = axes1.plot(filtered_means, alpha=0.6, label="mcf")
axes1.legend(['x', 'y', 'FM'])

# fig1, axes1 = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
# p1 = axes1[0].plot(xs[:, 0], alpha=1, label="xs")
# #p2 = axes1[0].plot(mclaf_out[1][:, 0], alpha=0.7, label="mclaf")
# #p22 = axes1[0].plot(ekf_out[1][:, 0], alpha=0.6, label="ekf")
# p3 = axes1[0].plot(mcf_out[1][:, 0], alpha=0.6, label="mcf")
# axes1[0].set_ylabel("X")
# axes1[0].set_xlabel("time")
# axes1[0].set_title("True States VS filter")
# axes1[0].legend(['x', 'MC'])

# p21 = axes1[1].plot(xs[:, 1], alpha=1, label="xs")
# #p22 = axes1[1].plot(mclaf_out[1][:, 1], alpha=0.7, label="mclaf")
# #p23 = axes1[1].plot(ekf_out[1][:, 1], alpha=0.6, label="ekf")
# p24 = axes1[1].plot(mcf_out[1][:, 1], alpha=0.6, label="mcf")
# axes1[1].set_ylabel("X")
# axes1[1].set_xlabel("time")
# axes1[1].set_title("True States VS UKF")
# axes1[1].legend(['x', 'MC'])

# fig2, axes2 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
# axes2.plot(ekf_out[1]-mclaf_out[1])

plt.show()
