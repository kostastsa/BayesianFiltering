import gaussfiltax.utils as utils
import gaussfiltax.gaussfilt as gf
import numpy as np
from jax import numpy as jnp
from numpy import random
import matplotlib.pyplot as plt
import gaussfiltax.gausssumfilt as gsf
import gaussfiltax.particlefilt as pf
import time

# Parameters

dx = 3
dy = 1
seq_length = 100
m0_sim = np.array([0.0, 1.0, 1.05])
m0 = np.zeros(dx)
P0 = 10 * np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 0.1 * np.eye(dx)
R = 1 * np.eye(dy)

## Define nonlinearity
##############################################################  1 Polynomial
A = 0.8 * np.eye(dx)

# f = lambda x: 1 * (-1/2 + 1 / (1 + jnp.exp(-4*x)))
# f = lambda x: jnp.array([x[0] + jnp.sin(x[1]), 0.9 * x[0]])
# f = lambda x: 0.5 * x + 25 * x / (1+x**2)
# f = lambda x: x
# r = 3.44940
# f = lambda x: jnp.array([max(r*x*(1-x), 1.0)])
# f = lambda x: jnp.sin(jnp.array([10 * x[0]**2 + x[1], x[0]*x[1]])) #* jnp.exp(-x**2 / 10)
# f = lambda x: jnp.sin(10 * x)
# Lorenz
def lorentz_63(x, sigma=10, rho=28, beta=2.667, dt=0.01):
    dx = dt * sigma * (x[1] - x[0])
    dy = dt * (x[0] * rho - x[1] - x[0] *x[2]) 
    dz = dt * (x[0] * x[1] - beta * x[2])
    return jnp.array([dx+x[0], dy+x[1], dz+x[2]])

# def f(x, sigma=10, rho=28, beta=2.667, dt=0.01):
#     dx = dt * sigma * (x[1]**3 - x[0]*x[1]*x[2])
#     dy = dt * (x[0] * rho - x[1] - x[0] *x[2]**2) 
#     dz = dt * (x[0] * x[1] - beta * x[2])
#     return jnp.array([dx+x[0], dy+x[1], dz+x[2]])



##############################################################  4
# p = 5
# coeff = jnp.array([0.0, -0.0, 1.4, 0.0001, -0.004])
# g = lambda x: 100 * jnp.dot(coeff, jnp.array([x**i for i in range(p)]))
# p=-1/2
# g = lambda x: jnp.array([(1 + jnp.dot(x, x))**(1/2)])
# g = lambda x: 0.1 * x**2
# g = lambda x: 20 * jnp.array([x[1], x[0]+x[1]]) ** 2 #+ 0.1 * x**2
# g = lambda x: jnp.cos(x)
g = lambda x: jnp.array([jnp.dot(x, x)])


verbose = False
Nsim = 1
ekf_rmse = np.zeros(Nsim)
ukf_rmse = np.zeros(Nsim)
ugsf_rmse = np.zeros(Nsim)
gsf_rmse = np.zeros(Nsim)
agsf_rmse = np.zeros(Nsim)
ekf_time = np.zeros(Nsim)
ukf_time = np.zeros(Nsim)
ugsf_time = np.zeros(Nsim)
gsf_time = np.zeros(Nsim)
agsf_time = np.zeros(Nsim)
bpf_rmse = np.zeros(Nsim)
bpf_time = np.zeros(Nsim)
for i in range(Nsim):
    print('sim {}/{}'.format(i+1, Nsim))
    # Generate Data
    ssm = gf.SSM(dx, dy, c=c, Q=Q, d=d, R=R, f=lorentz_63, g=g)
    xs, ys = ssm.simulate(seq_length, m0_sim)

    # # Gaussian Sum filter
    # M0 = 10
    # gsf1 = gsf.GaussSumFilt(ssm, M=M0)
    # gsf_out = gsf1.run(ys, m0, P0, verbose=verbose)

    # # Extended Kalman Filter
    # ekf = gf.EKF(ssm, order=1)
    # ekf_out = ekf.run(ys, m0, P0, verbose=verbose)

    # # Unscented Kalman Filter
    # ukf = gf.UKF(ssm)
    # ukf_out = ukf.run(ys, m0, P0, verbose=verbose)

    # Unscented Gaussian Sum Filter
    # M1 = 10
    # ukf1 = gf.UKF(ssm)
    # ugsf = gf.GaussSumFilt(ukf1, num_models=M1)
    # ugsf_out = ugsf.run(ys, m0, P0, verbose=verbose)

#     # Bootstrap Particle Filter
    num_prt = 100
    bpf = pf.BootstrapPF(ssm, num_prt)
    bpf_out = bpf.run(ys, m0, P0, verbose=verbose)
    bpf_mean = np.sum(bpf_out[:seq_length], 1) / num_prt

    # Augmented Gaussian Sum filter
    M = 3
    N = 2
    L = 2
    AGSF = gsf.AugGaussSumFilt(ssm, M, N, L)
    AGSF.set_aug_selection_params(0.5, 0.5, a='prop', b='prop') # options are ['prop', 'opt_lip', 'opt_max_grad', 'input']
    agsf_out = AGSF.run(ys, m0, P0, verbose=verbose)

    print(agsf_out[3])

#     # Computation of errors
#     # ekf_rmse[i] = utils.rmse(ekf_out[1], xs)
#     # ekf_time[i] = ekf.time
#     # ukf_rmse[i] = utils.rmse(ukf_out[1], xs)
#     # ukf_time[i] = ukf.time
#     # ugsf_rmse[i] = utils.rmse(ugsf_out[1], xs)
#     # ugsf_time[i] = ugsf.time
#     # gsf_rmse[i] = utils.rmse(gsf_out[3], xs)
#     # gsf_time[i] = gsf1.time
    agsf_rmse[i] = utils.rmse(agsf_out[2], xs)
    agsf_time[i] = AGSF.time
    bpf_rmse[i] = utils.rmse(bpf_mean, xs)
    bpf_time[i] = bpf.time

#     # print('EKF RMSE:', ekf_rmse[i])
#     # print('EKF time:', ekf_time[i])
#     # print('UKF RMSE:', ukf_rmse[i])
#     # print('UKF time:', ukf_time[i])
#     # print('GSF RMSE:', gsf_rmse[i])
#     # print('GSF time:', gsf_time[i])
#     # print('UGSF RMSE:', ugsf_rmse[i])
#     # print('UGSF time:', ugsf_time[i])
    print('AGSF RMSE:', agsf_rmse[i])
    print('AGSF time:', agsf_time[i])
    print('BPF RMSE:', bpf_rmse[i])
    print('BPF time:', bpf_time[i])

#     EKF_W = np.zeros(seq_length)
#     GSF_W = np.zeros(seq_length)
#     AGSF_W = np.zeros(seq_length)
#
#     for t in range(seq_length):
#         particles = bpf_out[t]
#         # EKF
#         means = ekf_out[1][t].reshape(1, dx)
#         covs = ekf_out[2][t].reshape(1, dx, dx)
#         EKF_W[t] = utils.W_distance(means, covs, particles, [1.0])
#
#         # GSF
#         means = gsf_out[0][t].reshape(M0, dx)
#         covs = gsf_out[1][t].reshape(M0, dx, dx)
#         weights = gsf_out[2][t]
#         GSF_W[t] = utils.W_distance(means, covs, particles, weights)
#
#         # AGSF
#         means = agsf_out[0][t].reshape(M, dx)
#         covs = agsf_out[1][t].reshape(M, dx, dx)
#         AGSF_W[t] = utils.W_distance(means, covs, particles, [1 / M] * M)
#
#

# #### Plots

## 3D Plot

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

# Data for AGSF
agsf0 = agsf_out[0][:, :, 0]
ax.scatter3D(*agsf0.T, s=0.1, c='b');

agsf1 = agsf_out[0][:, :, 1]
ax.scatter3D(*agsf1.T, s=0.1, c='r');

agsf2 = agsf_out[0][:, :, 2]
ax.scatter3D(*agsf2.T, s=0.1, c='y');

# Data for BPF
ax.scatter3D(*bpf_mean.T, s=0.1, c='g');

# ax1 = plt.figure().add_subplot()

# ax1.plot(ys)

plt.show()

