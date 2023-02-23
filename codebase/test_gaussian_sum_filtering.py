import utils
import gaussfilt as gf
import numpy as np
from jax import numpy as jnp
from numpy import random
import matplotlib.pyplot as plt
import gausssumfilt as gsf
import particlefilt as pf
import time

# Parameters

dx = 1
dy = 1
seq_length = 100
m0 = 0.1 * np.ones(dx)
P0 = 1 * np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 1 * np.eye(dx)
R = 1 * np.eye(dy)

## Define nonlinearity
##############################################################  1 Polynomial
A = 0.8 * np.eye(dx)

# f = lambda x: 1 * (-1/2 + 1 / (1 + jnp.exp(-4*x)))
#f = lambda x: jnp.array([x[0] + jnp.sin(x[1]), 0.9 * x[0]])
#f = lambda x: 0.5 * x + 25 * x / (1+x**2)
# f = lambda x: x
# r = 3.44940
# f = lambda x: jnp.array([max(r*x*(1-x), 1.0)])
f = lambda x: jnp.sin(10 * x) #* jnp.exp(-x**2 / 10)

##############################################################  4
# p = 5
# coeff = jnp.array([0.0, -0.0, 1.4, 0.0001, -0.004])
# g = lambda x: 100 * jnp.dot(coeff, jnp.array([x**i for i in range(p)]))
# p=-1/2
# g = lambda x: jnp.array([(1 + jnp.dot(x, x))**(p/2)])
g = lambda x: 10 * x ** 2 #+ 0.1 * x**2
# g = lambda x: jnp.cos(x)


verbose = False
Nsim = 10
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
    print('sim {}/{}'.format(i, Nsim))
    # Generate Data
    ssm = gf.SSM(dx, dy, c=c, Q=Q, d=d, R=R, f=f, g=g)
    xs, ys = ssm.simulate(seq_length, m0)

    # Gaussian Sum filter
    M0 = 10
    gsf1 = gsf.GaussSumFilt(ssm, M=M0)
    gsf_out = gsf1.run(ys, m0, P0, verbose=verbose)

    # Extended Kalman Filter
    ekf = gf.EKF(ssm, order=1)
    ekf_out = ekf.run(ys, m0, P0, verbose=verbose)

    # Unscented Kalman Filter
    ukf = gf.UKF(ssm)
    ukf_out = ukf.run(ys, m0, P0, verbose=verbose)

    # Unscented Gaussian Sum Filter
    M1 = 10
    ukf1 = gf.UKF(ssm)
    ugsf = gf.GaussSumFilt(ukf1, num_models=M1)
    ugsf_out = ugsf.run(ys, m0, P0, verbose=verbose)

    # Bootstrap Particle Filter
    num_prt = 100
    bpf = pf.BootstrapPF(ssm, num_prt)
    bpf_out = bpf.run(ys, m0, P0, verbose=verbose)
    bpf_mean = np.sum(bpf_out[:seq_length], 1) / num_prt
#
#     # Augmented Gaussian Sum filter
    M = 5
    N = 5
    L = 5
    AGSF = gsf.AugGaussSumFilt(ssm, M, N, L)
    AGSF.set_aug_selection_params(0.2, 0.8, a='input', b='input') # options are ['prop', 'opt_lip', 'opt_max_grad'], a='opt_max_grad', b='opt_max_grad')
    agsf_out = AGSF.run(ys, m0, P0, verbose=verbose)

    # Computation of errors
    ekf_rmse[i] = utils.rmse(ekf_out[1], xs)
    ekf_time[i] = ekf.time
    ukf_rmse[i] = utils.rmse(ukf_out[1], xs)
    ukf_time[i] = ukf.time
    ugsf_rmse[i] = utils.rmse(ugsf_out[1], xs)
    ugsf_time[i] = ugsf.time
    gsf_rmse[i] = utils.rmse(gsf_out[3], xs)
    gsf_time[i] = gsf1.time
    agsf_rmse[i] = utils.rmse(agsf_out[2], xs)
    agsf_time[i] = AGSF.time
    bpf_rmse[i] = utils.rmse(bpf_mean, xs)
    bpf_time[i] = bpf.time

    print('EKF RMSE:', ekf_rmse[i])
    print('EKF time:', ekf_time[i])
    print('UKF RMSE:', ukf_rmse[i])
    print('UKF time:', ukf_time[i])
    print('GSF RMSE:', gsf_rmse[i])
    print('GSF time:', gsf_time[i])
    print('UGSF RMSE:', ugsf_rmse[i])
    print('UGSF time:', ugsf_time[i])
    print('AGSF RMSE:', agsf_rmse[i])
    print('AGSF time:', agsf_time[i])
    print('BPF RMSE:', bpf_rmse[i])
    print('BPF time:', bpf_time[i])
#
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
# Plots
# means
# fig1, axes1 = plt.subplots(5, 1, sharex=True, figsize=(10, 4))
# axes1[0].plot(xs[:, 0], alpha=1, label="xs")
# axes1[0].plot(agsf_out[2], alpha=0.7, label="agsf")
# axes1[0].plot(gsf_out[3], alpha=0.7, label="gsf")
# axes1[0].plot(ugsf_out[3], alpha=0.7, label="gsf")
# axes1[0].plot(ekf_out[1][:, 0], alpha=0.6, label="ekf")
# axes1[0].plot(ukf_out[1][:, 0], alpha=0.6, label="ekf")
# axes1[0].plot(bpf_mean, alpha=0.7, label="agsf")
# # p14 = axes1[0].plot(mcf_out[1][:, 0], alpha=0.6, label="mcf")
# axes1[0].set_ylabel("X")
# #axes1[0].set_xlabel("time")
# axes1[0].set_title("Delta_fac = {}, Lambda_fac = {}".format(AGSF.df, AGSF.lf))
# axes1[0].legend(['x', 'AGSF', 'GSF', 'UGSF' , 'EKF', 'UKF', 'BPF'])
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
# axes1[3].plot(xs[:, 0], alpha=1, label="xs")
# axes1[3].plot(agsf_out[2], alpha=0.7, label="agsf")
# #axes1[3].plot(ukf_out[1][:, 0], alpha=0.7, label="agsf")
# axes1[3].plot(bpf_mean, alpha=0.7, label="agsf")
# axes1[3].legend(['x', 'AGSF', 'BPF'])
#
# for prt in range(num_prt):
#     axes1[4].plot(bpf_out[:, prt], alpha=0.7)
# axes1[4].set_xlabel("time")
# axes1[4].set_title("BPF particles")
#
# plt.show()

# covs
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
#     axes2[2].plot(gsf_out[1].squeeze()[:seq_length, m], alpha=0.7)
# axes2[2].set_ylabel("X")
# axes2[2].set_xlabel("time")
# axes2[2].set_title("GSF covariances")
#
# plt.show()

