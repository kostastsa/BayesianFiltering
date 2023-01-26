import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
import time

dx = 2
dy = 1

## Define nonlinearity
##############################################################  1
# f = lambda x: 0.0 * jnp.array([x[0] + jnp.sin(x[1]), 0.9 * x[0]])
# A1 = jnp.array([[2, 0], [0, -1]])
# g = lambda x: jnp.array([jnp.dot(x, A1 @ x) / 2])

##############################################################  1
# p = -3
# g = lambda x: jnp.array([(1 + jnp.dot(x, x))**(p/2)])

##############################################################  2
# depth = 3
# #random.seed(1)
# weights_tensor = np.multiply(np.random.random([depth, dx, dx]),
#                              np.random.choice([0, 1], (depth, dx, dx), p=[0.1, 0.9]))  # masked
#
#
# def g(x):
#     activation = ReLU
#     for layer in range(depth):
#         W = weights_tensor[layer]
#         out_layer = jnp.array(list(map(activation, W @ x)))
#         x = out_layer
#     return jnp.array([jnp.dot(x, x)])
#
#
# def sigmoid(x, alpha=1):
#     return 1 / (1 + jnp.exp(-alpha * x))
#
#
# def ReLU(x):
#     return max(0.0, x)


##############################################################  3
# f = lambda x: 0.0 * jnp.array([1])
# g = lambda x: jnp.array([(jnp.cos(jnp.dot(x, x)))])
##############################################################  4

##############################################################  5
# A = 0.8 * np.eye(dx)
# B = 0.5 * np.eye(dy, dx)
# f = lambda x: A @ x
# g = lambda x: B @ x

# 3 Linear-Nonlinear product  (dx = 2)
a = 1
g = lambda x: jnp.array([a * x[0] * jnp.sin(x[0] * x[1])])

## 4 Linear-Nonlinear sum
# (dx = 2)
# g = lambda x: jnp.array([x[0] + jnp.sin(x[1])])
#(dx = 5)
# g = lambda x: jnp.array([x[0] + 10 * x[3] + 3 * jnp.sin(x[2] * x[1]) * x[4] ** 2])

# g = lambda x: jnp.array([x**2])

# Generate Data

#np.random.seed(seed=1)
m = 1.0 * np.ones(dx)
P = np.multiply(np.diag(2 * np.arange(1, dx+1)), np.eye(dx))

# Baseline estimate

baseline_sample_size = 100000
baseline_sample = random.multivariate_normal(m, P, baseline_sample_size)
trans_baseline_sample = np.array(list(map(g, baseline_sample)))
baseline_estimate = np.sum(trans_baseline_sample) / baseline_sample_size
baseline_var = np.sum((trans_baseline_sample - baseline_estimate) ** 2) / (baseline_sample_size - 1)

print('Baseline Estimate', baseline_estimate)
print('Baseline Estimate Variance', baseline_var)

random.seed(int(100 * random.rand()))
# Simple MC
N = 10
N_trials = 100
mc_estimates = np.zeros(N_trials)
for trial in range(N_trials):
    mc_sample = random.multivariate_normal(m, P, N)
    trans_mc_sample = np.array(list(map(g, mc_sample)))
    mc_estimates[trial] = np.sum(trans_mc_sample) / N

over_mc_estimate = np.sum(mc_estimates) / N_trials
mc_estimates_var = np.sum((mc_estimates - over_mc_estimate) ** 2) / (N_trials - 1)
mc_estimates_mse = np.sum((mc_estimates - baseline_estimate) ** 2) / (N_trials - 1)

# Augmented
# N = 10
#
# N_trials = 100
hessian = jacfwd(jacrev(g))
hess_m = jnp.sum(hessian(m), axis=0)

for p in [11]:  # range(11, 12):

    # Set Delta
    Delta = p * 0.1 * P
    if p == 11:
        Delta_opt = utils.sdp_opt(dx, N, 0.1, P, P, hess_m, 10, 0.01)
        Delta = Delta_opt

    #  Monte-Carlo Linear and Quadratic Estimates
    mcl_estimates = np.zeros(N_trials)
    mcq_estimates = np.zeros(N_trials)
    mc2_estimates = np.zeros(N_trials)
    for trial in range(N_trials):
        mcl_sample = random.multivariate_normal(m, P - Delta, N)
        trans_mcl_sample = np.array(list(map(g, mcl_sample)))
        hess_at_sample = np.array(list(map(hessian, mcl_sample)))
        H = jnp.sum(hess_at_sample, axis=0).squeeze() / N
        if dx == 1:
            H = H.reshape((1, 1))
        mcl_estimates[trial] = np.sum(trans_mcl_sample) / N
        mcq_estimates[trial] = np.sum(trans_mcl_sample) / N + np.trace(H @ Delta) / 2

        # MC2
        Nz = 10
        #mc2_sample = random.multivariate_normal(m, P - Delta, Nz)  # no importance sampling
        mc2_sample = random.multivariate_normal(m, P, Nz)  # for importance sampling density q

        # Sample allocations
        prop_allocations = np.zeros(Nz)
        i = 0
        for sample in mc2_sample:
            prop_allocations[i] = multivariate_normal.pdf(sample, m, P - Delta, allow_singular=True)\
                                  / multivariate_normal.pdf(sample, m, P, allow_singular=True)
            i += 1
        prop_allocations = prop_allocations / np.sum(prop_allocations)

        i = 0
        for sample in mc2_sample:
            # Importance Sampling with proportional allocation
            Mn = int(prop_allocations[i] * Nz)
            i += 1
            # print('trial', trial, 'Mn', Mn)
            if Mn > 0:
                Mn = int(min(Mn, 10))
                mc2_sample_2 = random.multivariate_normal(sample, Delta, Mn)
                trans_mc2_sample = np.array(list(map(g, mc2_sample_2)))
                mc2_estimates[trial] += np.sum(trans_mc2_sample) / N

    simple_quadratic_estimate = g(m) + np.trace(hess_m @ P) / 2
    smart_quadratic_estimate = g(m) + np.trace(hess_m @ Delta) / 2

# MSE estimates
    over_mcl_estimate = np.sum(mcl_estimates) / N_trials
    mcl_estimates_var = np.sum((mcl_estimates - over_mcl_estimate) ** 2) / (N_trials - 1)
    mcl_estimates_mse = np.sum((mcl_estimates - baseline_estimate) ** 2) / (N_trials - 1)

    over_mcq_estimate = np.sum(mcq_estimates) / N_trials
    mcq_estimates_var = np.sum((mcq_estimates - over_mcq_estimate) ** 2) / (N_trials - 1)
    mcq_estimates_mse = np.sum((mcq_estimates - baseline_estimate) ** 2) / (N_trials - 1)

    over_mc2_estimate = np.sum(mc2_estimates) / N_trials
    mc2_estimates_var = np.sum((mc2_estimates - over_mc2_estimate) ** 2) / (N_trials - 1)
    mc2_estimates_mse = np.sum((mc2_estimates - baseline_estimate) ** 2) / (N_trials - 1)

# Prints
    print('p', p * .1)
    print('-' * 20)
    print(' ** simple MC **')
    #print('I_MC estimates', mc_estimates)
    print('I_MC variance', mc_estimates_var)
    print('I_MC MSE', mc_estimates_mse)
    print('-' * 20)
    print(' ** MC2 **')
    #print('I_MC2 estimates', mc2_estimates)
    print('I_MC2 variance', mc2_estimates_var)
    print('I_MC2 MSE', mc2_estimates_mse)
    print('-' * 20)
    print(' ** Linear Estimates **')
    #print('I_MCL estimates', mcl_estimates)
    print('I_MCL variance', mcl_estimates_var)
    print('I_MCL MSE', mcl_estimates_mse)
    print(' ** simple Linear **')
    print('I_L', float(g(m)))
    print('I_L MSE', (g(m) - baseline_estimate) ** 2)
    print('-' * 20)
    print(' ** Quadratic Estimates **')
    #print('I_MCQ estimates', mcq_estimates)
    print('I_MCQ variance', mcq_estimates_var)
    print('I_MCQ MSE', mcq_estimates_mse)
    print('-' * 20)
    print(' ** simple Quadratic **')
    print('I_Q', float(simple_quadratic_estimate))
    print('I_Q MSE', (simple_quadratic_estimate - baseline_estimate) ** 2)
    print(' ** smart Quadratic **')
    print('I_SQ', float(smart_quadratic_estimate))
    print('I_SQ MSE', (smart_quadratic_estimate - baseline_estimate) ** 2)
    print('-' * 20)
    if p == 11:
        print('Delta_opt', Delta_opt)
        print('-' * 20)

# Plots
    fig, ax = plt.subplots(1, 4, sharex=True, figsize=(10, 4))
    ax[0].axvline(x=baseline_estimate, color='b', label='axvline - full height')
    ax[0].hist(mc_estimates, 20, density=False, facecolor='g', alpha=0.75)
    ax[0].set_ylabel("freq")
    ax[0].set_xlabel("est")
    ax[0].set_title("MC histogram")
    ax[0].legend(['baseline', 'MC'])

    ax[1].axvline(x=baseline_estimate, color='b', label='axvline - full height')
    ax[1].hist(mc2_estimates, 20, density=False, facecolor='g', alpha=0.75)
    ax[1].set_ylabel("freq")
    ax[1].set_xlabel("est")
    ax[1].set_title("MC2 histogram, p={:.1f}".format(p * 0.1))
    ax[1].legend(['baseline', 'MC2'])

    ax[2].axvline(x=baseline_estimate, color='b', label='axvline - full height')
    ax[2].hist(mcl_estimates, 20, density=False, facecolor='g', alpha=0.75)
    ax[2].set_ylabel("freq")
    ax[2].set_xlabel("est")
    ax[2].set_title("MCL histogram, p={:.1f}".format(p * 0.1))
    ax[2].legend(['baseline', 'MCL'])

    ax[3].axvline(x=baseline_estimate, color='b', label='axvline - full height')
    ax[3].axvline(x=simple_quadratic_estimate, color='r', label='axvline - full height')
    ax[3].axvline(x=smart_quadratic_estimate, color='m', label='axvline - full height')
    ax[3].hist(mcq_estimates, 20, density=False, facecolor='g', alpha=0.75)
    ax[3].set_ylabel("freq")
    ax[3].set_xlabel("est")
    ax[3].set_title("MCQ histogram, p={:.1f}".format(p * 0.1))
    ax[3].legend(['baseline', 'simple Q', 'MCQ'])

    plt.savefig('../output/histogram_{:.1f}.png'.format(p * 0.1))

plt.show()
