import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')

from jax import numpy as jnp
from jax import random as jr
from jax import vmap
import pandas as pd

import gaussfiltax.utils as utils
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf
from gaussfiltax.inference import ParamsUKF
from gaussfiltax.containers import num_prt1, num_prt2

from gaussfiltax.models import ParamsNLSSM, NonlinearSSM, ParamsBPF

import scienceplots
import matplotlib.pyplot as plt
import matplotlib_inline

plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


# Parameters
state_dim = 10
state_noise_dim = 10
emission_dim = 10
emission_noise_dim = 10
seq_length = 140
mu0 = 0.0 * jnp.zeros(state_dim)
q0 = jnp.zeros(state_noise_dim)
r0 = 0.0001 * jnp.ones(emission_noise_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 10.0 * jnp.eye(state_noise_dim)
R = 1e-1 * jnp.eye(emission_noise_dim)

# Model functions
Phi = 0.8 * jnp.eye(state_dim)
fmsv = lambda x, q, u: Phi @ x + q

sigma = 4.0
beta = 0.5
H0 = jnp.eye(emission_dim, state_dim)
glmsv = lambda x, r, u: u * beta * jnp.multiply(jnp.exp(x / sigma), r) + (1-u) * (H0 @ x + r)

def lmsvlp(x,y,u):
    M = u * beta * jnp.diag(jnp.exp(x / sigma)) + (1-u) * jnp.eye(emission_dim)
    return MVN(loc = glmsv(x, r0, u), covariance_matrix = M @ R @ M.T).log_prob(y)

# Inputs
# inputs = jnp.array([0]*int(seq_length/2) + [1]*int(seq_length/2)) # off - on
# inputs = vmap(lambda x: jnp.sin(0.1 * x)**2)(jnp.arange(seq_length)) # sine_sq
inputs = vmap(lambda x: (1 - jnp.sin(0.2 * x))/2)(jnp.arange(seq_length)) # sine
# inputs = jr.uniform(jr.PRNGKey(0), (seq_length,))
# inputs = jnp.zeros(seq_length) # off
# inputs = jnp.ones(seq_length) # on
# plt.plot(inputs)
# plt.show()

f = fmsv
g = glmsv
glp = lmsvlp


# initialization
model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)

params = ParamsNLSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias= q0,
    dynamics_noise_covariance=Q,
    emission_function=g,
    emission_noise_bias= r0,
    emission_noise_covariance=R,
)

params_bpf = ParamsBPF(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias=q0,
    dynamics_noise_covariance=Q,
    emission_function=g,
    emission_noise_bias=r0,
    emission_noise_covariance=R,
    emission_distribution_log_prob = glp
)

verbose = False
Nsim = 20
gsf_mse = jnp.zeros(Nsim)
ugsf_mse = jnp.zeros(Nsim)
agsf_mse = jnp.zeros(Nsim)
uagsf_mse = jnp.zeros(Nsim)
bpf_mse = jnp.zeros(Nsim)
apf_mse = jnp.zeros(Nsim)
gspf_mse = jnp.zeros(Nsim)

gsf_norm = jnp.zeros((Nsim, seq_length))
ugsf_norm = jnp.zeros((Nsim, seq_length))
agsf_norm = jnp.zeros((Nsim, seq_length))
uagsf_norm = jnp.zeros((Nsim, seq_length))
bpf_norm = jnp.zeros((Nsim, seq_length))
apf_norm = jnp.zeros((Nsim, seq_length))
gspf_norm = jnp.zeros((Nsim, seq_length))

gsf_time = jnp.zeros(Nsim)
ugsf_time = jnp.zeros(Nsim)
agsf_time = jnp.zeros(Nsim)
uagsf_time = jnp.zeros(Nsim)
bpf_time = jnp.zeros(Nsim)
apf_time = jnp.zeros(Nsim)
gspf_time = jnp.zeros(Nsim)

next_key = jr.PRNGKey(0)

for i in range(Nsim):

    print('sim {}/{}'.format(i+1, Nsim))
    # Generate Data
    print('next_key: ', next_key)
    key0, key, next_key = jr.split(next_key, 3)
    print('key0: ', key0)
    print('key: ', key)
    states, emissions = model.sample(params, key0, seq_length, inputs = inputs)

    # GSF
    M = 10
    tin = time.time()
    posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, M, 1, key, inputs)
    point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)
    tout = time.time()
    t_gsf= tout - tin
    print('       Time taken for GSF: ', tout - tin)

    # U-GSF
    tin = time.time()
    uparams = ParamsUKF(1,0,0)
    posterior_filtered_ugsf = gf.unscented_gaussian_sum_filter(params, uparams, emissions, M, 1, key, inputs)
    point_estimate_ugsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_ugsf.means, posterior_filtered_ugsf.weights), axis=0)
    tout = time.time()
    t_ugsf= tout - tin
    print('       Time taken for UGSF: ', tout - tin)

    # AGSF
    opt_args = (0.9, 1e-6)
    num_components = [100, 5, 5]
    tin = time.time()
    posterior_filtered_agsf, aux_outputs_agsf = gf.speedy_augmented_gaussian_sum_filter(params, emissions, num_components, rng_key = key, opt_args = opt_args, inputs=inputs)    
    point_estimate_agsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_agsf.means, posterior_filtered_agsf.weights), axis=0)
    tout = time.time()
    t_agsf= tout - tin
    print('       Time taken for AGSF: ', tout - tin)

    # U-AGSF
    tin = time.time()
    # opt_args = (0.9, 1.0)
    # posterior_filtered_uagsf, aux_outputs = gf.unscented_agsf(params, uparams, emissions, num_components, rng_key = key, opt_args = opt_args, inputs=inputs)
    posterior_filtered_uagsf, aux_outputs_uagsf = gf.speedy_unscented_agsf(params, uparams, emissions, num_components, key, opt_args, inputs=inputs)
    point_estimate_uagsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_uagsf.means, posterior_filtered_uagsf.weights), axis=0)
    tout = time.time()
    t_uagsf= tout - tin
    print('       Time taken for UAGSF: ', tout - tin)

    # BPF
    tin = time.time()
    num_particles = 50000
    posterior_bpf = gf.bootstrap_particle_filter(params_bpf, emissions, num_particles, key, inputs)
    point_estimate_bpf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_bpf["particles"], posterior_bpf["weights"]), axis=0)
    tout = time.time()
    t_bpf = tout - tin
    print('       Time taken for BPF: ', tout - tin)

    # APF
    # tin = time.time()
    # posterior_apf = gf.auxiliary_particle_filter(params_bpf, emissions, num_particles, key, inputs)
    # point_estimate_apf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_apf["particles"], posterior_apf["weights"]), axis=0)
    # tout = time.time()
    # t_apf = tout - tin
    # print('       Time taken for APF: ', tout - tin)

    # GS-PF
    # num_comp = 1
    # num_prt = 100
    # tin = time.time()
    # posterior_gspf = gf.gaussian_sum_particle_filter(params_bpf, emissions, num_comp, num_prt, key, inputs)
    # point_estimate_gspf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_gspf.means, posterior_gspf.weights), axis=0)
    # tout = time.time()
    # t_gspf = tout - tin
    # print('       Time taken for GSPF: ', tout - tin)


    # Computation of errors

    gsf_norm = gsf_norm.at[i].set(jnp.linalg.norm(point_estimate_gsf[:,(0,2)] - states[:,(0,2)], axis = 1))
    ugsf_norm = ugsf_norm.at[i].set(jnp.linalg.norm(point_estimate_ugsf[:,(0,2)] - states[:,(0,2)], axis = 1))
    agsf_norm = agsf_norm.at[i].set(jnp.linalg.norm(point_estimate_agsf[:,(0,2)] - states[:,(0,2)], axis = 1))
    uagsf_norm = uagsf_norm.at[i].set(jnp.linalg.norm(point_estimate_uagsf[:,(0,2)] - states[:,(0,2)], axis = 1))
    bpf_norm = bpf_norm.at[i].set(jnp.linalg.norm(point_estimate_bpf[:,(0,2)] - states[:,(0,2)], axis = 1))
    # apf_norm = apf_norm.at[i].set(jnp.linalg.norm(point_estimate_apf[:,(0,2)] - states[:,(0,2)], axis = 1))
    # gspf_norm = gspf_norm.at[i].set(jnp.linalg.norm(point_estimate_gspf[:,(0,2)] - states[:,(0,2)], axis = 1))

    gsf_mse = gsf_mse.at[i].set(jnp.mean(gsf_norm[i]))
    ugsf_mse = ugsf_mse.at[i].set(jnp.mean(ugsf_norm[i]))
    agsf_mse = agsf_mse.at[i].set(jnp.mean(agsf_norm[i]))
    uagsf_mse = uagsf_mse.at[i].set(jnp.mean(uagsf_norm[i]))
    bpf_mse = bpf_mse.at[i].set(jnp.mean(bpf_norm[i]))
    # apf_mse = apf_mse.at[i].set(jnp.mean(apf_norm[i]))
    # gspf_mse = gspf_mse.at[i].set(jnp.mean(gspf_norm[i]))

    gsf_time = gsf_time.at[i].set(t_gsf)
    ugsf_time = ugsf_time.at[i].set(t_ugsf)
    agsf_time = agsf_time.at[i].set(t_agsf)
    uagsf_time = uagsf_time.at[i].set(t_uagsf)
    bpf_time = bpf_time.at[i].set(t_bpf)
    # apf_time = apf_time.at[i].set(t_apf)
    # gspf_time = gspf_time.at[i].set(t_gspf)


# Remove NaNs and compute success percentage
ind_gsf = jnp.unique(jnp.argwhere(jnp.isnan(gsf_mse)))
gsf_mse = jnp.delete(gsf_mse, ind_gsf, axis = 0)
gsf_norm = jnp.delete(gsf_norm, ind_gsf, axis = 0)
gsf_success_pct = int(100 - 100*len(ind_gsf) / Nsim)

ind_ugsf = jnp.unique(jnp.argwhere(jnp.isnan(ugsf_mse)))
ugsf_mse = jnp.delete(ugsf_mse, ind_ugsf, axis = 0)
ugsf_norm = jnp.delete(ugsf_norm, ind_ugsf, axis = 0)
ugsf_success_pct = int(100 - 100* len(ind_ugsf) / Nsim)

ind_agsf = jnp.unique(jnp.argwhere(jnp.isnan(agsf_mse)))
agsf_mse = jnp.delete(agsf_mse, ind_agsf, axis = 0)
agsf_norm = jnp.delete(agsf_norm, ind_agsf, axis = 0)
agsf_success_pct = int(100 - 100*len(ind_agsf) / Nsim)

ind_uagsf = jnp.unique(jnp.argwhere(jnp.isnan(uagsf_mse)))
uagsf_mse = jnp.delete(uagsf_mse, ind_uagsf, axis = 0)
uagsf_norm = jnp.delete(uagsf_norm, ind_uagsf, axis = 0)
uagsf_success_pct = int(100 - 100*len(ind_uagsf) / Nsim)

ind_bpf = jnp.unique(jnp.argwhere(jnp.isnan(bpf_mse)))
bpf_mse = jnp.delete(bpf_mse, ind_bpf, axis = 0)
bpf_norm = jnp.delete(bpf_norm, ind_bpf, axis = 0)
bpf_success_pct = int(100 - 100*len(ind_bpf) / Nsim)

# ind_apf = jnp.unique(jnp.argwhere(jnp.isnan(apf_mse)))
# apf_mse = jnp.delete(apf_mse, ind_apf, axis = 0)
# apf_norm = jnp.delete(apf_norm, ind_apf, axis = 0)
# apf_success_pct = int(100 - 100*len(ind_apf) / Nsim)

# ind_gspf = jnp.unique(jnp.argwhere(jnp.isnan(gspf_mse)))
# gspf_mse = jnp.delete(gspf_mse, ind_gspf, axis = 0)
# gspf_norm = jnp.delete(gspf_norm, ind_gspf, axis = 0)
# gspf_success_pct = int(100 - 100*len(ind_gspf) / Nsim)


keys = jr.split(jr.PRNGKey(0), 7)
B = 100
gsf_boot = utils.bootstrap(keys[0], gsf_mse, B)
ugsf_boot = utils.bootstrap(keys[1], ugsf_mse, B)
agsf_boot = utils.bootstrap(keys[2], agsf_mse, B)
uagsf_boot = utils.bootstrap(keys[3], uagsf_mse, B)
bpf_boot = utils.bootstrap(keys[4], bpf_mse, B)
# apf_boot = utils.bootstrap(keys[5], apf_mse, B)
# gspf_boot = utils.bootstrap(keys[6], gspf_mse, B)

gsf_amse = jnp.mean(gsf_boot)
ugsf_amse = jnp.mean(ugsf_boot)
agsf_amse = jnp.mean(agsf_boot)
uagsf_amse = jnp.mean(uagsf_boot)
bpf_amse = jnp.mean(bpf_boot)
# apf_amse = jnp.mean(apf_boot)
# gspf_amse = jnp.mean(gspf_boot)

gsf_atime = jnp.mean(gsf_time)
ugsf_atime = jnp.mean(ugsf_time)
agsf_atime = jnp.mean(agsf_time)
uagsf_atime = jnp.mean(uagsf_time)
bpf_atime = jnp.mean(bpf_time)
# apf_atime = jnp.mean(apf_time)
# gspf_atime = jnp.mean(gspf_time)

gsf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(gsf_amse, jnp.std(gsf_boot), gsf_success_pct)
ugsf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(ugsf_amse, jnp.std(ugsf_boot), ugsf_success_pct)
agsf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(agsf_amse, jnp.std(agsf_boot), agsf_success_pct)
uagsf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(uagsf_amse, jnp.std(uagsf_boot), uagsf_success_pct)
bpf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(bpf_amse, jnp.std(bpf_boot), bpf_success_pct)
# apf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(apf_amse, jnp.std(apf_boot), apf_success_pct)
# gspf_tab_out = '${:2.3f}\pm{:2.3f}({}\%)$'.format(gspf_amse, jnp.std(gspf_boot), gspf_success_pct)

gsf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(gsf_atime, jnp.std(gsf_time))
ugsf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(ugsf_atime, jnp.std(ugsf_time))
agsf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(agsf_atime, jnp.std(agsf_time))
uagsf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(uagsf_atime, jnp.std(uagsf_time))
bpf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(bpf_atime, jnp.std(bpf_time))
# apf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(apf_atime, jnp.std(apf_time))
# gspf_tab_out1 = '${:2.2f}\pm{:2.2f}$'.format(gspf_atime, jnp.std(gspf_time))

df = pd.DataFrame(columns = [' ','RMSE','time(s)'])
# df[' '] = ['GSF', 'AGSF', 'AGSF Optimal', 'BPF']
# df['RMSE'] = [gsf_tab_out, agsf_tab_out, agsf_opt_tab_out, bpf_tab_out]
# df['time(s)'] = [gsf_tab_out1, agsf_tab_out1, agsf_opt_tab_out1, bpf_tab_out1]
df[' '] = ['GSF','UGSF', 'AGSF', 'UAGSF', 'BPF']
df['RMSE'] = [gsf_tab_out, ugsf_tab_out, agsf_tab_out, uagsf_tab_out, bpf_tab_out]
df['time(s)'] = [gsf_tab_out1, ugsf_tab_out1, agsf_tab_out1, uagsf_tab_out1, bpf_tab_out1]
print(df.to_latex(index=False))
df

plt.figure(figsize=(10, 7))
plt.plot(jnp.mean(gsf_norm, axis=0), linestyle = 'dashed', label = 'GSF')
plt.plot(jnp.mean(ugsf_norm, axis=0), linestyle = 'dotted', label = 'UGSF')
plt.plot(jnp.mean(agsf_norm, axis=0), label = 'L-AGSF')
plt.plot(jnp.mean(uagsf_norm, axis=0), label = 'U-AGSF')
plt.plot(jnp.mean(bpf_norm, axis=0), label = 'BPF')
# plt.plot(apf_norm.sum(axis=0)/Nsim, label = 'APF')
# plt.plot(gspf_norm.sum(axis=0)/Nsim, label = 'GS-PF')
# plt.axvline(x=seq_length/2, color = 'gray', linestyle = 'dashed', label = 'change-point')
plt.ylabel('error', fontsize=15)
# plt.xlabel('time', fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=0)
plt.legend(prop = { "size": 15 })
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/error.eps', format='eps')

# plot Lambdas
plt.figure(figsize=(10, 2))
plt.plot(jnp.mean(jnp.trace(aux_outputs_agsf['Deltas'], axis1=2, axis2=3),axis=1), label = 'Deltas AGSF')
plt.plot(jnp.mean(jnp.trace(aux_outputs_uagsf['Deltas'], axis1=2, axis2=3),axis=1), label = 'Deltas UAGSF')
plt.plot(jnp.mean(jnp.trace(aux_outputs_agsf['Lambdas'], axis1=2, axis2=3),axis=1), label = 'Lambdas AGSF')
plt.plot(jnp.mean(jnp.trace(aux_outputs_uagsf['Lambdas'], axis1=2, axis2=3),axis=1), label = 'Lambdas UAGSF')
plt.ylabel('$tr(\Lambda)$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('time', fontsize=15)
plt.legend(prop = { "size": 10 })
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/covs.eps', format='eps')

# plot rhos
plt.figure(figsize=(10, 2))
plt.plot(jnp.mean(aux_outputs_agsf['rho2s'],axis=1), label = r'AGSF')
plt.plot(jnp.mean(aux_outputs_uagsf['rho2s'],axis=1), label = r'UAGSF')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('time', fontsize=15)
plt.ylabel(r'$\rho_2$', fontsize=15)
plt.legend(prop = { "size": 11 })
# plt.axvline(x=seq_length/2, color = 'gray', linestyle = 'dashed', label = 'change-point')
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/rhos.eps', format='eps')