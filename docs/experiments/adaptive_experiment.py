import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')

from jax import numpy as jnp
from jax import random as jr
from jax import vmap

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


#Helper functions
def bootstrap(key, rmse_array, B):
    N = rmse_array.shape[0]
    rmse_boot = jnp.zeros((B,))
    for b in range(B):
        key, subkey = jr.split(key)
        ind = jr.randint(subkey, (N,), 0, N)
        rmse_boot = rmse_boot.at[b].set(jnp.mean(rmse_array[ind]))
    return rmse_boot

# Parameters
state_dim = 10
state_noise_dim = 10
emission_dim = 10
emission_noise_dim = 10
seq_length = 200
mu0 = 0.0 * jnp.zeros(state_dim)
q0 = jnp.zeros(state_noise_dim)
r0 = 0.0001 * jnp.ones(emission_noise_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 10.0 * jnp.eye(state_noise_dim)
R = 0.1 * jnp.eye(emission_noise_dim)

# Model functions
Phi = 0.8 * jnp.eye(state_dim)
fmsv = lambda x, q, u: Phi @ x + q #((1-u) + 0.3*u)*q

sigma = 5.0
beta = 0.5
H0 = jnp.eye(emission_dim, state_dim)
glmsv = lambda x, r, u: u * beta * jnp.multiply(jnp.exp(x / sigma), r) + (1-u) * (H0 @ x + r)
def lmsvlp(x,y,u):
    M = u * beta * jnp.diag(jnp.exp(x / sigma)) + (1-u) * jnp.eye(emission_dim)
    return MVN(loc = glmsv(x, r0, u), covariance_matrix = M @ R @ M.T).log_prob(y)

# Inputs
# inputs = jnp.array([0]*int(seq_length/2) + [1]*int(seq_length/2)) # off - on
inputs = vmap(lambda x: jnp.sin(0.1 * x)**2)(jnp.arange(seq_length)) # sine
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


verbose = False
Nsim = 5
gsf_rmse = jnp.zeros(Nsim)
ugsf_rmse = jnp.zeros(Nsim)
agsf_rmse = jnp.zeros(Nsim)
uagsf_rmse = jnp.zeros(Nsim)

bpf_rmse = jnp.zeros(Nsim)
gsf_norm = jnp.zeros((Nsim, seq_length))
ugsf_norm = jnp.zeros((Nsim, seq_length))
agsf_norm = jnp.zeros((Nsim, seq_length))
uagsf_norm = jnp.zeros((Nsim, seq_length))
bpf_norm = jnp.zeros((Nsim, seq_length))

gsf_time = jnp.zeros(Nsim)
ugsf_time = jnp.zeros(Nsim)
agsf_time = jnp.zeros(Nsim)
uagsf_time = jnp.zeros(Nsim)
bpf_time = jnp.zeros(Nsim)

gsf_bin_error = jnp.zeros(Nsim)
agsf_bin_error = jnp.zeros(Nsim)
uagsf_bin_error = jnp.zeros(Nsim)
bpf_bin_error = jnp.zeros(Nsim)


next_key = jr.PRNGKey(0)
for i in range(Nsim):
    print('sim {}/{}'.format(i+1, Nsim))
    # Generate Data
    keys = jr.split(next_key, 5)
    next_key = keys[4]
    states, emissions = model.sample(params, keys[0], seq_length, inputs = inputs)

    # GSF
    M = 1000
    tin = time.time()
    posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, M, 1, inputs)
    point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)
    t_gsf = time.time() - tin
    print('       Time taken for GSF: ', t_gsf)

    # # U-GSF
    uparams = ParamsUKF(1,0,0)
    tin = time.time()
    posterior_filtered_ugsf = gf.gaussian_sum_filter(params, emissions, M, 1, inputs)
    point_estimate_ugsf =  jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_ugsf.means, posterior_filtered_ugsf.weights), axis=0)
    tout = time.time()
    t_ugsf = tout - tin
    print('       Time taken for UGSF: ', t_ugsf)

    # AGSF
    opt_args = (0.8, 0.1)
    tin = time.time()
    num_components = [10, 2, 2]
    posterior_filtered_agsf, aux_outputs = gf.speedy_augmented_gaussian_sum_filter(params, emissions, num_components, rng_key = keys[1], opt_args = opt_args, inputs=inputs)    
    point_estimate_agsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_agsf.means, posterior_filtered_agsf.weights), axis=0)
    t_agsf = time.time() - tin
    print('       Time taken for AGSF: ', t_agsf)

    # # U-AGSF
    tin = time.time()
    posterior_filtered_uagsf, aux_outputs = gf.speedy_augmented_gaussian_sum_filter(params, emissions, num_components, rng_key = keys[2], opt_args = opt_args, inputs=inputs)
    point_estimate_uagsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_uagsf.means, posterior_filtered_uagsf.weights), axis=0)
    tout = time.time()
    t_uagsf = tout - tin
    print('       Time taken for UAGSF: ', t_uagsf)

    # BPF
    tin = time.time()
    num_particles = 10000

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

    posterior_bpf = gf.bootstrap_particle_filter(params_bpf, emissions, num_particles, keys[3], inputs)
    point_estimate_bpf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_bpf["particles"], posterior_bpf["weights"]), axis=0)
    t_bpf  = time.time() - tin
    print('       Time taken for BPF: ', t_bpf)

    # Computation of errors
    gsf_rmse = gsf_rmse.at[i].set(utils.rmse(point_estimate_gsf, states))
    ugsf_rmse = ugsf_rmse.at[i].set(utils.rmse(point_estimate_ugsf, states))
    agsf_rmse = agsf_rmse.at[i].set(utils.rmse(point_estimate_agsf, states))
    uagsf_rmse = uagsf_rmse.at[i].set(utils.rmse(point_estimate_uagsf, states))
    bpf_rmse = bpf_rmse.at[i].set(utils.rmse(point_estimate_bpf, states))

    ## Compute average binned error
    gsf_bin_error = gsf_bin_error.at[i].set((utils.rmse(point_estimate_gsf[0:int(seq_length/2)], states[0:int(seq_length/2)]) \
                                            + utils.rmse(point_estimate_gsf[int(seq_length/2)+1:], states[int(seq_length/2)+1:])) / 2)
    agsf_bin_error = agsf_bin_error.at[i].set((utils.rmse(point_estimate_agsf[0:int(seq_length/2)], states[0:int(seq_length/2)]) \
                                            + utils.rmse(point_estimate_agsf[int(seq_length/2)+1:], states[int(seq_length/2)+1:])) / 2)
    uagsf_bin_error = uagsf_bin_error.at[i].set((utils.rmse(point_estimate_uagsf[0:int(seq_length/2)], states[0:int(seq_length/2)]) \
                                            + utils.rmse(point_estimate_uagsf[int(seq_length/2)+1:], states[int(seq_length/2)+1:])) / 2)
    bpf_bin_error = bpf_bin_error.at[i].set((utils.rmse(point_estimate_bpf[0:int(seq_length/2)], states[0:int(seq_length/2)]) \
                                            + utils.rmse(point_estimate_bpf[int(seq_length/2)+1:], states[int(seq_length/2)+1:])) / 2)


    gsf_time = gsf_time.at[i].set(t_gsf)
    ugsf_time = ugsf_time.at[i].set(t_ugsf)
    agsf_time = agsf_time.at[i].set(t_agsf)
    uagsf_time = uagsf_time.at[i].set(t_uagsf)
    bpf_time = bpf_time.at[i].set(t_bpf)

    print('              GSF RMSE:', gsf_rmse[i])
    print('              UGSF RMSE:', ugsf_rmse[i])                                                                           
    print('              AGSF RMSE:', agsf_rmse[i])
    print('              UAGSF RMSE:', uagsf_rmse[i])
    print('              BPF RMSE:', bpf_rmse[i])

    gsf_norm = gsf_norm.at[i].set(jnp.linalg.norm(point_estimate_gsf - states, axis = 1))
    ugsf_norm = ugsf_norm.at[i].set(jnp.linalg.norm(point_estimate_ugsf - states, axis = 1))
    agsf_norm = agsf_norm.at[i].set(jnp.linalg.norm(point_estimate_agsf - states, axis = 1))
    uagsf_norm = uagsf_norm.at[i].set(jnp.linalg.norm(point_estimate_uagsf - states, axis = 1))
    bpf_norm = bpf_norm.at[i].set(jnp.linalg.norm(point_estimate_bpf - states, axis = 1))

ind = jnp.argwhere(jnp.isnan(gsf_norm[:,99])).flatten()
gsf_norm = jnp.delete(gsf_norm, ind, axis = 0)     

plt.figure(figsize=(10, 7))
plt.plot(gsf_norm.sum(axis=0)/(Nsim-len(ind)) , label = 'L-GSF')
# plt.plot(ugsf_norm.sum(axis=0)/Nsim, label = 'UGSF')
plt.plot(agsf_norm.sum(axis=0)/Nsim, label = 'L-AGSF')
plt.plot(uagsf_norm.sum(axis=0)/Nsim, label = 'U-AGSF')
plt.plot(bpf_norm.sum(axis=0)/Nsim, label = 'BPF')
# plt.axvline(x=seq_length/2, color = 'gray', linestyle = 'dashed', label = 'change-point')
plt.ylabel('error', fontsize=15)
# plt.xlabel('time', fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=0)
plt.legend(prop = { "size": 15 })
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/error.eps', format='eps')

# plot Lambdas
plt.figure(figsize=(10, 2))
plt.plot(jnp.trace(aux_outputs["Lambdas"], axis1=2, axis2=3).sum(axis=1)/num_components[0]*num_components[1]/state_dim**2)
# plt.axvline(x=seq_length/2, color = 'gray', linestyle = 'dashed', label = 'change-point')
plt.ylabel('$tr(\Lambda)$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('time', fontsize=15)
plt.legend(prop = { "size": 15 })
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/lambdas.eps', format='eps')

import pandas as pd

keys = jr.split(jr.PRNGKey(0), 5)
B = 100
gsf_boot = bootstrap(keys[0], gsf_rmse, B)
ugsf_boot = bootstrap(keys[1], ugsf_rmse, B)
agsf_boot = bootstrap(keys[2], agsf_rmse, B)
uagsf_boot = bootstrap(keys[3], uagsf_rmse, B)
bpf_boot = bootstrap(keys[4], bpf_rmse, B)

gsf_bin_boot = bootstrap(keys[0], gsf_bin_error, B)
agsf_bin_boot = bootstrap(keys[1], agsf_bin_error, B)
uagsf_bin_boot = bootstrap(keys[2], uagsf_bin_error, B)
bpf_bin_boot = bootstrap(keys[3], bpf_bin_error, B)

gsf_armse = jnp.mean(gsf_boot)
ugsf_armse = jnp.mean(ugsf_boot)
agsf_armse = jnp.mean(agsf_boot)
uagsf_armse = jnp.mean(uagsf_boot)
bpf_armse = jnp.mean(bpf_boot)

gsf_bin_armse = jnp.mean(gsf_bin_boot)
agsf_bin_armse = jnp.mean(agsf_bin_boot)
uagsf_bin_armse = jnp.mean(uagsf_bin_boot)
bpf_bin_armse = jnp.mean(bpf_bin_boot)

gsf_time_boot = bootstrap(keys[0], gsf_time, B)
ugsf_time_boot = bootstrap(keys[1], ugsf_time, B)
agsf_time_boot = bootstrap(keys[2], agsf_time, B)
uagsf_time_boot = bootstrap(keys[3], uagsf_time, B)
bpf_time_boot = bootstrap(keys[4], bpf_time, B)

gsf_atime = jnp.mean(gsf_time_boot)
ugsf_atime = jnp.mean(ugsf_time_boot)
agsf_atime = jnp.mean(agsf_time_boot)
uagsf_atime = jnp.mean(uagsf_time_boot)
bpf_atime = jnp.mean(bpf_time_boot)

gsf_tab_out = '{:2.3f}±{:2.3f}'.format(gsf_armse, jnp.std(gsf_boot))
ugsf_tab_out = '{:2.3f}±{:2.3f}'.format(ugsf_armse, jnp.std(ugsf_boot))
agsf_tab_out = '{:2.3f}±{:2.3f}'.format(agsf_armse, jnp.std(agsf_boot))
uagsf_tab_out = '{:2.3f}±{:2.3f}'.format(uagsf_armse, jnp.std(uagsf_boot))
bpf_tab_out = '{:2.3f}±{:2.3f}'.format(bpf_armse, jnp.std(bpf_boot))

gsf_bin_tab_out = '{:2.3f}±{:2.3f}'.format(gsf_bin_armse, jnp.std(gsf_bin_boot))
agsf_bin_tab_out = '{:2.3f}±{:2.3f}'.format(agsf_bin_armse, jnp.std(agsf_bin_boot))
uagsf_bin_tab_out = '{:2.3f}±{:2.3f}'.format(uagsf_bin_armse, jnp.std(uagsf_bin_boot))
bpf_bin_tab_out = '{:2.3f}±{:2.3f}'.format(bpf_bin_armse, jnp.std(bpf_bin_boot))

gsf_tab_out1 = '{:2.2f}±{:2.2f}'.format(gsf_atime, jnp.std(gsf_time_boot))
ugsf_tab_out1 = '{:2.2f}±{:2.2f}'.format(ugsf_atime, jnp.std(ugsf_time_boot))
agsf_tab_out1 = '{:2.2f}±{:2.2f}'.format(agsf_atime, jnp.std(agsf_time_boot))
uagsf_tab_out1 = '{:2.2f}±{:2.2f}'.format(uagsf_atime, jnp.std(uagsf_time_boot))
bpf_tab_out1 = '{:2.2f}±{:2.2f}'.format(bpf_atime, jnp.std(bpf_time_boot))

df = pd.DataFrame(columns = [' ','RMSE', 'Bin RMSE','time(s)'])
df[' '] = ['GSF','UGSF', 'AGSF', 'UAGSF', 'BPF']
df['RMSE'] = [gsf_tab_out, ugsf_tab_out, agsf_tab_out, uagsf_tab_out, bpf_tab_out]
df['Bin RMSE'] = [gsf_bin_tab_out, 0, agsf_bin_tab_out, uagsf_bin_tab_out, bpf_bin_tab_out]
df['time(s)'] = [gsf_tab_out1, ugsf_tab_out1, agsf_tab_out1, uagsf_tab_out1, bpf_tab_out1]
print(df.to_latex(index=False))

