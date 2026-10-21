import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')

import time

from jax import numpy as jnp
from jax import random as jr
from jax import lax, vmap, pmap, jacfwd, jacrev, debug, device_put, jit
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import List, Optional, NamedTuple, Union
from jaxtyping import Array, Float, Int

# import tensorflow as tf
# tf.test.gpu_device_name()

import gaussfiltax.utils as utils
import gaussfiltax.inference as gf
from gaussfiltax.inference import ParamsUKF, _ukf_predict_nonadditive, PosteriorGaussianSumFiltered, _process_input, _get_params, swap_axes_on_values
from gaussfiltax.models import ParamsNLSSM, NonlinearSSM, ParamsBPF


import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
# Parameters
state_dim = 4
state_noise_dim = 4
emission_dim = 4
emission_noise_dim = 4
seq_length = 200
mu0 = 0.0 * jnp.zeros(state_dim)
q0 = jnp.zeros(state_noise_dim)
r0 = 0.0001 * jnp.ones(emission_noise_dim)
Sigma0 = 1.0 * jnp.eye(state_dim)
Q = 10.0 * jnp.eye(state_noise_dim)
R = 4 * 1e-3 * jnp.eye(emission_noise_dim)

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
inputs = vmap(lambda x: jnp.sin(0.1 * x)**2)(jnp.arange(seq_length)) # sine_sq
# inputs = vmap(lambda x: (1 - jnp.sin(0.1 * x))/2)(jnp.arange(seq_length)) # sine
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
Nsim = 20
# rho1, rho2 = jnp.meshgrid(0.99 * jnp.arange(1, 2), 0.1 * jnp.arange(1, 10))
rho_values = jnp.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]) #  
# rho_values = jnp.array([1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1-1e-4])
rho1, rho2 = jnp.meshgrid(rho_values, rho_values)
opt_arg_pairs = jnp.array([rho1.ravel(), rho2.ravel()]).T

gsf_mse = jnp.zeros(Nsim)
ugsf_mse = jnp.zeros(Nsim)
agsf_mse = jnp.zeros((Nsim, len(opt_arg_pairs)))
uagsf_mse = jnp.zeros((Nsim, len(opt_arg_pairs)))
bpf_mse = jnp.zeros(Nsim)

gsf_norm = jnp.zeros((Nsim, seq_length))
ugsf_norm = jnp.zeros((Nsim, seq_length))
agsf_norm = jnp.zeros((Nsim, seq_length, len(opt_arg_pairs)))
uagsf_norm = jnp.zeros((Nsim, seq_length, len(opt_arg_pairs)))
bpf_norm = jnp.zeros((Nsim, seq_length))

gsf_time = jnp.zeros(Nsim)
ugsf_time = jnp.zeros(Nsim)
agsf_time = jnp.zeros(Nsim)
uagsf_time = jnp.zeros(Nsim)
bpf_time = jnp.zeros(Nsim)
gspf_time = jnp.zeros(Nsim)
next_key = jr.PRNGKey(10)


for i in range(Nsim):
    print('sim {}/{}'.format(i+1, Nsim))
    # Generate Data
    key0, key, next_key = jr.split(next_key, 3)
    print('key0: ', key0)
    print('key: ', key)
    states, emissions = model.sample(params, key0, seq_length, inputs = inputs)

    M = 50 
    # GSF   
    # tin = time.time()
    # posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, M, 1, inputs)
    # point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)
    # tout = time.time()
    # t_gsf= tout - tin
    # print('       Time taken for GSF: ', tout - tin)

    # U-GSF
    tin = time.time()
    uparams = ParamsUKF(1,0,0)
    posterior_filtered_ugsf = gf.unscented_gaussian_sum_filter(params, uparams, emissions, M, 1, inputs)
    point_estimate_ugsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_ugsf.means, posterior_filtered_ugsf.weights), axis=0)
    tout = time.time()
    t_ugsf= tout - tin
    print('       Time taken for UGSF: ', tout - tin)

    num_components = [50, 2, 2]

    # AGSF
    # tin = time.time()
    # posterior_filtered_agsf, aux_outputs = vmap(gf.speedy_augmented_gaussian_sum_filter, in_axes=(None, None, None, None, None, 0, None))(params, emissions, num_components, key, 1, opt_arg_pairs, inputs)    
    # point_estimates_agsf = vmap(lambda posterior_filtered: jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered.means, posterior_filtered.weights), axis=0))(posterior_filtered_agsf)
    # tout = time.time()
    # t_agsf= tout - tin
    # print('       Time taken for AGSF: ', tout - tin)

    # U-AGSF
    tin = time.time()
    posterior_filtered_uagsf, aux_outputs = vmap(gf.speedy_unscented_agsf, in_axes=(None, None, None, None, None, 0, None))(params, uparams, emissions, num_components, key, opt_arg_pairs, inputs)
    point_estimates_uagsf = vmap(lambda posterior_filtered: jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered.means, posterior_filtered.weights), axis=0))(posterior_filtered_uagsf)
    tout = time.time()
    t_uagsf = tout - tin
    print('       Time taken for UAGSF: ', tout - tin)

    # BPF
    tin = time.time()
    num_particles = 50

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

    posterior_bpf = gf.bootstrap_particle_filter(params_bpf, emissions, num_particles, key, inputs, 1.0)
    point_estimate_bpf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_bpf["particles"], posterior_bpf["weights"]), axis=0)
    tout = time.time()
    t_bpf = tout - tin
    print('       Time taken for BPF: ', tout - tin)


    # Computation of errors
    gsf_norm = gsf_norm.at[i].set(jnp.linalg.norm(point_estimate_ugsf - states, axis = 1)**2)
    agsf_norm = agsf_norm.at[i].set(vmap(lambda point_estimate: jnp.linalg.norm(point_estimate[:] - states[:], axis = 1)**2)(point_estimates_uagsf).T)
    bpf_norm = bpf_norm.at[i].set(jnp.linalg.norm(point_estimate_bpf - states, axis = 1)**2)

    gsf_mse = gsf_mse.at[i].set(jnp.mean(gsf_norm[i]))
    agsf_mse = agsf_mse.at[i].set(jnp.mean(agsf_norm[i], axis = 0))
    bpf_mse = bpf_mse.at[i].set(jnp.mean(bpf_norm[i]))


import pandas as pd

# Remove NaNs and compute success percentage
ind_gsf = jnp.unique(jnp.argwhere(jnp.isnan(gsf_mse)))
gsf_mse = jnp.delete(gsf_mse, ind_gsf, axis = 0)
gsf_success_pct = int(100 - 100*len(ind_gsf) / Nsim)

ind_ugsf = jnp.unique(jnp.argwhere(jnp.isnan(ugsf_mse)))
# ugsf_rmse = jnp.delete(ugsf_rmse, ind_ugsf, axis = 0)
ugsf_success_pct = int(100 - 100* len(ind_ugsf) / Nsim)

ind_bpf = jnp.unique(jnp.argwhere(jnp.isnan(bpf_mse)))
bpf_rmse = jnp.delete(bpf_mse, ind_bpf, axis = 0)
bpf_success_pct = int(100 - 100*len(ind_bpf) / Nsim)


keys = jr.split(jr.PRNGKey(0), 5)
B = 100
gsf_boot = utils.bootstrap(keys[0], gsf_mse, B)
ugsf_boot = utils.bootstrap(keys[1], ugsf_mse, B)
bpf_boot = utils.bootstrap(keys[2], bpf_rmse, B)
pct_agsf, armse_agsf, std_agf = utils.vec_clean_and_rmse(agsf_mse, keys[3], B, Nsim)
# pct_uagsf, armse_uagsf, std_uagf = utils.vec_clean_and_rmse(uagsf_rmse, keys[4], B, Nsim)

# gsf_armse = jnp.mean(gsf_boot)
# ugsf_armse = jnp.mean(ugsf_boot)
# bpf_armse = jnp.mean(bpf_boot)

# gsf_atime = jnp.mean(gsf_time)
# ugsf_atime = jnp.mean(ugsf_time)
# agsf_atime = jnp.mean(agsf_time)
# uagsf_atime = jnp.mean(uagsf_time)
# bpf_atime = jnp.mean(bpf_time)

# gsf_tab_out = '{:2.3f}±{:2.2f}({}%)'.format(gsf_armse, jnp.std(gsf_rmse), gsf_success_pct)
# # ugsf_tab_out = '{:2.3f}±{:2.2f}({}%)'.format(ugsf_armse, jnp.std(ugsf_rmse), ugsf_success_pct)
# bpf_tab_out = '{:2.3f}±{:2.2f}({}%)'.format(bpf_armse, jnp.std(bpf_boot), bpf_success_pct)
# agsf_tab_out = ['{:2.3f}±{:2.2f}({}%)'.format(armse_agsf[i], std_agf[i], pct_agsf[i]) for i in range(len(armse_agsf))]
# uagsf_tab_out = ['{:2.3f}±{:2.2f}({}%)'.format(armse_uagsf[i], std_uagf[i], pct_uagsf[i]) for i in range(len(armse_uagsf))]
print(opt_arg_pairs)
print(jnp.array(armse_agsf))
print(pct_agsf)
print(opt_arg_pairs[jnp.argmin(jnp.array(armse_agsf))])
print(jnp.min(jnp.array(armse_agsf)))
print(jnp.mean(gsf_mse))
print(gsf_success_pct)
print(jnp.mean(bpf_rmse))
print(bpf_success_pct)
import seaborn as sns
import matplotlib.pylab as plt

data = jnp.array(armse_agsf).reshape(len(rho_values), len(rho_values))
vmin = jnp.min(data) 
vmax = jnp.max(data) 
ax = sns.heatmap(data, cmap="YlGnBu", linewidth=.5, vmin = vmin, vmax = vmax)
ax.set_xticklabels(rho_values)
ax.set_yticklabels(rho_values)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle


# Sample data for demonstration purposes
# Replace this with your actual data

# Define a custom colormap
colors = ["#08306b", "#4292c6", "#9ecae1", "#c6dbef", "#f7fbff"]
n_bins = 100  # Discretize the interpolation into bins
cmap_name = "custom_blue"

# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
colormap = sns.color_palette("YlGnBu", as_cmap=True)


# Normalize the data to emphasize the middle range
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a simple example...
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        result = np.ma.masked_array(np.interp(result, [vmin, midpoint, vmax], [0, 0.5, 1.0]), np.isnan(result))
        if is_scalar:
            result = result[0]
        return result

# Creating the heatmap
vmin = jnp.min(data) # jnp.min(jnp.array([jnp.min(data), jnp.min(gsf_mse), jnp.min(bpf_mse)]))
vmax = jnp.max(data) # jnp.max(jnp.array([jnp.max(data), jnp.max(gsf_mse), jnp.max(bpf_mse)]))
kappa = 0.8 # Adjust the value of kappa as needed
norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint= vmin + kappa * (vmax - vmin))  # Adjust the midpoint as needed
ax = sns.heatmap(jnp.flip(data, 0), cmap='YlGnBu', linewidth=.5, norm=norm)

# Setting the labels for x and y ticks
ax.set_xticklabels(rho_values)
ax.set_yticklabels(jnp.flip(rho_values, 0))
ax.set_xlabel(r'$\rho_1$')
ax.set_ylabel(r'$\rho_2$')
ax.collections[0].colorbar.set_label('MSE')
ax.collections[0].colorbar.set_ticks([5.0, 10.0, 15.0, 20.0, 23.0, 25.0]) 
# ax.collections[0].colorbar.set_ticklabels([5.0, 12.5, 20.0, 25.0, 27.5])

circle_gsf_color = colormap(jnp.mean(gsf_mse))
circle_bpf_color = colormap(jnp.mean(bpf_mse))
circle_gsf = Circle((1.0, 0.0), 0.02, transform=ax.transAxes, color=circle_gsf_color, zorder=10)
circle_bpf = Circle((0.0, 1.0), 0.02, transform=ax.transAxes, color=circle_bpf_color, zorder=10)

# ax.add_patch(circle_gsf)
# ax.add_patch(circle_bpf)

# Display the plot
plt.show()
plt.savefig(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/output/heatmap.eps', format='eps')

