from ssm import StateSpaceModel
from ssm import LinearModelParameters
import numpy as np
import matplotlib.pyplot as plt


T = 100
Nsim = 10

dx = 1
dy = 1

## 1-D
params = LinearModelParameters(0, 0, 1, 1)
g = lambda x: x + np.random.normal(0, params.R)
jacob_obs = lambda x: 1


## n-D
# params = LinearModelParameters(np.eye(dx), np.eye(dy,dx), np.eye(dx), np.eye(dy))
# g = lambda x: np.matmul(x, params.H.T) + np.random.multivariate_normal(np.zeros(dy), np.eye(dy))
# jacob_obs = lambda x: params.H


num_param_vals = 10
num_comp = 10
init = [np.zeros(dx), np.eye(dx)]
error = np.zeros(Nsim)
error_latent = np.zeros(Nsim)
error_UKF = np.zeros(Nsim)
mean_error = np.zeros(num_param_vals)
mean_error_lat = np.zeros(num_param_vals)
mean_error_UKF = np.zeros(num_param_vals)
freq = 10
i = 0
count = 0
for param_idx in range(0, num_param_vals):
    latent_cov =  np.eye(dx) / (2.0 + param_idx * 10)
    ## 1-D
    f = lambda x:  np.sin(x) * np.sin(freq * x) + np.random.normal(0, params.Q)
    jacob_dyn = lambda x:  np.sin(x) * freq * np.cos(freq * x) + np.cos(x) * np.sin(freq * x)

    ## 2-D
    # Lotka-Voltera
    # (a,b,c,d) = (2/3, 4/3, 1, 1)
    # f = lambda x: 0.1 * np.matmul(np.array([[1-a, -b*x[0]], [d*x[1], 1 - c]]), x) + np.random.multivariate_normal(np.zeros(2), params.Q)
    # jacob_dyn = lambda x: 0.1 * np.array([[1-b*x[1], -b*x[0]], [d*x[1], d*x[0] - c]])
    ssm = StateSpaceModel(dx, dy, f, g)
    for run in range(Nsim):
        simul = ssm.simulate(T, np.zeros(dx))
        means, covs = ssm.extended_kalman_filter(simul[1], jacob_dyn, jacob_obs, params, init)
        means_latent, covs_latent = ssm.latent_ekf(simul[1], num_comp, latent_cov, jacob_dyn, jacob_obs, params, init)
        means_UKF, covs_UKF = ssm.unscented_kalman_filter(simul[1], init, params, 10, 2, 1) # alpha = 1 / np.sqrt(dx)
        error[run] = np.linalg.norm(means[1:] - simul[0][:T-1])
        error_latent[run] = np.linalg.norm(means_latent[1:] - simul[0][:T-1])
        error_UKF[run] = np.linalg.norm(means_UKF[1:] - simul[0][:T - 1])
        print(count)
        count += 1
    mean_error[i] = np.mean(error)
    mean_error_lat[i] = np.mean(error_latent)
    mean_error_UKF[i] = np.mean(error_UKF)
    i += 1

print('Error EKF:', mean_error)
print('Error LEKF:', mean_error_lat)
print('Error UKF:', mean_error_UKF)


fig1, axes1 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes1.plot(simul[0])
p2 = axes1.plot(simul[1])
p3 = axes1.plot(means[1:])
p4 = axes1.plot(means_latent[1:])
p5 = axes1.plot(means_UKF[1:])
axes1.set_ylabel("X")
axes1.set_xlabel("time")
axes1.legend(['x', 'y', 'EKF', 'LEKF', 'UKF'])

fig2, axes2 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes2.plot(mean_error)
p2 = axes2.plot(mean_error_lat)
p3 = axes2.plot(mean_error_UKF)
axes2.set_ylabel("error")
axes2.set_xlabel("freq")
#axes2.set_title("Ex Cond VS Est")
axes2.legend(['EKF', 'LEKF', 'UKF'])

plt.show()




