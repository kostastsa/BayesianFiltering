from ssm import StateSpaceModel
from ssm import LinearModelParameters
from SLDS import SLDS
from simulation import Simulation
from ssm import LGSSM
import numpy as np
import copy
import matplotlib.pyplot as plt

Q = 1
R = 1


g = lambda x: x + np.random.normal(0, R)


jacob_obs = lambda x: 1
params = LinearModelParameters(0, 0, Q, R)

T = 100
Nsim = 10
Ngrid = 100
mesh_size = 1


init = np.array([0, 1])
error = np.zeros(Nsim)
mean_error = np.zeros(int(Ngrid/mesh_size))
i = 0
for freq in range(1, Ngrid+1, mesh_size):
    f = lambda x: np.sin(x) * np.sin(freq * x) + np.random.normal(0, Q)
    jacob_dyn = lambda x: np.sin(x) * freq * np.cos(freq * x) + np.cos(x) * np.sin(freq * x)
    ssm = StateSpaceModel(1, 1, f, g)
    for run in range(Nsim):
        simul = ssm.simulate(T, 0.0)
        means, covs = ssm.extended_kalman_filter(simul[1], jacob_dyn, jacob_obs, params, init)
        error[run] = np.linalg.norm(means[1:] - simul[0][:T-1])
    mean_error[i] = np.mean(error)
    i += 1


fig1, axes1 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes1.plot(simul[0])
p2 = axes1.plot(simul[1])
p3 = axes1.plot(means[1:])
axes1.set_ylabel("X")
axes1.set_xlabel("time")
axes1.legend(['x', 'y', 'EKF'])

fig2, axes2 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes2.plot(mean_error)
axes2.set_ylabel("error")
axes2.set_xlabel("freq")
#axes2.set_title("Ex Cond VS Est")
#axes2.legend(['Ex Cond', 'GPB', 'IMM'])

plt.show()