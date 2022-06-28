from ssm import StateSpaceModel
from ssm import LinearModelParameters
from SLDS import SLDS
from simulation import Simulation
from ssm import LGSSM
import numpy as np
import copy
import matplotlib.pyplot as plt

# Data Generation

## 1 dimensional model
a = np.array([1])
h = np.array([1])
q = np.array([0.01])
r = np.eye(1)

T = 100

params = LinearModelParameters(a, h, q, r)
model1 = LGSSM(1, 1, params)
sim1 = Simulation(model1, T, init_state=np.array([0]))
print(sim1.states)

## SLDS
## SLDS
M = 2
dx = 1
dy = 1
T = 100
model_parameter_array = np.empty([M], dtype=LinearModelParameters)
init_state = np.zeros([dx])

# Fast forgetting
A0 = 1 * np.eye(dx)
H0 = 1 * np.eye(dy, dx)
Q0 = 0.01 * np.eye(dx) #np.array([[10, 0] , [ 0, 0.1]])
R0 = 1 * np.eye(dy)

# Slow forgetting
theta = np.pi*5/360
A1 = -A0 # 1 * np.array([[np.cos(theta), -np.sin(theta)] , [ np.sin(theta), np.cos(theta)]])
H1 = 1 * np.eye(dy, dx)
Q1 = 0.01 * np.eye(dx) # np.array([[0.1, 0] , [ 0, 10]])
R1 = 1 * np.random.rand() * np.eye(dy)

model_parameter_array[0] = params
model_parameter_array[1] = LinearModelParameters(A1, H1, Q1, R1)

SLDS1 = SLDS(dx, dy, model_parameter_array)

# alpha = np.random.choice(range(1, 50), M)
# mat = np.random.dirichlet(alpha, M) # Random tranisiton matrix with Dirichlet(alpha) rows
mat = np.array([[0.9, 0.1] , [0.9, 0.1]])

SLDS1.set_transition_matrix(mat)

# Filtering
## LGSSM 1D
filt_model = copy.deepcopy(model1)
dx = filt_model.dx
dy = filt_model.dy
init = [np.zeros(dx), np.eye(dx)]
out = filt_model.kalman_filter(sim1.observs, init)
mean_pred = out[0]

### IMM
filt_slds_model = copy.deepcopy(SLDS1)
dx_slds = filt_slds_model.dx
mean_out_IMM,cov_out_IMM, weights_out_IMM = filt_slds_model.IMM(sim1.observs, init)

##

# Plots

fig1, axes1 = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
p1 = axes1.plot(sim1.states, alpha=0.6, label="States")
p2 = axes1.plot(mean_pred, alpha=0.6)
axes1.plot(mean_out_IMM, alpha=0.6)
axes1.set_ylabel("X")
axes1.set_xlabel("time")
axes1.set_title("True States VS Est")
axes1.legend(['true', 'KF', 'IMM'])



plt.show()
