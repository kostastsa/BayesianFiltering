# Bayesian Filtering test file and examples

```python
from ssm import LinearModelParameters
from slds import SLDS
from simulation import Simulation
from ssm import LGSSM
import numpy as np
```
## Data Generation

#### Simulate all variables of a given 1D LGSSM
```python
a = np.array([0.1])
h = np.array([1])
q = np.array([0.01])
r = np.array([0.01])

params = LinearModelParameters(a, h, q, r)
model1 = LGSSM(1,1,params)
sim1 = Simulation(model1, T = 10, init_state=np.array([0]))

#print(sim1.all_data)
```


## Two dimensional model
#### Simulate all variables of a given 2D LGSSM
```python
dx = 2
dy = 2
A = np.eye(dx)
H = 1 * np.eye(dy, dx)
Q = 0.1 * np.eye(dx)
R = 10 * np.eye(dy)

init_state = np.zeros([dx])
params2 = LinearModelParameters(A, H, Q, R)
model2 = LGSSM(dx, dy, params2)
sim2 = Simulation(model2, T = 10, init_state=init_state)

#print(sim2.all_data)
```


## SLDS
#### Simulate all variables of a given Switching Linear Dynamical System 
```python
M = 10 # number of models/regimes
dx = 2
dy = 2

#define a random set of models
var_max = 10
correl_mask_dx = np.ones([dx, dx]) - np.eye(dx)
model_parameter_array = np.empty([M], dtype=LinearModelParameters)
for m in range(M):
    A = np.eye(dx)
    H = 1 * np.random.random([dy, dx])
    Q_nonsym = var_max * np.multiply(np.random.random([dx, dx]), np.eye(dx)) + \
               np.multiply(np.random.random([dx, dx]), correl_mask_dx)
    Q = Q_nonsym + Q_nonsym.T
    R = 1 * np.eye(dy)
    model_parameter_array[m] = LinearModelParameters(A, H, Q, R)
   
# Define SLDS and create a random transition matrix (Dirichlet rows)
SLDS1 = SLDS(dx, dy, model_parameter_array)
alpha = np.random.choice(range(1, 50), M)
mat = np.random.dirichlet(alpha, M)
SLDS1.set_transition_matrix(mat)

sim3 = Simulation(SLDS1, T = 10, init_state=[0, init_state])
print(sim3.all_data)
```

## Filtering
#### Do filtering on a given model and a given set of observations
```python
filt_model = copy.deepcopy(model2)
dx = filt_model.dx
dy = filt_model.dy
init = [np.zeros(dx), np.eye(dx)]
out = filt_model.kalman_filter(sim2.observs, init)
mean_pred = out[0]
```