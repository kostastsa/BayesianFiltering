import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
import particlefilt as pf
import matplotlib.pyplot as plt
import time


dx = 1
dy = 1
seq_length = 100
m0 = 0.1 * np.ones(dx)
P0 =10 * np.eye(dx)
c = np.zeros(dx)
d = np.zeros(dy)
Q = 4 * np.eye(dx)
R = 1 * np.eye(dy)

## Define nonlinearity
A = 0.8 * np.eye(dx)
B = np.eye(dy, dx)
f = lambda x: jnp.sin(x)
g = lambda x: x**2


# Generate Data

np.random.seed(seed=1)

ssm = gf.SSM(dx, dy, c=c, Q=P0, d=d, R=R, f=f, g=g)
xs, ys = ssm.simulate(seq_length, m0)

num_prt = 10
bpf = pf.BootstrapPF(ssm, num_prt)
bpf_out = bpf.run(ys, m0, P0)

fig1, axes1 = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
axes1[0].plot(xs[:, 0], alpha=1, label="xs")
axes1[0].plot(np.sum(bpf_out[:seq_length], 1) / num_prt, alpha=0.7, label="agsf")
axes1[0].legend(['x', 'BPF'])
axes1[1].plot(xs[:, 0], alpha=1, label="xs")
plt.show()