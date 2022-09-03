import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
from jax import grad, jit, vmap


dx = 1
dy = 1

m0 = np.zeros(dx)
P0 = np.eye(dx)

A = np.eye(dx) # np.random.random([dx,dx])
B = np.eye(dy, dx) # np.random.random([dy,dx])
c = random.random([1, dx])
d = random.random((1, dy))

f = lambda x: 0.5 * A @ x
g = lambda x: 0.5 * B @ x #+ d
#jnp.dot(x, x) ** 2 #

## Generate Data
seq_length = 10
ssm = gf.SSM(dx, dy, np.zeros(dx), np.eye(dx), np.zeros(dy), np.eye(dy), f, g)
xs, ys = ssm.simulate(seq_length, m0)
#print(xs, ys)

## Test UKF Class
ukf = gf.UKF(ssm, 1e-3, 2, 0)
ukf_out = ukf.run(ys, m0, P0)

## Test MCF Class
num_particles = 10
mcf = gf.MCF(ssm, num_particles)
mcf_out = mcf.run(ys, m0, P0)

## Test EKF Class
ekf = gf.EKF(ssm, order=2)
ekf_out = ekf.run(ys, m0, P0)

ukf_out

