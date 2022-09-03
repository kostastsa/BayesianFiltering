import utils
import gaussfilt as gf
import numpy as np
import jax.numpy as jnp
from numpy import random
from jax import grad, jit, vmap


dx = 2
dy = 1

A = np.eye(dx) # np.random.random([dx,dx])
B = np.eye(dy, dx) # np.random.random([dy,dx])
c = random.random([1, dx])
d = random.random((1, dy))

f = lambda x: 0.5 * A @ x
g = lambda x: 0.5 * B @ x #+ d
#jnp.dot(x, x) ** 2 #

## Generate Data
ssm = gf.SSM(dx, dy, np.zeros(dx), np.eye(dx), np.zeros(dy), np.eye(dy), f, g)
xs, ys = ssm.simulate(100, np.zeros(dx))

## Test UKF Class
ukf = gf.UKF(ssm, 1e-3, 2, 0)
#print(ukf.run(ys, np.zeros(dx), np.eye(dx)))

## Test MCF Class
mcf = gf.MCF(ssm, 100)
#print(mcf.run(ys, np.zeros(dx), np.eye(dx)))

## Test EKF Class
ekf = gf.EKF(ssm, order=2)
ekf.run(ys, np.zeros(dx), np.eye(dx))

