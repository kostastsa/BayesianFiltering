import numpy as np

# 1
f1 = lambda x, p: (1 + np.dot(x, x))**(p/2)
J1 = lambda x, p: p * (1 + np.dot(x, x))**(p/2-1) * x
H1 = lambda x, p: 2 * p*(p/2-1)*(1 + np.dot(x, x))**(p/2-2)*np.outer(x, x) + np.eye(dx) * p * (1 + np.dot(x, x))**(p/2-1)

# 2 Sinc
f2 = lambda x: np.sin(np.dot(x, x))/np.dot(x,x)
J2 = lambda x: 2 * (np.dot(x, x)*np.cos(np.dot(x, x)) - np.sin(np.dot(x, x))) / (np.dot(x, x))**2 * x
H2 = lambda x: -4*(np.sin(np.dot(x, x))/np.dot(x, x) +
                   2*(np.cos(np.dot(x, x))*np.dot(x, x) -
                     np.sin(np.dot(x, x))) / np.dot(x, x)**3) * \
                     np.outer(x, x) + 2 * (np.dot(x, x)*np.cos(np.dot(x, x)) -
                                          np.sin(np.dot(x, x))) / \
                                          (np.dot(x, x))**2 * np.eye(dx)

# 3 Linear-Nonlinear product <------------------- Hard one
f3 = lambda x: x[0] * np.sin(x[1])
J3 = lambda x: np.array([np.sin(x[1]), x[0]*np.cos(x[1])])
H3 = lambda x: np.array([[0, np.cos(x[1])],[np.cos(x[1]), -x[0]*np.sin(x[1])]])

# 4 Linear-Nonlinear sum
f4 = lambda x: x[0] + np.sin(x[1])
J4 = lambda x: np.array([1, np.cos(x[1])])
H4 = lambda x: np.array([[0, 0], [0, -np.sin(x[1])]])

# 5 Quadratic
a = 1
b = 1
A = np.array([[a, 0], [0, b]])
f5 = lambda x: np.dot(x, np.matmul(A, x))/2
J5 = lambda x: np.matmul(A, x)
H5 = lambda x: A


# Lorentz 96
alpha = 1.0
beta = 1.0
gamma = 8.0
dt = 0.01
H = jnp.zeros((emission_dim, state_dim))
for row in range(emission_dim):
    col = 2*row
    H = H.at[row,col].set(1.0)
CP = lambda n: jnp.block([[jnp.zeros((1,n-1)), 1.0 ],[jnp.eye(n-1), jnp.zeros((n-1,1))]])
A = CP(state_dim)
B = jnp.power(A, state_dim-1) - jnp.power(A, 2)
f96 = lambda x, q, u: x + dt * (alpha * jnp.multiply(A @ x, B @ x) - beta * x + gamma * jnp.ones(state_dim)) + q
g96 = lambda x, r, u: H @ x + r
def g96lp(x,y,u):
  return MVN(loc = g96(x, 0.0, u), covariance_matrix = R).log_prob(y)