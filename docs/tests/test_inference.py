from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax
from jax import random as jr
from jax import tree_util as jtu
import csv

import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from gaussfiltax.models import ParamsNLSSM, NonlinearGaussianSSM, NonlinearSSM, ParamsBPF
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf
from gaussfiltax.utils import sdp_opt

import matplotlib.pyplot as plt

class TestInference:

    # Parameters
    state_dim = 1
    emission_dim = 1
    seq_length = 100
    mu0 = jnp.zeros(state_dim)
    Sigma0 = 1.0 * jnp.eye(state_dim)
    Q = 1.0 * jnp.eye(state_dim)
    R = 1.0 * jnp.eye(emission_dim)

    # Nonlinearities
    f1 = lambda x: jnp.sin(10 * x)
    g1 = lambda x: 0.1 * jnp.array([jnp.dot(x, x)])

    f2 = lambda x: 0.8 * x
    def g2(x):
        return 2 * x
    #    return jnp.array([[1, 0], [0, 1], [0, 0]]) @ x

    # stochastic growth model
    f3 = lambda x, u: x / 2. + 25. * x / (1+ jnp.power(x, 2)) + u
    g3 = lambda x, u: x**2/20.

    # Inputs
    inputs = 8. * jnp.cos(jnp.arange(seq_length))

    # Model definition 
    model = NonlinearGaussianSSM(state_dim, emission_dim)
    params = ParamsNLSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_function=f3,
        dynamics_noise_bias=jnp.zeros(Q.shape[0]),
        dynamics_noise_covariance=Q,
        emission_function=g3,
        emission_noise_bias=jnp.zeros(R.shape[0]),
        emission_noise_covariance=R,
    )

    # Generate synthetic data 
    key = jr.PRNGKey(0)
    states, emissions = model.sample(params, key, seq_length, inputs = inputs)


    def test_gaussian_sum_filter(self):
        num_components = 5
        posterior_filtered = gf.gaussian_sum_filter(self.params, self.emissions, num_components, 1, self.inputs)

    def test_augmented_gaussian_sum_filter(self):
        num_components = [5, 3, 3] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter(self.params, self.emissions, num_components, opt_args = (20, 0.001, 50), inputs=self.inputs)

# test = TestInference()

# tin = time.time()
# test.test_gaussian_sum_filter()
# tout = time.time()
# print('GSF time:', tout - tin)

# tin = time.time()
# test.test_augmented_gaussian_sum_filter()
# tout = time.time()
# print('AGSF time:', tout - tin)



            # # Parameters
            # state_dim = 1
            # state_noise_dim = 1
            # emission_dim = 1
            # emission_noise_dim = 1
            # seq_length = 100
            # mu0 = 1.0 * jnp.zeros(state_dim)
            # Sigma0 = 0.001 * jnp.eye(state_dim)
            # Q = 1.0 * jnp.eye(state_noise_dim)
            # R = 1.0 * jnp.eye(emission_noise_dim)

            # # ICASSP
            # f1 = lambda x, q, u: (1-u) * x / 2.  + u * jnp.sin(10 * x) + q
            # g1 = lambda x, r, u:  0.1 * jnp.dot(x, x) + r
            # def g1lp(x,y,u):
            #     return MVN(loc = g1(x, 0.0, u), covariance_matrix = R).log_prob(y)


            # # Lorenz 63
            # def lorentz_63(x, sigma=10, rho=28, beta=2.667, dt=0.01):
            #     dx = dt * sigma * (x[1] - x[0])
            #     dy = dt * (x[0] * rho - x[1] - x[0] *x[2]) 
            #     dz = dt * (x[0] * x[1] - beta * x[2])
            #     return jnp.array([dx+x[0], dy+x[1], dz+x[2]])
            # f63 = lambda x, q, u: lorentz_63(x) + q

            # # Lorentz 96
            # alpha = 1.0
            # beta = 1.0
            # gamma = 8.0
            # dt = 0.01
            # H = jnp.zeros((emission_dim,state_dim))
            # for row in range(emission_dim):
            #     col = 2*row
            #     H = H.at[row,col].set(1.0)
            # CP = lambda n: jnp.block([[jnp.zeros((1,n-1)), 1.0 ],[jnp.eye(n-1), jnp.zeros((n-1,1))]])
            # A = CP(state_dim)
            # B = jnp.power(A, state_dim-1) - jnp.power(A, 2)
            # f96 = lambda x, q, u: x + dt * (alpha * jnp.multiply(A @ x, B @ x) - beta * x + gamma * jnp.ones(state_dim)) + q
            # g96 = lambda x, r, u: H @ x + r
            # def g96lp(x,y,u):
            # return MVN(loc = g96(x, 0.0, u), covariance_matrix = R).log_prob(y)

            # # stochastic growth model
            # f3 = lambda x, q, u: x / 2. + 25. * x / (1 + jnp.power(x, 2)) * u + q
            # g3 = lambda x, r, u: 0.8 * x + r
            # def g3lp(x,y,u):
            #     return MVN(loc = g3(x, 0.0, u), covariance_matrix = R).log_prob(y)


            # # Stochastic Volatility
            # alpha = 0.91
            # sigma = 1.0
            # beta = 0.5
            # fsv= lambda x, q, u: alpha * x + sigma * q
            # gsv = lambda x, r, u: beta * jnp.exp(x/2) * r
            # def svlp(x,y,u):
            #     return MVN(loc = gsv(x, 0.0, u), covariance_matrix = gsv(x, 1.0, u)**2 * R).log_prob(y)

            # # Linear - Stochastic Volatility
            # alpha = 0.91
            # sigma = 1.0
            # beta = 0.5
            # glsv = lambda x, r, u: u * beta * jnp.exp(x/2) * r + (1-u) * (0.8 * x + r)
            # def lsvlp(x,y,u):
            #     return MVN(loc = glsv(x, 0.0, u), covariance_matrix = u * beta ** 2 * jnp.exp(x) * R + (1-u) * R).log_prob(y)

            # # Multivariate SV
            # Phi = 0.8 * jnp.eye(state_dim)
            # fmsv = lambda x, q, u: Phi @ x +  q
            # gmsv = lambda x, r, u:  u *0.5 * jnp.multiply(jnp.exp(x/2), r)
            # def msvlp(x,y,u):
            #     return MVN(loc = gmsv(x, 0.0, u), covariance_matrix = jnp.diag(jnp.exp(x/2.0)) @ R @ jnp.diag(jnp.exp(x/2.0))).log_prob(y)

            # # Linear - MS Volatility
            # alpha = 0.91
            # sigma = 1.0
            # beta = 0.5
            # glmsv = lambda x, r, u: u * 0.5 * jnp.multiply(jnp.exp(x/2), r) + (1-u) * (0.8 * x + r)
            # def lmsvlp(x,y,u):
            #     M = u * 0.5 * jnp.diag(jnp.exp(x/2.0)) + (1-u) * jnp.eye(state_dim)
            #     return MVN(loc = glsv(x, 0.0, u), covariance_matrix = M @ R @ M.T).log_prob(y)


            # # Inputs
            # # inputs = 1. * jnp.cos(0.1 * jnp.arange(seq_length))
            # sm = lambda x : jnp.exp(x) / (1+jnp.exp(x))
            # inputs = sm(jnp.arange(seq_length)-50) # off - on
            # # inputs = 1.0 * jnp.ones(seq_length) # on - on


            # f = f1
            # g = g1
            # glp = g1lp


            # # initialization 
            # model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
            # params = ParamsNLSSM(
            #     initial_mean=mu0,
            #     initial_covariance=Sigma0,
            #     dynamics_function=f,
            #     dynamics_noise_bias=jnp.zeros(state_noise_dim),
            #     dynamics_noise_covariance=Q,
            #     emission_function=g,
            #     emission_noise_bias=jnp.zeros(emission_noise_dim),
            #     emission_noise_covariance=R,
            # )

            # # Generate synthetic data 
            # key = jr.PRNGKey(0)
            # states, emissions = model.sample(params, key, seq_length, inputs = inputs)


            # # GSF
            # M = 5
            # tin = time.time()
            # posterior_filtered_gsf = gf.gaussian_sum_filter(params, emissions, M, 1, inputs)
            # point_estimate_gsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_gsf.means, posterior_filtered_gsf.weights), axis=0)
            # tout = time.time()
            # print('Time taken for GSF: ', tout - tin)

            # # AGSF
            # tin = time.time()
            # num_components = [M, 3, 3] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
            # posterior_filtered_agsf, aux_outputs = gf.augmented_gaussian_sum_filter(params, emissions, num_components, rng_key = key, opt_args = (1.0, 1.0), inputs=inputs)    
            # point_estimate_agsf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_filtered_agsf.means, posterior_filtered_agsf.weights), axis=0)
            # tout = time.time()
            # print('Time taken for AGSF: ', tout - tin)

            # # BPF
            # tin = time.time()
            # num_particles = 100

            # params_bpf = ParamsBPF(
            #     initial_mean=mu0,
            #     initial_covariance=Sigma0,
            #     dynamics_function=f,
            #     dynamics_noise_bias=jnp.zeros(state_noise_dim),
            #     dynamics_noise_covariance=Q,
            #     emission_function=g,
            #     emission_noise_bias=jnp.zeros(emission_noise_dim),
            #     emission_noise_covariance=R,
            #     emission_distribution_log_prob = glp
            # )

            # posterior_bpf = gf.bootstrap_particle_filter(params_bpf, emissions, num_particles, key, inputs)
            # point_estimate_bpf = jnp.sum(jnp.einsum('ijk,ij->ijk', posterior_bpf["particles"], posterior_bpf["weights"]), axis=0)
            # tout = time.time()
            # print('Time taken for BPF: ', tout - tin)


            # # Mean squared errors
            # gsf_mse = jnp.linalg.norm(point_estimate_gsf - states, axis = 1)
            # agsf_mse = jnp.linalg.norm(point_estimate_agsf - states, axis = 1)
            # bpf_mse = jnp.linalg.norm(point_estimate_bpf - states, axis = 1)
            # print('GSF MSE: ', jnp.mean(gsf_mse))
            # print('AGSF MSE: ', jnp.mean(agsf_mse))
            # print('BPF MSE: ', jnp.mean(bpf_mse))


            # plt.plot(gsf_mse)
            # plt.plot(agsf_mse)
            # plt.plot(bpf_mse)
            # plt.legend(['GSF', 'AGSF', 'BPF'])
            # plt.show()


state_dim = 3
A = jnp.eye(state_dim-1, state_dim)
f = lambda x: jnp.concatenate((A @ x, jnp.array([jnp.dot(x.T,x)])), axis=0)
N = 100
P = 10*jnp.eye(state_dim)
m = jnp.ones((state_dim,))
jacobian = jacfwd(f)(m)
hessian = jacrev(jacfwd(f))(m)
Lambda = sdp_opt(state_dim, N, P, jacobian, hessian, 100)
print(jnp.linalg.eigvals(Lambda))
print(Lambda)