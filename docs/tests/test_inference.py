import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')
from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax
from jax import random as jr
from jax import tree_util as jtu
import csv

from dynamax.utils.utils import psd_solve
import gaussfiltax.utils as utils
import gaussfiltax.containers as containers
from gaussfiltax.models import ParamsNLSSM, NonlinearGaussianSSM, NonlinearSSM, ParamsBPF
from gaussfiltax.inference import ParamsUKF
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
import time
import gaussfiltax.inference as gf
from gaussfiltax.utils import sdp_opt

import matplotlib.pyplot as plt

_MVN_log_prob = lambda mean, cov, y: MVN(mean, cov).log_prob(jnp.atleast_1d(y))

def lorentz_63(x, sigma=10, rho=28, beta=2.667, dt=0.01):
    dx = dt * sigma * (x[1] - x[0])
    dy = dt * (x[0] * rho - x[1] - x[0] *x[2]) 
    dz = dt * (x[0] * x[1] - beta * x[2])
    return jnp.array([dx+x[0], dy+x[1], dz+x[2]])

state_dim = 1
state_noise_dim = 1
emission_dim = 1
emission_noise_dim = 1

class TestInference:
    # Parameters
    seq_length = 100
    mu0 = jnp.zeros(state_dim)
    Sigma0 = 1.0 * jnp.eye(state_dim)
    Q = 1.0 * jnp.eye(state_dim)
    R = .1 * jnp.eye(emission_dim)

    fLG = lambda x, q, u: 0.8 * jnp.eye(state_dim, state_dim) @ x + q
    gLG = lambda x, r, u: 1.0 * jnp.eye(emission_dim, state_dim) @ x + r

    # stochastic growth model
    # f3 = lambda x, q, u: x / 2. + 25. * x / (1+ jnp.power(x, 2)) + u + q
    f3 = lambda x, q, u: 0.8 * x + q
    g3 = lambda x, r, u: x**2/20. + r

    # Lorenz 63
    f63 = lambda x, q, u: lorentz_63(x) + q
    g1 = lambda x, r, u:  0.05 * jnp.dot(x, x) + r

    # Inputs
    inputs = 0. * jnp.cos(jnp.arange(seq_length))

    # Model definition 
    model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
    params = ParamsNLSSM(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_function=fLG,
        dynamics_noise_bias=jnp.zeros(Q.shape[0]),
        dynamics_noise_covariance=Q,
        emission_function=gLG,
        emission_noise_bias=jnp.zeros(R.shape[0]),
        emission_noise_covariance=R,
    )

    # Generate synthetic data 
    key = jr.PRNGKey(0)
    states, emissions = model.sample(params, key, seq_length, inputs = inputs)

    def test_gaussian_sum_filter(self):
        num_components = 5
        posterior_filtered = gf.gaussian_sum_filter(self.params, self.emissions, num_components, 1, self.inputs)
        return posterior_filtered

    def test_augmented_gaussian_sum_filter(self):
        num_components = [2, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=self.inputs)
        return posterior_filtered

    def test_augmented_gaussian_sum_filter_optimal(self):
        num_components = [5, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        posterior_filtered, aux_outputs = gf.augmented_gaussian_sum_filter_optimal(self.params, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=self.inputs)
        return posterior_filtered
    
    def test_unscented_gaussian_sum_filter(self):
        num_components = 5
        uparams = ParamsUKF(alpha=1e-3, beta=2.0, kappa=10.0)
        posterior_filtered = gf.unscented_gaussian_sum_filter(self.params, uparams, self.emissions, num_components, 1, self.inputs)
        return posterior_filtered
    
    def test_unscented_agsf(self):
        num_components = [2, 5, 5] # has to be set correctly OW "TypeError: Cannot interpret '<function <lambda> at 0x12eae3ee0>' as a data type". Check internal containers._branch_from_node
        uparams = ParamsUKF(alpha=1e-3, beta=2.0, kappa=0.0)
        posterior_filtered, aux_outputs = gf.unscented_agsf(self.params, uparams, self.emissions, num_components, opt_args = (0.1, 0.1), inputs=self.inputs)
        return posterior_filtered

if __name__ == "__main__":
    test = TestInference()
    posterior_filtered = test.test_unscented_gaussian_sum_filter()
    print(posterior_filtered.means)
    plt.plot(test.states, label='true')
    for m in range(2):
        plt.plot(posterior_filtered.means[m], label='{}'.format(m))
    plt.legend()
    plt.show()
    
    # def _ukf_condition_on_additive(m, P, h, R, u, y, uparams):
    #     r"""Condition a Gaussian potential on a new observation using first-order additive UKF.
    #     """
    #     state_dim = m.shape[0]
    #     emission_dim = y.shape[0]
    #     r0 = jnp.zeros(emission_dim)
    #     lamda = uparams.alpha**2 * (state_dim + uparams.kappa) - state_dim
    #     sigma_points = utils._get_sigma_points(m, P, lamda)
    #     new_sigma_points = vmap(h, in_axes=(0, None, None))(sigma_points, r0, u)

    #     mu_pred = jnp.sum(new_sigma_points, axis=0) / (2*(lamda+state_dim)) + h(m, r0, u) * lamda / (lamda + state_dim)
    #     S = (new_sigma_points - mu_pred).T @ (new_sigma_points - mu_pred) / ( 2 * (lamda+state_dim)) + \
    #                 (h(m, r0, u)-mu_pred) @ (h(m, r0, u) - mu_pred).T * (lamda / (lamda + state_dim) + 1-uparams.alpha**2+uparams.beta) + R
    #     C = (new_sigma_points - mu_pred).T @ (sigma_points - m) / (2*(lamda+state_dim)) 
        
    #     K = psd_solve(S, C).T
    #     posterior_cov = P - K @ S @ K.T
    #     posterior_mean = m + K @ (y - mu_pred)
    #     ll = _MVN_log_prob(mu_pred, S, y)
    #     return ll, posterior_mean, posterior_cov
    
    # def _ukf_predict_additive(m, P, f, u, Q, uparams):
    #     r"""Predict next mean and covariance using first-order additive UKF`
            
    #     Args:
    #         m (D_hid,): prior mean.
    #         P (D_hid,D_hid): prior covariance.
    #         f (Callable): dynamics function.
    #         Q (D_hid,D_hid): dynamics covariance matrix.
    #         u (D_in,): inputs.
    #     Returns:
    #         mu_pred (D_hid,): predicted mean.
    #         Sigma_pred (D_hid,D_hid): predicted covariance.
    #     """
    #     state_dim = m.shape[0]
    #     q0 = jnp.zeros((state_dim,))
    #     lamda = uparams.alpha**2 * (state_dim + uparams.kappa) - state_dim
    #     sigma_points = utils._get_sigma_points(m, P, lamda)
    #     new_sigma_points = vmap(f, in_axes=(0, None, None))(sigma_points, q0, u)
    
    #     mu_pred = jnp.sum(new_sigma_points, axis=0) / (2*(lamda+state_dim)) + f(m, q0, u) * lamda / (lamda + state_dim)
    #     Sigma_pred = (new_sigma_points - mu_pred).T @ (new_sigma_points - mu_pred) / ( 2 * (lamda+state_dim)) + \
    #                 (f(m, q0, u)-mu_pred) @ (f(m, q0, u) - mu_pred).T * (lamda / (lamda + state_dim) + 1-uparams.alpha**2+uparams.beta) + Q
    #     return mu_pred, Sigma_pred

    # uparams = ParamsUKF(alpha=1.0, beta=2.0, kappa=0.0)
    # fLG = lambda x, q, u: 0.8 * jnp.eye(state_dim) @ x + q
    # gLG = lambda x, r, u: 10 * jnp.eye(emission_dim, state_dim) @ x + r
    # ll, posterior_mean, posterior_cov = _ukf_condition_on_additive(test.mu0, test.Sigma0, gLG, test.R, test.inputs[0], test.emissions[0], uparams)

    # mu_pred, Sigma_pred = _ukf_predict_additive(posterior_mean, posterior_cov, fLG, test.inputs[0], test.Q, uparams)

    # print(posterior_mean, posterior_cov)
    # print(mu_pred, Sigma_pred)