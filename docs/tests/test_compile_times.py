import sys 
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax')
from jax import numpy as jnp
from jax import jacfwd, jacrev, jit, vmap, lax, make_jaxpr
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

# Parameters
state_dim = 4
state_noise_dim = 2
emission_dim = 4
emission_noise_dim = 4
seq_length = 30
# mu0 = 1.0 * jnp.array([-0.05, 0.001, 0.7, -0.05])
mu0 = jnp.ones(state_dim)
q0 = jnp.zeros(state_noise_dim)
r0 = jnp.zeros(emission_noise_dim)
Sigma0 = 1.0 * jnp.array([[0.1, 0.0, 0.0, 0.0],[0.0, 0.005, 0.0, 0.0],[0.0, 0.0, 0.1, 0.0],[0.0, 0.0, 0.0, 0.01]])
Q = 1 * jnp.eye(state_noise_dim)
R = 25*1e-6 * jnp.eye(emission_noise_dim)

dt = 0.5
FCV = jnp.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
acc = 0.5
Omega = lambda x, acc: 0.1 * acc / jnp.sqrt(x[1]**2 + x[3]**2)
FCT =  lambda x, a: jnp.array([[1, jnp.sin(dt * Omega(x, a)) / Omega(x, a), 0, -(1-jnp.cos(dt * Omega(x, a))) / Omega(x, a)],
                            [0, jnp.cos(dt * Omega(x, a)), 0, -jnp.sin(dt * Omega(x, a))],
                            [0, (1-jnp.cos(dt * Omega(x, a))) / Omega(x, a), 1, jnp.sin(dt * Omega(x, a)) / Omega(x, a)],
                            [0, jnp.sin(dt * Omega(x, a)), 0, jnp.cos(dt * Omega(x, a))]])

G = jnp.array([[0.5, 0],[1, 0],[0, 0.5],[0, 1]])
fBOT = lambda x, q, u: FCV @ x + G @ q
fManBOT = lambda x, q, u: (0.5*(u-1)*(u-2)*FCV - u*(u-2)*FCT(x, acc) + 0.5*u*(u-1) * FCT(x, -acc)) @ x + G @ q
gBOT = lambda x, r, u: jnp.arctan2(x[2], x[0]) + r
gLin = lambda x, r, u: FCT(x, acc) @ x + r
# inputs = jnp.zeros((seq_length, 1))
inputs = jnp.array([1]*int(seq_length/3) + [0]*int(seq_length/3) + [2]*int(seq_length/3)) # maneuver inputs

f = fBOT
g = gLin

F_x, H_x = jacfwd(f, argnums=0), jacfwd(g, argnums=0)
F_q, H_r = jacfwd(f, argnums=1), jacfwd(g, argnums=1)
F_xx, H_xx = jit(jacrev(F_x)), jit(jacrev(H_x))

# Model definition 
model = NonlinearSSM(state_dim, state_noise_dim, emission_dim, emission_noise_dim)
params = ParamsNLSSM(
    initial_mean=mu0,
    initial_covariance=Sigma0,
    dynamics_function=f,
    dynamics_noise_bias=jnp.zeros(Q.shape[0]),
    dynamics_noise_covariance=Q,
    emission_function=g,
    emission_noise_bias=jnp.zeros(R.shape[0]),
    emission_noise_covariance=R,
)


class TestCompile:

    M = 2
    num_prt1 = containers.num_prt1
    num_prt2 = containers.num_prt2
    num_components = (M, num_prt1, num_prt2)

    rng_key = jr.PRNGKey(0)
    nums_to_split = jnp.array([num_components[1]]*num_components[0])

    initial_means = MVN(params.initial_mean, params.initial_covariance).sample(num_components[0], jr.PRNGKey(0))
    initial_covs = jnp.array([params.initial_covariance for i in range(num_components[0])])
    initial_weights = jnp.ones(shape=(num_components[0],)) / num_components[0]
    init_components = containers._gaussian_sum_to_components(containers.GaussianSum(initial_means, initial_covs, initial_weights))

    Deltas, nums_to_split = vmap(gf._autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(initial_means, initial_covs, F_x, F_xx, nums_to_split, q0, 0.0, 1.0)


    def test_compilev0(self):
        rng_key = jr.PRNGKey(0)
        key, subkey = jr.split(rng_key)
        tin = time.time()
        _components_to_predict = containers._branches_from_tree1(self.init_components, list(self.Deltas), list(self.nums_to_split), subkey)
        leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, containers.GaussianComponent))
        _sum_to_predict = containers._components_to_gaussian_sum(leaves)
        t_branch1 = time.time() - tin

        jaxpr_branches_from_tree = str(make_jaxpr(containers._branches_from_tree1)(self.init_components, list(self.Deltas), list(self.nums_to_split), subkey))
        jaxpr_components_to_gaussian_sum = str(make_jaxpr(containers._components_to_gaussian_sum)(leaves))

        return jaxpr_branches_from_tree, jaxpr_components_to_gaussian_sum
    
    def _test_compile_vautocov(self):
        vautocov = lambda a,b,c: vmap(gf._autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(a, b, F_x, F_xx, c, q0, 0.0, 1.0)
        jaxpr_vautocov = str(make_jaxpr(vautocov)(self.initial_means, self.initial_covs, self.nums_to_split))
        autocov = lambda a,b,c: gf._autocov1(a, b, F_x, F_xx, c, q0, 0.0, 1.0)
        jaxpr_autocov = str(make_jaxpr(autocov)(self.initial_means[0], self.initial_covs[0], self.nums_to_split[0]))
        return jaxpr_autocov, jaxpr_vautocov
    
    def test_compile_predict(self):
        predict = lambda x: gf._predict(self.initial_means[0], self.initial_covs[0], f, F_x, F_q, Q, q0, 0.0)
        vpredict = lambda a,b: vmap(gf._predict, in_axes=(0,0,None,None,None,None,None,None))(a, b, f, F_x, F_q, Q, q0, 0.0)
        jaxpr_predict = str(make_jaxpr(predict)(None))
        jaxpr_vpredict = str(make_jaxpr(vpredict)(self.initial_means, self.initial_covs))
        return jaxpr_predict, jaxpr_vpredict
    
    def test_compile_predict2(self):

        def _custom_predict(m, P, f, F_x, F_q, Q, q0, u):
            r"""Predict next mean and covariance using first-order additive EKF
                p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                            = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)
            Args:
                m (D_hid,): prior mean.
                P (D_hid,D_hid): prior covariance.
                f (Callable): dynamics function.
                F (Callable): Jacobian of dynamics function.
                Q (D_hid,D_hid): dynamics covariance matrix.
                u (D_in,): inputs.
            Returns:
                mu_pred (D_hid,): predicted mean.
                Sigma_pred (D_hid,D_hid): predicted covariance.
            """
            Jac_x = F_x(m, q0, u)
            Jac_q = F_q(m, q0, u)
            mu_pred = f(m, q0, u)
            Sigma_pred = Jac_x @ P @ Jac_x.T + Jac_q @ Q @ Jac_q.T
            return mu_pred, Sigma_pred, Jac_x
        
        _custom_predict2 = lambda x: _custom_predict(self.initial_means[0], self.initial_covs[0], f, F_x, F_q, Q, q0, 0.0)
        jaxpr_predict = str(make_jaxpr(_custom_predict2)(None))
        return jaxpr_predict
    
    def test_compile_condition_on(self):

        def _condition_on(m, P, h, H_x, H_r, R, r0, u, y):
            r"""Condition a Gaussian potential on a new observation.
            p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
                propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
                = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
                = N(z_t | mm, SS)
            where
                mm = m + K*(y - yhat) = mu_cond
                yhat = h(m, u)
                S = R + H(m,u) * P * H(m,u)'
                K = P * H(m, u)' * S^{-1}
                SS = P - K * S * K' = Sigma_cond
            **Note! This can be done more efficiently when R is diagonal.**
            Args:
                m (D_hid,): prior mean.
                P (D_hid,D_hid): prior covariance.
                h (Callable): emission function.
                H (Callable): Jacobian of emission function.
                R (D_obs,D_obs): emission covariance matrix.
                u (D_in,): inputs.
                y (D_obs,): observation.
                num_iter (int): number of re-linearizations around posterior for update step.
            Returns:
                mu_cond (D_hid,): filtered mean.
                Sigma_cond (D_hid,D_hid): filtered covariance.
            """
            H_x = H_x(m, r0, u)
            H_r = H_r(m, r0, u)
            S = H_r @ R @ H_r.T + H_x @ P @ H_x.T
            K = psd_solve(S, H_x @ P).T
            posterior_cov = P - K @ S @ K.T
            posterior_mean = m + K @ (y - h(m, r0, u))
            return posterior_mean, posterior_cov, H_x, K

        condition_on = lambda x: _condition_on(self.initial_means[0], self.initial_covs[0], g, H_x, H_r, R, r0, 0.0, jnp.zeros(emission_dim))
        jaxpr_condition_on = str(make_jaxpr(condition_on)(None))
        return jaxpr_condition_on

    
    def test_compile_step(self):

        def _step(carry, t):
            filtered_components = carry
            filtered_sum = containers._components_to_gaussian_sum(filtered_components)
            filtered_means = jnp.array(filtered_sum.means)
            filtered_covs = jnp.array(filtered_sum.covariances)
            filtered_weights = jnp.array(filtered_sum.weights)
            
            # Get parameters and inputs for time index t
            Q = gf._get_params(params.dynamics_noise_covariance, 2, t)
            q0 = gf._get_params(params.dynamics_noise_bias, 2, t)
            R = gf._get_params(params.emission_noise_covariance, 2, t)
            r0 = gf._get_params(params.emission_noise_bias, 2, t)
            u = inputs[t]
            y = jnp.zeros(emission_dim)

            # Autocov 1
            tin = time.time()
            nums_to_split = jnp.array([self.num_components[1]]*self.num_components[0])
            # Deltas, nums_to_split = vmap(gf._autocov1, in_axes=(0, 0, None, None, 0, None, None, None))(filtered_means, filtered_covs, F_x, F_xx, nums_to_split, q0, u, 1.0)
            # Deltas = jnp.array([1.0 * filtered_covs[i] for i in range(self.num_components[0])])

            state_dim = filtered_covs[0].shape[0]
            hessian = F_xx(filtered_means[0], q0, u)
            J = F_x(filtered_means[0], q0, u)
            Delta = utils.sdp_opt(state_dim, self.num_components[0], filtered_covs[0], J, hessian, 1.0)
            Deltas = jnp.array([Delta for i in range(self.num_components[0])])
            t_autocov1 = time.time() - tin

            # Branch 1
            key, subkey = jr.split(jr.PRNGKey(0))
            tin = time.time()
            _components_to_predict = gf._branches_from_tree1(filtered_components, list(Deltas), list(nums_to_split), subkey)
            leaves, treedef = jtu.tree_flatten(_components_to_predict, is_leaf=lambda x: isinstance(x, containers.GaussianComponent))
            _sum_to_predict = containers._components_to_gaussian_sum(leaves)
            t_branch1 = time.time() - tin

            # Predict
            tin = time.time()
            predicted_means, predicted_covs, grads_dyn = vmap(gf._predict, in_axes=(0,0,None,None,None,None,None,None))(jnp.array(_sum_to_predict.means), jnp.array(_sum_to_predict.covariances), f, F_x, F_q, Q, q0, u)
            t_predict = time.time() - tin

            # Recast 1
            tin = time.time()
            predicted_sum = containers.GaussianSum(list(predicted_means), list(predicted_covs), _sum_to_predict.weights)
            predicted_components = containers._gaussian_sum_to_components(predicted_sum)
            t_recast1= time.time() - tin

            # Autocov before update
            tin = time.time()
            nums_to_split = jnp.array([self.num_components[2]] * self.num_components[0]*self.num_components[1])
            # Lambdas, nums_to_split = vmap(gf._autocov2, in_axes=(0, 0, None, None, 0, None, None, None))(predicted_means, predicted_covs, H_x, H_xx, nums_to_split, r0, u, 1.0)
            Lambdas = jnp.array([1.0 * predicted_covs[i] for i in range(self.num_components[0]*self.num_components[1])])
            
            t_autocov2 = time.time() - tin

            # Branching before update
            key, subkey = jr.split(key)
            tin = time.time()
            _components_to_update = containers._branches_from_tree2(predicted_components, list(Lambdas), list(nums_to_split), subkey)
            leaves, treedef = jtu.tree_flatten(_components_to_update, is_leaf=lambda x: isinstance(x, gf.GaussianComponent))
            _sum_to_update = containers._components_to_gaussian_sum(leaves)
            t_branch2 = time.time() - tin

            # Update
            tin = time.time()
            lls, updated_means, updated_covs, grads_obs, gain = vmap(gf._condition_on, in_axes=(0,0,None,None,None,None,None,None,None))(jnp.array(_sum_to_update.means), jnp.array(_sum_to_update.covariances), g, H_x, H_r, R, r0, u, y)
            lls -= jnp.max(lls)
            ls = jnp.exp(lls)
            weights = jnp.multiply(ls, jnp.array(_sum_to_update.weights))
            weights /= jnp.sum(weights)
            pre_weights = weights
            t_update = time.time() - tin

            # Resampling 
            tin = time.time()
            resampled_idx = jr.choice(jr.PRNGKey(0), jnp.arange(weights.shape[0]), shape=(self.num_components[0], ), p=weights)
            filtered_means = jnp.take(updated_means, resampled_idx, axis=0)
            filtered_covs = jnp.take(updated_covs, resampled_idx, axis=0)
            # filtered_covs = 10*filtered_covs # Covariance re-inflation
            weights = jnp.ones(shape=(self.num_components[0],)) / self.num_components[0]
            t_re =  time.time() - tin

            carry = containers._gaussian_sum_to_components(containers.GaussianSum(list(filtered_means), list(filtered_covs), weights))
            outputs = {
                "weights": weights,
                "means": filtered_means,
                "covariances": filtered_covs
            }

            aux_outputs = {
                "Deltas": Deltas,
                "Lambdas": Lambdas,
                "grads_dyn": grads_dyn,
                "grads_obs": grads_obs,
                "gain": gain,
                "timing": jnp.array([t_autocov1, t_branch1, t_predict, t_recast1, t_autocov2, t_branch2, t_update, t_re]),
                "updated_means": updated_means,
                "pre_weights": pre_weights
            }

            return carry, (outputs, aux_outputs)
        
        carry = self.init_components
        out = str(make_jaxpr(_step)(carry, 0))
        return out
 
if __name__ == "__main__":
    test = TestCompile()

    # Test containers
    jaxpr1, jaxpr2 = test.test_compilev0()
    print('branches_from_tree', len(jaxpr1))
    print('_components_to_gaussian_sum',len(jaxpr2))

    # Test autovoc
    jaxpr1, jaxpr2 = test._test_compile_vautocov()
    print('autocov', len(jaxpr1))
    print('vautocov', len(jaxpr2))

    # Test predict
    jaxpr1, jaxpr2 = test.test_compile_predict()
    print('predict', len(jaxpr1))
    print('vpredict', len(jaxpr2))

    # Test predict2
    jaxpr = test.test_compile_predict2()
    print('predict2', len(jaxpr))

    # Test condition_on
    jaxpr1 = test.test_compile_condition_on()
    print('condition_on', len(jaxpr1))

    # Test _step
    jaxpr = test.test_compile_step()
    print('step', len(jaxpr))