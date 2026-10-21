"""
Standalone script for running filters on simulated data.

This script encapsulates simulator setup and filtering logic, allowing easy
modification and testing of different simulators and filters without needing
to interact with the experiment framework.
"""

from datetime import date
import os
import random
import sys
sys.path.append(r'/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax')

from gaussfiltax import ssm
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore
from functools import partial
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN # type: ignore

import scienceplots # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib_inline # type: ignore

import time
from typing import Dict, Tuple, Optional

# Import required modules
from simulators.ssm import LGSSM, NonlinearSSM # type: ignore
from filters.gsf import LGSF, UGSF # type: ignore
from filters.agsf import LAGSF, UAGSF # type: ignore
from filters.pf import BPF, APF # type: ignore
from util.param import initialize, sample_prior # type: ignore
from util.bijectors import RealToPSDBijector # type: ignore
import util.io # type: ignore
from util.numerics import kde_error, min_error, log_prob_t, mse # type: ignore
from jax.scipy.special import logsumexp as lse # type: ignore
from jax import vmap # type: ignore

# Reduce PDF size while preserving vector text/paths.
mpl.rcParams["pdf.compression"] = 9
mpl.rcParams["pdf.fonttype"] = 42


# =============================================================================
# SIMULATOR DEFINITIONS
# =============================================================================

class SimulatorConfig:
    """Base class for simulator configurations."""
    
    def __init__(self, state_dim: int, emission_dim: int, input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
    
    def setup(self) -> Dict:
        """Set up the simulator and return SSM, props, and metadata."""
        raise NotImplementedError
    
    def get_inputs(self, key, num_timesteps: int):
        """Generate inputs for the simulator."""

        # return jnp.ones(num_timesteps).at[num_timesteps // 2].set(10)
        return jnp.ones(num_timesteps) # No inputs by default


class LGSSMConfig(SimulatorConfig):
    """Linear Gaussian State Space Model configuration."""
    
    def setup(self) -> Dict:
        """Set up LGSSM."""
        
        # Define parameter structure
        param_names = [['mean', 'cov'],
                      ['weights', 'bias', 'input_weights', 'cov'],
                      ['weights', 'bias', 'input_weights', 'cov']]
        
        constrainers = [[None, RealToPSDBijector],
                       [None, None, None, RealToPSDBijector],
                       [None, None, None, RealToPSDBijector]]
        
        # Initialize parameter values
        initial_mean = jnp.zeros(self.state_dim)
        initial_covariance = jnp.eye(self.state_dim) * 0.1

        dynamics_weights = 0.9 * jnp.eye(self.state_dim)
        dynamics_bias = jnp.zeros(self.state_dim)
        dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        dynamics_covariance = jnp.eye(self.state_dim) * 0.1

        emission_weights = jnp.eye(self.emission_dim, self.state_dim)
        emission_bias = jnp.zeros(self.emission_dim)
        emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        emission_covariance = jnp.eye(self.emission_dim) * 0.1

        init_vals = [[initial_mean, initial_covariance],
                     [dynamics_weights, dynamics_bias, dynamics_input_weights, dynamics_covariance],
                     [emission_weights, emission_bias, emission_input_weights, emission_covariance]]
        
        props = initialize(init_vals, param_names, constrainers)
        
        return {
            'ssm': LGSSM(self.state_dim, self.emission_dim, self.input_dim),
            'props': props,
            'param_names': param_names,
            'constrainers': constrainers,
            'init_vals': init_vals
        }


class COVINFLConfig(SimulatorConfig):
    """Covariance Inflation model with nonlinear dynamics."""
    
    def __init__(self, state_dim: int, emission_dim: int, a: float = .9, 
                 b: float = 1.0, input_dim: int = 0):
        super().__init__(state_dim, emission_dim, input_dim)
        self.a = a  # Dynamics coefficient
        self.b = b  # Emission coefficient
    
    def setup(self) -> Dict:
        """Set up COVINFL model."""
        
        # Define nonlinear dynamics and emission functions
        f = lambda x, q, u, params: jnp.sin(params.dynamics.coeff.value * x) + q 
        g = lambda x, r, u, params: params.emissions.coeff.value * x.T @ x + r
        
        # Define parameter structure
        param_names = [['mean', 'cov'],
                      ['coeff', 'bias', 'cov'],
                      ['coeff', 'bias', 'cov']]
        
        constrainers = [[None, RealToPSDBijector],
                       [None, None, RealToPSDBijector],
                       [None, None, RealToPSDBijector]]
        
        # Initialize parameter values
        initial_mean = jnp.zeros(self.state_dim)
        initial_covariance = jnp.eye(self.state_dim)

        dynamics_coeff = jnp.array([self.a])
        dynamics_bias = jnp.zeros(self.state_dim)
        dynamics_covariance = jnp.eye(self.state_dim)

        emission_coeff = jnp.array([self.b])
        emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance = 1.0 * jnp.eye(self.emission_dim)
        
        init_vals = [[initial_mean, initial_covariance],
                    [dynamics_coeff, dynamics_bias, dynamics_covariance],
                    [emission_coeff, emission_bias, emission_covariance]]
        
        props = initialize(init_vals, param_names, constrainers)
        
        return {
            'ssm': NonlinearSSM(self.state_dim, self.state_dim, 
                               self.emission_dim, self.emission_dim, f, g),
            'props': props,
            'param_names': param_names,
            'constrainers': constrainers,
            'init_vals': init_vals
        }


class MTTConfig(SimulatorConfig):
    """Multiple Target Tracking model with coordinated turn maneuvers."""
    
    def __init__(self, Q_scale: float = 1e-6, R_scale: float = 25*1e-6, acc: float = 0.05):
        """
        Initialize MTT configuration.
        
        Args:
            Q_scale: Process noise covariance scale
            R_scale: Measurement noise covariance scale
        """
        super().__init__(state_dim=4, emission_dim=2, input_dim=1)
        self.Q_scale = Q_scale
        self.R_scale = R_scale
        self.acc = acc
    
    def setup(self) -> Dict:
        """Set up MTT model."""
        
        # MTT dynamics parameters
        acc = self.acc
        dt = 1.0
        FCV = 1.0 * jnp.array([[1, dt, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, dt],
                               [0, 0, 0, 1]])
        
        # Helper functions for coordinated turn
        Omega = lambda x, acc: acc / jnp.sqrt(x[1]**2 + x[3]**2)
        FCT = lambda x, a: jnp.array([
            [1, jnp.sin(dt * Omega(x, a)) / Omega(x, a), 0, -(1-jnp.cos(dt * Omega(x, a))) / Omega(x, a)],
            [0, jnp.cos(dt * Omega(x, a)), 0, -jnp.sin(dt * Omega(x, a))],
            [0, (1-jnp.cos(dt * Omega(x, a))) / Omega(x, a), 1, jnp.sin(dt * Omega(x, a)) / Omega(x, a)],
            [0, jnp.sin(dt * Omega(x, a)), 0, jnp.cos(dt * Omega(x, a))]
        ])
        
        G = jnp.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
        
        # Dynamics: switches between constant velocity and coordinated turn based on input u
        def f(x, q, u, dyn_params):
            u_scalar = u if jnp.ndim(u) == 0 else u[0]
            return (0.5*(u_scalar-1)*(u_scalar-2) * FCV - 
                   u_scalar*(u_scalar-2) * FCT(x, acc) + 
                   0.5*u_scalar*(u_scalar-1) * FCT(x, -acc)) @ x + G @ q
        
        # Emissions: range and bearing measurements
        def g(x, r, u, em_params):
            return jnp.array([jnp.sqrt(x[0]**2 + x[2]**2), 
                                       jnp.arctan2(x[2], x[0]) 
                            ]) + r
        
        # Define parameter structure
        param_names = [['mean', 'cov'],
                      ['bias', 'cov'],
                      ['bias', 'cov']]
        
        constrainers = [[None, RealToPSDBijector],
                       [None, RealToPSDBijector],
                       [None, RealToPSDBijector]]
        
        # Initialize parameter values
        initial_mean = 1.0 * jnp.array([-0.05, 0.001, 0.7, -0.05])
        initial_covariance = 1.0 * jnp.array([[0.1, 0.0, 0.0, 0.0],
                                              [0.0, 0.005, 0.0, 0.0],
                                              [0.0, 0.0, 0.1, 0.0],
                                              [0.0, 0.0, 0.0, 0.01]])
        
        dynamics_bias = jnp.zeros(2)
        dynamics_covariance = jnp.eye(2) * self.Q_scale
        
        emission_bias = jnp.zeros(self.emission_dim)
        emission_covariance = jnp.eye(self.emission_dim) * self.R_scale
        
        init_vals = [[initial_mean, initial_covariance],
                    [dynamics_bias, dynamics_covariance],
                    [emission_bias, emission_covariance]]
        
        props = initialize(init_vals, param_names, constrainers)

        return {
            'ssm': NonlinearSSM(state_dim=4, state_noise_dim=2, 
                               emission_dim=2, emission_noise_dim=2, dynamics_fn=f, emission_fn=g),
            'props': props,
            'param_names': param_names,
            'constrainers': constrainers,
            'init_vals': init_vals
        }
    
    def get_inputs(self, key, num_timesteps: int):
        """
        Generate maneuver inputs for MTT.
        
        Returns sequence switching between different maneuvers:
        - u=0: left turn
        - u=1: constant velocity  
        - u=2: right turn
        """
        # Maneuver sequence: CV -> left turn -> CV
        # inputs = jr.choice(key, jnp.array([0, 1, 2]), shape = (num_timesteps, 1))
        inputs = jnp.array([1]*int(2*num_timesteps/5) + 
                          [0]*int(num_timesteps/5) + 
                          [2]*int(2*num_timesteps/5))
        return inputs


class LMSVConfig(SimulatorConfig):
    """Linear Multivariate Stochastic Volatility model."""
    
    def __init__(self, state_dim: int = 10, emission_dim: int = 10, 
                 sigma: float = 4.0, beta: float = 0.5):
        """
        Initialize LMSV configuration.
        
        Args:
            state_dim: Dimension of state space
            emission_dim: Dimension of observation space
            sigma: Volatility scaling parameter
            beta: Volatility coefficient
        """
        super().__init__(state_dim, emission_dim, input_dim=1)
        self.sigma = sigma
        self.beta = beta
    
    def setup(self) -> Dict:
        """Set up LMSV model."""
        
        # LMSV dynamics: linear autoregressive
        # x_{t+1} = Phi * x_t + q_t
        def f(x, q, u, params):
            return params.dynamics.weights.value @ x + q
        
        # LMSV emission: input-controlled stochastic volatility
        # y_t = u * beta * diag(exp(x / sigma)) * r + (1 - u) * (H * x + r)
        def g(x, r, u, params):

            sigma = params.emissions.sigma.value
            beta = params.emissions.beta.value
            H = params.emissions.weights.value
            
            # Multiplicative stochastic volatility term
            volatility_term = beta * jnp.multiply(jnp.exp(x / sigma), r)
            
            # Linear term
            linear_term = H @ x + r
            
            # Mix based on input u
            return u * volatility_term + (1.0 - u) * linear_term

        def log_prob(params, emissions, state, inputs=None):
            """
            Log-probability of emissions for the mixed linear/MSV observation model.
            """
            u = inputs if inputs is not None else 0.0
            sigma = params.emissions.sigma.value
            beta = params.emissions.beta.value
            H = params.emissions.weights.value
            bias = params.emissions.bias.value
            cov = params.emissions.cov.value

            emission_dim = cov.shape[0]
            M = u * beta * jnp.diag(jnp.exp(state / sigma)) + (1.0 - u) * jnp.eye(emission_dim)
            mean = (1.0 - u) * (H @ state) + M @ bias
            scaled_cov = M @ cov @ M.T

            dist = MVN(mean, scaled_cov)
            return dist.log_prob(emissions)

        # Define parameter structure
        param_names = [
            ['mean', 'cov'],
            ['weights', 'bias', 'cov'],
            ['weights', 'bias', 'cov', 'sigma', 'beta']
        ]
        
        constrainers = [
            [None, RealToPSDBijector],
            [None, None, RealToPSDBijector],
            [None, None, RealToPSDBijector, None, None]
        ]
        
        # Initialize parameter values
        initial_mean = 0.0 * jnp.ones(self.state_dim)
        initial_covariance = 1.0 * jnp.eye(self.state_dim)

        dynamics_weights = 0.8 * jnp.eye(self.state_dim)
        dynamics_bias = 0.0 * jnp.ones(self.state_dim)
        dynamics_covariance = 1.0 * jnp.eye(self.state_dim)

        emission_weights = jnp.eye(self.emission_dim, self.state_dim)
        emission_bias = 0.0001 * jnp.ones(self.emission_dim)
        emission_covariance = 1e-1 * jnp.eye(self.emission_dim)
        
        # LMSV-specific parameters
        sigma_val = jnp.array([self.sigma])
        beta_val = jnp.array([self.beta])
        
        init_vals = [
            [initial_mean, initial_covariance],
            [dynamics_weights, dynamics_bias, dynamics_covariance],
            [emission_weights, emission_bias, emission_covariance, sigma_val, beta_val]
        ]
        
        props = initialize(init_vals, param_names, constrainers)
        ssm = NonlinearSSM(
                state_dim=self.state_dim, 
                state_noise_dim=self.state_dim,
                emission_dim=self.emission_dim, 
                emission_noise_dim=self.emission_dim,
                dynamics_fn=f, 
                emission_fn=g
            )
        
        ssm.emission_log_prob = log_prob
        
        return {
            'ssm': ssm,
            'props': props,
            'param_names': param_names,
            'constrainers': constrainers,
            'init_vals': init_vals
        }
    
    def get_inputs(self, key, num_timesteps: int):
        """
        Generate inputs for LMSV controlling volatility regime.
        
        Returns:
            Input sequence where:
            - u=0: low volatility (linear regime)
            - u=1: high volatility (stochastic volatility regime)
            - u in (0,1): mixture of both
        """

        # Sinusoidal modulation of volatility
        # inputs = vmap(lambda x: (1 - jnp.sin(0.2 * x)) / 2)(jnp.arange(num_timesteps))

        # Step function: switch every 20 timesteps between 0 and 1
        # inputs = (jnp.floor_divide(jnp.arange(num_timesteps), 20) % 2).astype(jnp.float32)
        # inputs = 1.0 * jnp.ones(num_timesteps)  # Default: high volatility
        
        # Fixed inputs 
        val = 1.0
        inputs = val * jnp.ones(num_timesteps)
        
        return inputs


# =============================================================================
# FILTER RUNNER
# =============================================================================

class FilterRunner:
    """Runs filtering algorithms on simulated data."""
    
    def __init__(self, simulator_config: SimulatorConfig, 
                 num_timesteps: int = 100, seed: int = 0):
        """
        Initialize the filter runner.
        
        Args:
            simulator_config: Configuration object for the simulator
            num_timesteps: Number of timesteps to simulate
            seed: Random seed for reproducibility
        """
        self.simulator_config = simulator_config
        self.num_timesteps = num_timesteps
        self.seed = seed
        self.key = jr.PRNGKey(seed)
        
        # Setup simulator
        sim_setup = self.simulator_config.setup()
        self.ssm = sim_setup['ssm']
        self.props = sim_setup['props']
        
        # Generate ground truth data
        self.key, subkey = jr.split(self.key)
        self.true_params = sample_prior(subkey, self.props, 1)[0]
        
        self.key, subkey = jr.split(self.key)
        inputs = self.simulator_config.get_inputs(subkey, num_timesteps)
        self.inputs = inputs
        
        self.key, subkey = jr.split(self.key)
        self.states, self.observations = self.ssm.simulate(
            subkey, self.true_params, num_timesteps, inputs
        )
        
        print(f"✓ Simulator initialized")
        print(f"  - State shape: {self.states.shape}")
        print(f"  - Observations shape: {self.observations.shape}")
    
    def run_lgsf(self, num_components: int = 10, num_iterations: int = 1) -> Dict:
        """
        Run Linear Gaussian Sum Filter (LGSF).
        
        Args:
            num_components: Number of Gaussian components
            num_iterations: Number of filter iterations
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running LGSF (components={num_components}, iterations={num_iterations})...")
        
        start_time = time.time()
        filter_obj = LGSF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results = filter_obj.run(
            self.true_params,
            self.observations,
            num_components=num_components,
            num_iter=num_iterations,
            rng_key=subkey,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        print(f"✓ LGSF completed in {elapsed:.2f}s")
        
        return results
    
    def run_ugsf(self, num_components: int = 10, num_iterations: int = 1,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0) -> Dict:
        """
        Run Unscented Gaussian Sum Filter (UGSF).
        
        Args:
            num_components: Number of Gaussian components
            num_iterations: Number of filter iterations
            alpha: Unscented transform parameter
            beta: Unscented transform parameter
            kappa: Unscented transform parameter
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running UGSF (components={num_components}, iterations={num_iterations})...")
        
        start_time = time.time()
        filter_obj = UGSF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results = filter_obj.run(
            key = subkey,
            params = self.true_params,
            emissions = self.observations,
            num_components=num_components,
            num_iter=num_iterations,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        print(f"✓ UGSF completed in {elapsed:.2f}s")
        
        return results
    
    def run_lagsf(self, M: int = 10, N: int = 5, L: int = 5, 
                  autocov1: str = 'prop', r1: float = 0.95,
                    autocov2: str = 'prop', r2: float = 0.95,   
                  num_iterations: int = 1) -> Dict:
        """
        Run Linear Adaptive Gaussian Sum Filter (LAGSF).
        
        Args:
            M: Number of mixture components in GMM approximation
            N: Number of particles in each component
            L: Number of particles in approximation
            num_iterations: Number of filter iterations
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running LAGSF (M={M}, N={N}, L={L}, iterations={num_iterations})...")
        
        start_time = time.time()
        filter_obj = LAGSF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results, aux_outputs = filter_obj.run(
            params=self.true_params,
            emissions=self.observations,
            M=M,
            N=N,
            L=L,
            autocov1=autocov1,
            r1 = r1,
            autocov2=autocov2,
            r2 = r2,
            num_iter=num_iterations,
            key=subkey,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        results['aux_outputs'] = aux_outputs
        results['autocov1'] = autocov1
        results['autocov2'] = autocov2
        self.last_autocov1 = autocov1
        self.last_autocov2 = autocov2
        print(f"✓ LAGSF completed in {elapsed:.2f}s")
        
        return results
    
    def run_uagsf(self, M: int = 10, N: int = 5, L: int = 5,
                  alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
                  autocov1: str = 'prop', r1: float = 0.95,
                  autocov2: str = 'prop', r2: float = 0.95,
                  num_iterations: int = 1) -> Dict:
        """
        Run Unscented Adaptive Gaussian Sum Filter (UAGSF).
        
        Args:
            M: Number of mixture components in GMM approximation
            N: Number of particles in each component
            L: Number of particles in approximation
            alpha: Unscented transform parameter
            beta: Unscented transform parameter
            kappa: Unscented transform parameter
            autocov1: Autocorrelation method for first stage
            r1: Autocorrelation coefficient for first stage
            autocov2: Autocorrelation method for second stage
            r2: Autocorrelation coefficient for second stage
            num_iterations: Number of filter iterations
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running UAGSF (M={M}, N={N}, L={L}, iterations={num_iterations})...")
        
        start_time = time.time()
        filter_obj = UAGSF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results, aux_outputs = filter_obj.run(
            params=self.true_params,
            emissions=self.observations,
            M=M,
            N=N,
            L=L,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            autocov1=autocov1,
            r1=r1,
            autocov2=autocov2,
            r2=r2,
            num_iter=num_iterations,
            key=subkey,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        results['aux_outputs'] = aux_outputs
        results['autocov1'] = autocov1
        results['autocov2'] = autocov2
        self.last_autocov1 = autocov1
        self.last_autocov2 = autocov2
        print(f"✓ UAGSF completed in {elapsed:.2f}s")
        
        return results
    
    def run_bpf(self, num_particles: int = 100, ess_threshold: float = 0.5) -> Dict:
        """
        Run Bootstrap Particle Filter (BPF).
        
        Args:
            num_particles: Number of particles
            ess_threshold: ESS threshold for resampling (0 to 1)
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running BPF (particles={num_particles}, ess_threshold={ess_threshold})...")
        
        start_time = time.time()
        filter_obj = BPF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results, lls = filter_obj.run(
            params = self.true_params,
            observations = self.observations,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            key=subkey,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        print(f"✓ BPF completed in {elapsed:.2f}s")

        return results, lls
    
    def run_apf(self, num_particles: int = 100, ess_threshold: float = 0.5) -> Dict:
        """
        Run Auxiliary Particle Filter (APF).
        
        Args:
            num_particles: Number of particles
            ess_threshold: ESS threshold for resampling (0 to 1)
            
        Returns:
            Dictionary with filter results and timing
        """
        print(f"\n Running APF (particles={num_particles}, ess_threshold={ess_threshold})...")
        
        start_time = time.time()
        filter_obj = APF(self.props, self.ssm)
        self.key, subkey = jr.split(self.key)
        
        results, lls = filter_obj.run(
            self.true_params,
            self.observations,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            key=subkey,
            inputs=self.inputs
        )
        
        elapsed = time.time() - start_time
        results['time'] = elapsed
        print(f"✓ APF completed in {elapsed:.2f}s")
        
        return results, lls
    
    def plot_errors(self, results: Dict, filter_name: str = "Filter"):
        """
        Plot comprehensive error metrics over time using numerics.py functions.
        
        Args:
            results: Output from a filter run
            filter_name: Name of the filter for the title
        """
        error_metrics = self.compute_errors(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Plot 1: MSE over time
        if 'mse_t' in error_metrics:
            mse_t = error_metrics['mse_t']
            axes[0, 0].plot(mse_t, linewidth=2, color='blue')
            axes[0, 0].axhline(y=error_metrics.get('mse', jnp.mean(mse_t)), 
                              color='red', linestyle='--', 
                              label=f"Mean MSE = {error_metrics.get('mse', 0):.6f}")
            axes[0, 0].set_title('Mean Squared Error Over Time', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'MSE not available', ha='center', va='center')
            axes[0, 0].set_title('Mean Squared Error', fontsize=12, fontweight='bold')
        
        # Plot 2: Min error over time
        if 'min_error_t' in error_metrics:
            min_err_t = error_metrics['min_error_t']
            axes[0, 1].plot(min_err_t, linewidth=2, color='green')
            axes[0, 1].axhline(y=error_metrics.get('min_error', jnp.mean(min_err_t)), 
                              color='red', linestyle='--', 
                              label=f"Mean Min Error = {error_metrics.get('min_error', 0):.6f}")
            axes[0, 1].set_title('Minimum Component Error Over Time', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Min Error')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Min Error not available', ha='center', va='center')
            axes[0, 1].set_title('Minimum Component Error', fontsize=12, fontweight='bold')
        
        # Plot 3: Log probability error over time
        if 'log_prob_error_t' in error_metrics:
            lp_err_t = error_metrics['log_prob_error_t']
            axes[1, 0].plot(lp_err_t, linewidth=2, color='purple')
            axes[1, 0].axhline(y=error_metrics.get('log_prob_error', jnp.mean(lp_err_t)), 
                              color='red', linestyle='--', 
                              label=f"Mean LP Error = {error_metrics.get('log_prob_error', 0):.6f}")
            axes[1, 0].set_title('Log Probability Error Over Time', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Log Prob Error')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Log Prob Error not available', ha='center', va='center')
            axes[1, 0].set_title('Log Probability Error', fontsize=12, fontweight='bold')
        
        # Plot 4: Error summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""{filter_name} Error Summary

"""
        if 'mse' in error_metrics:
            summary_text += f"MSE:           {error_metrics['mse']:.8f}\n"
        if 'min_error' in error_metrics:
            summary_text += f"Min Error:     {error_metrics['min_error']:.8f}\n"
        if 'log_prob_error' in error_metrics:
            summary_text += f"Log Prob Error: {error_metrics['log_prob_error']:.8f}\n"
        
        summary_text += f"\nExecution Time: {results.get('time', 'N/A'):.3f}s"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f'{filter_name} - Numerics Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig

    def plot_error_comparison(self, results_by_name: Dict[str, Dict], title: str):
        """
        Plot error metrics over time with multiple algorithms overlaid.
        """
        if not results_by_name:
            print("Warning: No results provided for error comparison")
            return

        metrics = [
            ("mse_t", "MSE"),
            ("log_prob_error_t", "Log Prob Error"),
        ]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 8), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        inputs_spans = None
        if hasattr(self, "inputs") and self.inputs is not None:
            inputs = jnp.asarray(self.inputs)
            if inputs.shape[0] > 0:
                inputs_np = jnp.asarray(inputs).tolist()
                spans = []
                start = 0
                current = inputs_np[0]
                for i, val in enumerate(inputs_np[1:], start=1):
                    if val != current:
                        spans.append((start, i, current))
                        start = i
                        current = val
                spans.append((start, len(inputs_np), current))
                inputs_spans = spans

        for name, results in results_by_name.items():
            try:
                error_metrics = self.compute_errors(results)
            except Exception as exc:
                print(f"Warning: Could not compute errors for {name}: {exc}")
                continue

            for idx, (key, label) in enumerate(metrics):
                if key in error_metrics:

                    axes[idx].plot(error_metrics[key], linewidth=2, label=name)

        for idx, (_, label) in enumerate(metrics):
            axes[idx].set_ylabel(label)
            axes[idx].grid(True, alpha=0.3)
            if inputs_spans is not None:
                for start, end, val in inputs_spans:
                    color = '#9ecae1' if val == 0 else '#ffe082'
                    axes[idx].axvspan(start, end, color=color, alpha=0.25, zorder=0)
            axes[idx].legend(loc="upper right")

        axes[-1].set_xlabel("Time")
        if inputs_spans is not None:
            from matplotlib.patches import Patch
            axes[0].legend(
                handles=[
                    Patch(facecolor='#9ecae1', edgecolor='none', alpha=0.25, label='linear'),
                    Patch(facecolor='#ffe082', edgecolor='none', alpha=0.25, label='non-linear'),
                ],
                loc="upper right",
            )
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

    def plot_pf_u_weights(self, results: Dict, filter_name: str = "PF", max_particles: int = 500):
        """
        Plot unnormalized PF weights over time in a separate figure.
        """
        if "u_weights" not in results:
            print(f"Warning: No unnormalized weights available for {filter_name}")
            return

        u_weights = jnp.asarray(results["u_weights"])
        if u_weights.ndim != 2:
            print(f"Warning: Unexpected u_weights shape for {filter_name}: {u_weights.shape}")
            return

        num_particles, num_timesteps = u_weights.shape
        if num_particles > max_particles:
            idx = jnp.linspace(0, num_particles - 1, max_particles).astype(int)
            u_plot = u_weights[idx]
        else:
            u_plot = u_weights

        eps = 1e-12
        log_u = jnp.log10(jnp.maximum(u_plot, eps))
        total_log_u = jnp.log10(jnp.maximum(jnp.sum(u_weights, axis=0), eps))

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        im = axes[0].imshow(
            log_u,
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        axes[0].set_ylabel("Particle (subsampled)")
        axes[0].set_title("log10 Unnormalized Weights (per particle)")
        fig.colorbar(im, ax=axes[0], label="log10(u_weights)")

        axes[1].plot(total_log_u, linewidth=2, color="steelblue")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("log10 Sum")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("log10 Sum of Unnormalized Weights")

        fig.suptitle(f"{filter_name} - Unnormalized Weight Diagnostics", fontsize=12, fontweight="bold")
        plt.tight_layout()

    def plot_mean_comparison(self, results_by_name: Dict[str, Dict], title: str):
        """
        Plot ground truth with mean estimates from multiple algorithms.
        """
        if not results_by_name:
            print("Warning: No results provided for mean comparison")
            return

        time = jnp.arange(self.states.shape[0])
        state_dim = self.states.shape[1]
        n_rows = (state_dim + 1) // 2
        n_cols = 2 if state_dim > 1 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        colors = plt.cm.tab10(jnp.linspace(0, 1, len(results_by_name)))

        for idx, (name, results) in enumerate(results_by_name.items()):
            if "particles" in results:
                mean_est = jnp.mean(results["particles"], axis=0)
            elif "means" in results:
                means = results["means"]
                weights = results.get("weights")
                if weights is not None and weights.ndim == 2 and weights.shape[1] == means.shape[1]:
                    w = weights
                    w_sum = jnp.sum(w, axis=0, keepdims=True)
                    w = jnp.where(w_sum > 0, w / w_sum, w)
                    mean_est = jnp.einsum("kt,ktd->td", w, means)
                else:
                    mean_est = jnp.mean(means, axis=0)
            else:
                continue

            for d in range(state_dim):
                
                axes[d].plot(time, mean_est[:, d], color=colors[idx], alpha=0.9, label=name)

        for d in range(state_dim):
            axes[d].plot(time, self.states[:, d], color="black", linewidth=2, label="Ground Truth")
            axes[d].set_title(f"State {d}")
            axes[d].set_xlabel("Time")
            axes[d].set_ylabel("Value")
            axes[d].grid(True, alpha=0.3)
            axes[d].legend(loc="upper right", fontsize=8)

        for ax in axes[state_dim:]:
            ax.axis("off")

        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()

    def plot_autocov_diagnostics(self, results: Dict, filter_name: str = "Filter"):
        """
        Plot autocov diagnostics when autocov1/2 are set to 'auto'.
        """
        aux_outputs = results.get("aux_outputs")
        if not aux_outputs:
            print(f"Warning: No autocov diagnostics available for {filter_name}")
            return

        autocov1 = results.get("autocov1", getattr(self, "last_autocov1", None))
        autocov2 = results.get("autocov2", getattr(self, "last_autocov2", None))
        show_delta = autocov1 == "auto" and "Deltas" in aux_outputs and "rho1s" in aux_outputs
        show_lambda = autocov2 == "auto" and "Lambdas" in aux_outputs and "rho2s" in aux_outputs

        if not show_delta and not show_lambda:
            return

        series = []
        if show_delta:
            deltas = aux_outputs["Deltas"]
            rho1s = aux_outputs["rho1s"]
            delta_traces = jnp.trace(deltas, axis1=-2, axis2=-1)
            mean_delta_traces = jnp.mean(delta_traces, axis=1) if delta_traces.ndim > 1 else delta_traces
            rho1_series = jnp.mean(rho1s, axis=1) if rho1s.ndim > 1 else rho1s
            time = jnp.arange(mean_delta_traces.shape[0])
            series.append(("Trace(Delta)", mean_delta_traces, "darkorange"))
            series.append(("rho1", rho1_series, "teal"))

        if show_lambda:
            lambdas = aux_outputs["Lambdas"]
            rho2s = aux_outputs["rho2s"]
            lambda_traces = jnp.trace(lambdas, axis1=-2, axis2=-1)
            mean_lambda_traces = jnp.mean(lambda_traces, axis=1) if lambda_traces.ndim > 1 else lambda_traces
            rho2_series = jnp.mean(rho2s, axis=1) if rho2s.ndim > 1 else rho2s
            time = jnp.arange(mean_lambda_traces.shape[0])
            series.append(("Trace(Lambda)", mean_lambda_traces, "darkorange"))
            series.append(("rho2", rho2_series, "teal"))

        fig, axes = plt.subplots(len(series), 1, figsize=(8, 3 * len(series)), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for idx, (label, values, color) in enumerate(series):
            ax = axes[idx]
            ax.plot(time, values, linewidth=2, color=color)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            if label == "rho2" and hasattr(self, "inputs") and self.inputs is not None:
                inputs = jnp.asarray(self.inputs)
                if inputs.shape[0] == values.shape[0]:
                    inputs_np = jnp.asarray(inputs).tolist()
                    spans = []
                    start = 0
                    current = inputs_np[0]
                    for i, val in enumerate(inputs_np[1:], start=1):
                        if val != current:
                            spans.append((start, i, current))
                            start = i
                            current = val
                    spans.append((start, len(inputs_np), current))

                    for start, end, val in spans:
                        color = '#9ecae1' if val == 0 else '#ffe082'
                        ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)

                    from matplotlib.patches import Patch
                    ax.legend(handles=[
                        Patch(facecolor='#9ecae1', edgecolor='none', alpha=0.3, label='input=0'),
                        Patch(facecolor='#ffe082', edgecolor='none', alpha=0.3, label='input=1'),
                    ], loc="upper right")

        axes[-1].set_xlabel("Time")
        fig.suptitle(f"{filter_name} - Autocov Diagnostics", fontsize=12, fontweight='bold')
        plt.tight_layout()

    def plot_pairwise_scatter(self, results: Dict, title: Optional[str] = None,
                               max_components: int = 50, max_particles: int = 500,
                               ellipse_every: int = 20, ellipse_n_sigma: float = 2.0,
                               ellipse_alpha: float = 0.25):
        """
        Create scatter plots for all pairs of state dimensions.

        For GSF/AGSF: overlays ground truth states, component means,
        and covariance ellipses (subsampled over time).
        For PF (BPF/APF): overlays ground truth states and particle trajectories.

        Args:
            results: Output dict from a filter run.
            title: Optional title for the figure.
            max_components: Limit number of GSF components plotted.
            max_particles: Limit number of PF particles plotted.
            ellipse_every: Plot an ellipse every N timesteps (GSF only).
            ellipse_n_sigma: Ellipse scale in sigmas (e.g., 2.0 for ~95%).
            ellipse_alpha: Alpha for ellipse outlines.
        """
        states = self.states  # (T, d)
        d = states.shape[1]

        # Build all unique dimension pairs (i < j)
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        n_pairs = len(pairs)
        if n_pairs == 0:
            print("No dimension pairs available for scatter plots.")
            return

        # Decide grid layout
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.ravel()

        # Detect result type
        is_pf = 'particles' in results
        is_gsf = ('means' in results) and ('covariances' in results or 'means' in results)

        if is_gsf:
            means = results['means']  # (K, T, d)
            covs = results.get('covariances', None)  # (K, T, d, d) expected
            K = means.shape[0]
            T = means.shape[1]
            K_plot = min(K, max_components)
            gsf_color = "tab:blue"

            # Local import to avoid global dependency
            from matplotlib.patches import Ellipse

            for idx, (i, j) in enumerate(pairs):
                ax = axes[idx]

                # Ground truth scatter
                ax.scatter(states[:, i], states[:, j], c='red', s=4, label='GT', alpha=0.8, zorder=10)

                # Component means scatter (subset for readability)
                for k in range(K_plot):
                    ax.scatter(means[k, :, i], means[k, :, j], c=gsf_color, s=0.6, alpha=0.5)

                # Covariance ellipses (if provided)
                if covs is not None:
                    step = max(1, int(ellipse_every))
                    for k in range(K_plot):
                        col = gsf_color
                        for t in range(0, T, step):
                            mx = float(means[k, t, i])
                            my = float(means[k, t, j])
                            # Extract 2x2 covariance for dims (i, j)
                            s11 = float(covs[k, t, i, i])
                            s12 = float(covs[k, t, i, j])
                            s21 = float(covs[k, t, j, i])
                            s22 = float(covs[k, t, j, j])
                            S = jnp.array([[s11, s12], [s21, s22]])
                            # Ensure symmetry
                            S = (S + S.T) / 2.0
                            # Eigen decomposition
                            vals, vecs = jnp.linalg.eigh(S)
                            # Sort by descending eigenvalue
                            order = jnp.argsort(vals)[::-1]
                            vals = vals[order]
                            vecs = vecs[:, order]
                            # Radii along principal axes (n-sigma)
                            a = ellipse_n_sigma * jnp.sqrt(jnp.maximum(vals[0], 0.0))
                            b = ellipse_n_sigma * jnp.sqrt(jnp.maximum(vals[1], 0.0))
                            # Angle of major axis
                            vx = vecs[0, 0]
                            vy = vecs[1, 0]
                            angle = jnp.degrees(jnp.arctan2(vy, vx))
                            # Create ellipse (outline only for less clutter)
                            e = Ellipse((mx, my), width=float(2*a), height=float(2*b),
                                        angle=float(angle), edgecolor=col, facecolor='none',
                                        lw=0.8, alpha=ellipse_alpha)
                            ax.add_patch(e)

                ax.set_xlabel(f"x{i}")
                ax.set_ylabel(f"x{j}")
                ax.grid(True, alpha=0.3)
                ax.set_title(f"Dims ({i}, {j})")

            # Legend on first subplot
            axes[0].legend(fontsize=8)

        elif is_pf:
            particles = results['particles']  # (M, T, d)
            pf_color = "tab:blue"
            M = particles.shape[0]
            M_plot = min(M, max_particles)

            for idx, (i, j) in enumerate(pairs):
                ax = axes[idx]

                # Ground truth scatter
                ax.scatter(states[:, i], states[:, j], c='red', s=4, label='GT', alpha=0.8, zorder=10)

                # Particle scatter (subset)
                for p in range(M_plot):
                    ax.scatter(particles[p, :, i], particles[p, :, j], c=pf_color, s=0.4, alpha=0.2)

                ax.set_xlabel(f"x{i}")
                ax.set_ylabel(f"x{j}")
                ax.grid(True, alpha=0.3)
                ax.set_title(f"Dims ({i}, {j})")

            axes[0].legend(fontsize=8)

        else:
            print("Warning: results do not contain GSF or PF fields for pairwise scatter.")
            return

        # Hide any unused axes
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')

        fig_title = title if title else "Pairwise State Scatter"
        fig.suptitle(fig_title, fontsize=14, fontweight='bold')
        plt.tight_layout()

    def plot_three_panel_scatter(
        self,
        comparison_results,
        title: Optional[str] = None,
        dims: Tuple[int, int] = (0, 1),
        max_components: int = 50,
        max_particles: int = 500,
        rasterize_points: bool = False,
        max_points_per_cloud: Optional[int] = 50000,
        save_path: Optional[str] = None,
        dpi: int = 150,
        rasterized_dpi: int = 200,
        show: bool = True,
        states_list: Optional[list] = None,
        row_titles: Optional[list] = None,
        algorithm_order: Optional[list] = None,
    ):
        """
        Create an RxC scatter comparison grid for filter outputs.

        For each panel:
        - GSF/AGSF results (means/covariances) plot GT, component means, and resampled particles.
        - PF results (particles) plot GT and particle cloud.
        """
        is_list = isinstance(comparison_results, (list, tuple))
        rows = len(comparison_results) if is_list else 1
        if not is_list:
            comparison_results = [comparison_results]
        if states_list is None:
            states_list = [self.states] * rows
        if row_titles is None:
            row_titles = [None] * rows
        if algorithm_order is None:
            algorithm_order = list(comparison_results[0].keys()) if comparison_results else []
        if not algorithm_order:
            print("Warning: no algorithms provided for comparison scatter.")
            return

        d = states_list[0].shape[1]
        i, j = dims
        if d <= max(i, j):
            print("Warning: requested scatter dims are out of range for this model.")
            return

        n_cols = len(algorithm_order)
        fig, axes = plt.subplots(rows, n_cols, figsize=(5 * n_cols, 4 * rows), squeeze=False)

        def downsample_xy(x_vals, y_vals):
            if max_points_per_cloud is None or max_points_per_cloud <= 0:
                return x_vals, y_vals
            n = int(x_vals.shape[0])
            if n <= max_points_per_cloud:
                return x_vals, y_vals
            step = (n + max_points_per_cloud - 1) // max_points_per_cloud
            idx = jnp.arange(0, n, step)[:max_points_per_cloud]
            return x_vals[idx], y_vals[idx]

        def resample_gsf_particles(results: Dict, num_samples: int, key: jr.PRNGKey):
            means = results.get('means', None)
            covs = results.get('covariances', None)
            weights = results.get('weights', None)
            if means is None or covs is None:
                return None
            if means.ndim != 3 or covs.ndim != 4:
                return None

            K, T, d = means.shape
            if covs.shape[:2] != (K, T):
                return None
            if weights is not None and weights.ndim != 2:
                return None

            samples = jnp.zeros((num_samples, T, d))
            for t in range(T):
                if weights is None:
                    w_t = jnp.ones(K) / K
                else:
                    w_t = weights[:, t]
                w_sum = jnp.sum(w_t)
                w_t = jnp.where(w_sum > 0, w_t / w_sum, jnp.ones_like(w_t) / K)
                key, subkey = jr.split(key)
                comp_idx = jr.choice(subkey, jnp.arange(K), shape=(num_samples,), p=w_t)
                key, subkey = jr.split(key)
                keys = jr.split(subkey, num_samples)
                means_t = means[:, t, :]
                covs_t = covs[:, t, :, :]
                sample_t = vmap(lambda k, idx: MVN(means_t[idx], covs_t[idx]).sample(seed=k))(keys, comp_idx)
                samples = samples.at[:, t, :].set(sample_t)
            return samples

        def dim_label(dim_index: int) -> str:
            if dim_index == 0:
                return r"$x_1$"
            if dim_index == 2:
                return r"$x_2$"
            if dim_index == 1:
                return r"$v_1$"
            if dim_index == 3:
                return r"$v_2$"
            return f"x{dim_index}"

        def plot_gsf(ax, results: Dict, panel_title: Optional[str], states):
            means = results.get('means', None)
            if means is None:
                ax.text(0.5, 0.5, 'Means not available', ha='center', va='center')
                if panel_title:
                    ax.set_title(panel_title)
                ax.axis('off')
                return
            K = means.shape[0]
            K_plot = min(K, max_components)

            ax.scatter(states[:, i], states[:, j], c='red', s=4, label='GT', alpha=0.8, zorder=10)

            # Plot original component means in light blue for reference.
            means_x = means[:K_plot, :, i].reshape(-1)
            means_y = means[:K_plot, :, j].reshape(-1)
            means_x, means_y = downsample_xy(means_x, means_y)
            ax.scatter(
                means_x,
                means_y,
                c="goldenrod",
                s=8.0,
                alpha=0.4,
                linewidths=0,
                marker='o',
                rasterized=rasterize_points,
                label="means",
            )

            # Plot resampled particles in black.
            sample_key = jr.PRNGKey(0)
            resampled = resample_gsf_particles(results, 100, sample_key)
            if resampled is not None:
                resampled_x = resampled[:, :, i].reshape(-1)
                resampled_y = resampled[:, :, j].reshape(-1)
                resampled_x, resampled_y = downsample_xy(resampled_x, resampled_y)
                ax.scatter(
                    resampled_x,
                    resampled_y,
                    c="black",
                    s=2,
                    alpha=0.2,
                    linewidths=0,
                    rasterized=rasterize_points,
                    label="samples",
                )

            if panel_title:
                ax.set_title(panel_title, fontsize=14, fontweight="bold")
            ax.set_xlabel(dim_label(i))
            ax.set_ylabel(dim_label(j))
            ax.grid(True, alpha=0.3)

        def plot_pf(ax, results: Dict, panel_title: Optional[str], states):
            particles = results.get('particles', None)
            if particles is None:
                ax.text(0.5, 0.5, 'Particles not available', ha='center', va='center')
                if panel_title:
                    ax.set_title(panel_title)
                ax.axis('off')
                return
            M = particles.shape[0]
            M_plot = min(M, max_particles)

            ax.scatter(states[:, i], states[:, j], c='red', s=4, label='GT', alpha=0.8, zorder=10)
            particle_x = particles[:M_plot, :, i].reshape(-1)
            particle_y = particles[:M_plot, :, j].reshape(-1)
            particle_x, particle_y = downsample_xy(particle_x, particle_y)
            ax.scatter(
                particle_x,
                particle_y,
                c="black",
                s=2.0,
                alpha=0.2,
                linewidths=0,
                rasterized=rasterize_points,
                label='particles',
            )

            if panel_title:
                ax.set_title(panel_title, fontsize=14, fontweight="bold")
            ax.set_xlabel(dim_label(i))
            ax.set_ylabel(dim_label(j))
            ax.grid(True, alpha=0.3)

        for row_idx, row_results in enumerate(comparison_results):
            states = states_list[row_idx]
            for col_idx, algo_name in enumerate(algorithm_order):
                ax = axes[row_idx, col_idx]
                panel_title = algo_name if row_idx == 0 else None
                algo_results = row_results.get(algo_name, None)
                if algo_results is None:
                    ax.text(0.5, 0.5, 'Results not available', ha='center', va='center')
                    if panel_title:
                        ax.set_title(panel_title, fontsize=14, fontweight="bold")
                    ax.axis('off')
                    continue

                if 'means' in algo_results:
                    plot_gsf(ax, algo_results, panel_title, states)
                elif 'particles' in algo_results:
                    plot_pf(ax, algo_results, panel_title, states)
                else:
                    ax.text(0.5, 0.5, 'Unsupported result format', ha='center', va='center')
                    if panel_title:
                        ax.set_title(panel_title, fontsize=14, fontweight="bold")
                    ax.axis('off')
                    continue

                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(fontsize=8, markerscale=3.0)
            if row_titles[row_idx]:
                axes[row_idx, 0].text(
                    -0.35,
                    0.5,
                    row_titles[row_idx],
                    transform=axes[row_idx, 0].transAxes,
                    rotation=90,
                    va='center',
                    ha='right',
                )

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            save_dpi = rasterized_dpi if rasterize_points else dpi
            fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        if not show:
            plt.close(fig)
    
    def plot_results(self, results: Dict, title: str = "Filtering Results"):
        """
        Plot filtering results against ground truth with improved trajectory visualization.
        Handles both Gaussian Sum Filters and Particle Filters.
        
        Args:
            results: Output from a filter run
            title: Title for the plot
        """
        # Distinguish between GSF and PF
        is_gsf = 'means' in results and 'covariances' in results
        is_pf = 'particles' in results
        
        if is_gsf:

            self._plot_gsf_trajectory(results, title)

        elif is_pf:

            self._plot_pf_trajectory(results, title)

        else:

            print("Warning: Results format not recognized for plotting")
    
    def _plot_gsf_trajectory(self, results: Dict, title: str):
        """
        Plot Gaussian Sum Filter trajectory with uncertainty bands.
        
        Args:
            results: GSF filter results with means and covariances
            title: Plot title
        """
        means = results['means']  # (n_comp, T, state_dim)
        covs = results['covariances']  # (n_comp, T, state_dim, state_dim)
        states = self.states  # (T, state_dim)
        
        # Extract diagonal covariances
        diag_covs = jnp.einsum('ijkk->ijk', covs)
        
        T = means.shape[1]
        time = jnp.arange(T)
        n_comp = means.shape[0]
        n_dim = means.shape[2]
        
        # Create subplot grid based on state dimension
        n_rows = (n_dim + 1) // 2
        n_cols = min(n_dim, 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows), squeeze=False)
        axes = axes.ravel()
        
        colors = plt.cm.viridis(jnp.linspace(0, 1, n_comp))
        labels = [f'comp {k}' for k in range(n_comp)] if n_comp < 10 else [''] * n_comp
        
        for d in range(n_dim):
            ax = axes[d]
            
            for k in range(n_comp):
                m = means[k, :, d]
                var = diag_covs[k, :, d]
                std = jnp.sqrt(var)
                
                ax.plot(time, m, label=labels[k], color=colors[k], alpha=0.7)
                ax.fill_between(time,
                                m - 2 * std,
                                m + 2 * std,
                                alpha=0.2,
                                color=colors[k])
            
            ax.plot(time, states[:, d], label='Ground Truth', linewidth=2, 
                   color='red', linestyle='--')
            ax.set_title(f'State Dimension {d}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        axes[0].legend(fontsize=8)
        
        # Hide unused subplots
        for d in range(n_dim, len(axes)):
            axes[d].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def _plot_pf_trajectory(self, results: Dict, title: str):
        """
        Plot Particle Filter trajectory with particle cloud visualization.
        
        Args:
            results: PF filter results with particles
            title: Plot title
        """
        particles = results['particles']  # (num_particles, T, state_dim)
        states = self.states  # (T, state_dim)
        means = particles.mean(axis=0)  # (T, state_dim)
        
        T = particles.shape[1]
        time = jnp.arange(T)
        n_particles = particles.shape[0]
        n_dim = particles.shape[2]
        
        # Create subplot grid based on state dimension
        n_rows = (n_dim + 1) // 2
        n_cols = min(n_dim, 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), squeeze=False)
        axes = axes.ravel()
        
        for d in range(n_dim):

            ax = axes[d]
            
            # Plot ground truth
            ax.plot(time, states[:, d], label='Ground Truth', linewidth=2, 
                   color='red', linestyle='-')
            
            # Plot mean trajectory
            ax.plot(time, means[:, d], label='Particle Mean', 
                   linewidth=2, color='blue', linestyle='dotted')
            
            # Plot particle cloud (subsample if too many)
            n_plot = min(200, n_particles)

            for p in range(n_plot):

                ax.plot(time, particles[p, :, d], color='gray', alpha=0.1, linewidth=0.5, label='Particles' if p == 0 else "")
            
            ax.set_title(f'Dimension {d}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for d in range(n_dim, len(axes)):
            axes[d].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def plot_slice(self, results: Dict, t: int, title: str = "Posterior PDF Slice"):
        """
        Plot posterior PDF slice at a specific time step.
        Handles both Gaussian Sum Filters (parametric) and Particle Filters (KDE).
        
        Args:
            results: Output from a filter run
            t: Time step to visualize
            title: Title for the plot
        """
        # Distinguish between GSF and PF
        is_gsf = 'means' in results and 'covariances' in results
        is_pf = 'particles' in results
        
        if is_gsf:
            self._plot_gsf_slice(results, t, title)
        elif is_pf:
            self._plot_pf_slice(results, t, title)
        else:
            print("Warning: Slice plotting requires either GSF or PF results")
    
    def _plot_gsf_slice(self, results: Dict, t: int, title: str):
        """
        Plot GSF posterior PDF slice using parametric Gaussians.
        
        Args:
            results: GSF filter results
            t: Time step to visualize
            title: Plot title
        """
        means = results['means']  # (n_comp, T, state_dim)
        covs = results['covariances']  # (n_comp, T, state_dim, state_dim)
        states = self.states  # (T, state_dim)
        
        diag_covs = jnp.einsum('ijkk->ijk', covs)
        
        n_comp = means.shape[0]
        n_dim = means.shape[2]
        n_sigma = 4
        
        fig, axes = plt.subplots(1, n_dim, figsize=(5*n_dim, 4))
        if n_dim == 1:
            axes = [axes]
        
        for d in range(n_dim):
            ax = axes[d]
            comp_means = means[:, t, d]
            comp_vars = diag_covs[:, t, d]
            comp_std = jnp.sqrt(comp_vars + 1e-12)
            comp_std = jnp.clip(comp_std, 1e-3, None)
            
            lo = jnp.min(comp_means - n_sigma * comp_std)
            hi = jnp.max(comp_means + n_sigma * comp_std)
            
            # Include ground truth and add padding
            gt = states[t, d]
            lo = jnp.minimum(lo, gt)
            hi = jnp.maximum(hi, gt)
            pad = 0.1 * (hi - lo + 1e-6)
            lo = float(lo - pad)
            hi = float(hi + pad)
            
            if hi - lo < 1e-3:
                lo -= 0.5
                hi += 0.5
            
            x_vals = jnp.linspace(lo, hi, 500)
            
            colors = plt.cm.viridis(jnp.linspace(0, 1, n_comp))
            
            for k in range(n_comp):
                m = float(comp_means[k])
                std = float(comp_std[k])
                pdf = (1.0 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-0.5 * ((x_vals - m) / std) ** 2)
                ax.plot(x_vals, pdf, 
                       label=f'comp {k}' if n_comp < 10 else "", 
                       alpha=0.8, color=colors[k])
            
            ax.axvline(float(gt), color='red', linestyle='--', linewidth=2, 
                      label='Ground Truth')
            ax.set_title(f'Posterior Marginal PDF at t={t}, dim={d}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def _plot_pf_slice(self, results: Dict, t: int, title: str):
        """
        Plot PF posterior PDF slice using kernel density estimation.
        
        Args:
            results: PF filter results
            t: Time step to visualize
            title: Plot title
        """
        particles = results['particles']  # (num_particles, T, state_dim)
        states = self.states  # (T, state_dim)
        
        T = particles.shape[1]
        t = min(max(0, t), T - 1)  # Clamp t to valid range
        
        n_sigma = 4
        n_pts = 500
        n_particles = particles.shape[0]
        n_dim = particles.shape[2]
        
        fig, axes = plt.subplots(1, n_dim, figsize=(4 * n_dim, 4), squeeze=False)
        axes = axes.ravel()
        
        for d in range(n_dim):
            samples = jnp.asarray(particles[:, t, d]).astype(jnp.float32)  # shape (M,)
            
            if samples.size == 0:
                axes[d].text(0.5, 0.5, "No particles", ha="center")
                continue
            
            # Silverman bandwidth (with safeguards)
            std = jnp.std(samples)
            std = jnp.where(std == 0.0, 1e-3, std)
            h = 1.06 * std * (samples.size ** (-1/5))
            h = jnp.maximum(h, 1e-3)
            
            # Adaptive x range around mixture components and ground truth
            comp_mean = jnp.mean(samples)
            lo = float(jnp.min(samples - n_sigma * std))
            hi = float(jnp.max(samples + n_sigma * std))
            gt = float(states[t, d])  # ground truth at time t
            lo = min(lo, gt)
            hi = max(hi, gt)
            pad = 0.1 * (hi - lo + 1e-12)
            lo -= pad
            hi += pad
            if hi - lo < 1e-3:
                lo -= 0.5
                hi += 0.5
            
            x_grid = jnp.linspace(lo, hi, n_pts)
            
            # Vectorized Gaussian kernel density estimate
            diffs = (x_grid[:, None] - samples[None, :]) / h
            kernel_vals = jnp.exp(-0.5 * (diffs ** 2)) / jnp.sqrt(2 * jnp.pi)
            pdf = jnp.mean(kernel_vals, axis=1) / h
            
            ax = axes[d]
            ax.plot(x_grid, pdf, lw=1.5, label=f'KDE (t={t})')
            
            # Show ground truth as vertical marker
            ax.axvline(gt, color='k', linestyle='--', label='GT')
            
            # Optional: rug plot for particles
            y_rug = 0.02 * jnp.max(pdf)
            ax.plot(samples, jnp.full_like(samples, y_rug), '|', 
                   color='C1', alpha=0.6, 
                   label='particles' if d == 0 else "")
            
            ax.set_title(f"Dim {d} posterior (t={t})")
            ax.set_xlabel("value")
            ax.set_ylabel("density")
            if d == 0:
                ax.legend(fontsize=8)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    def compute_errors(self, results: Dict) -> Dict:
        """
        Compute error metrics using numerics.py functions.
        Handles both Gaussian Sum Filters (means/covariances) and Particle Filters (particles).
        
        Args:
            results: Output from a filter run
            
        Returns:
            Dictionary with various error metrics
        """
        error_metrics = {}
        
        # Determine filter type and prepare data structure for numerics functions
        if 'particles' in results:
            # Particle Filter (BPF, APF)
            particles = results['particles']  # Shape: (num_particles, T, state_dim)
            
            # Format for numerics.py: list with particles dict
            data = [{'particles': particles}]
            inf_type = 'bpf'  # Will be handled as particle filter
            
            print(f"  - Detected Particle Filter (particles shape: {particles.shape})")
            
        elif 'means' in results:
            # Gaussian Sum Filter (LGSF, UGSF, LAGSF, etc.)
            means = results['means']  # Shape: (num_components, T, state_dim)
            covs = results.get('covariances')
            
            # Format for numerics.py: dict with means/covariances
            data = {
                'means': means,
                'covariances': covs if covs is not None else means,
                'weights': results.get('weights')
            }
            inf_type = 'lgsf'  # Generic GSF format
            
            print(f"  - Detected Gaussian Sum Filter (means shape: {means.shape})")
        else:
            raise ValueError("Results do not contain 'particles' or 'means'")
        
        # MSE: mean squared error
        try:
            mse_val, mse_t = mse(data, inf_type, self.states)
            error_metrics['mse'] = float(mse_val)
            error_metrics['mse_t'] = mse_t
        except Exception as e:
            print(f"  Warning: Could not compute MSE: {e}")
        
        # Min error: minimum distance to nearest component
        try:
            min_err, min_err_t = min_error(data, inf_type, self.states)
            error_metrics['min_error'] = float(min_err)
            error_metrics['min_error_t'] = min_err_t
        except Exception as e:
            print(f"  Warning: Could not compute min_error: {e}")
        
        # Log probability error: negative log likelihood
        try:
            # Use get_means_covs to handle both GSF and BPF formats
            from util.numerics import get_means_covs, get_weights
            means_all, covs_all = get_means_covs(data, inf_type)
            weights_all = get_weights(data, inf_type)
            
            # means_all shape: (num_components_or_particles, T, state_dim)
            # covs_all shape: (num_components_or_particles, T, state_dim, state_dim)
            
            def lp_t_single(t):
                return -log_prob_t(means_all[:, t, :], covs_all[:, t, :, :], self.states[t], weights_all[:, t])
            
            lp_errs = vmap(lp_t_single)(jnp.arange(self.states.shape[0]))
            error_metrics['log_prob_error'] = float(jnp.mean(lp_errs))
            error_metrics['log_prob_error_t'] = lp_errs
        except Exception as e:
            print(f"  Warning: Could not compute log probability error: {e}")
        
        return error_metrics
    
    def print_summary(self, results: Dict, filter_name: str):

        """Print summary statistics of filtering results using numerics.py functions."""

        print(f"\n{filter_name} Results Summary:")
        exec_time = results.get('time', 'N/A')

        if isinstance(exec_time, float):

            print(f"  - Execution time: {exec_time:.3f}s")

        else:

            print(f"  - Execution time: {exec_time}s")
        
        try:

            error_metrics = self.compute_errors(results)
            
            if 'mse' in error_metrics:

                print(f"  - MSE: {error_metrics['mse']:.6f}")

            if 'min_error' in error_metrics:

                print(f"  - Min Error: {error_metrics['min_error']:.6f}")

            if 'log_prob_error' in error_metrics:

                print(f"  - Log Prob Error: {error_metrics['log_prob_error']:.6f}")
                
        except (ValueError, KeyError) as e:

            print(f"  - Error metrics: N/A ({str(e)})")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # CONFIGURATION: Choose which filters to run
    # =========================================================================
    
    # Individual filter toggles (ignored if RUN_ALL_FILTERS is True)
    RUN_LGSF = False              # Linear Gaussian Sum Filter
    RUN_UGSF = False              # Unscented Gaussian Sum Filter
    RUN_LAGSF = True             # Linear Adaptive Gaussian Sum Filter
    RUN_UAGSF = True             # Unscented Adaptive Gaussian Sum Filter
    RUN_BPF = False               # Bootstrap Particle Filter
    RUN_APF = False              # Auxiliary Particle Filter
    
    # =========================================================================
    # CONFIGURATION: Plotting options
    # =========================================================================
    PLOT_SLICES = True          # Plot posterior PDF slices at specific time steps
    PLOT_PAIRWISE_SCATTER = False # Plot pairwise state scatter for results

    PLOT_COMPARISON_SCATTER = True # Plot multi-panel comparison scatter for enabled filters
    PLOT_ONLY_COMPARISON_SCATTER = False  # Skip other plots for faster use
    COMPARISON_SCATTER_DIR = "figures"  # Where to save comparison scatter plots
    SHOW_COMPARISON_SCATTER = False  # Show comparison scatter window (False = save only)
    # =========================================================================

    if PLOT_ONLY_COMPARISON_SCATTER:

        plt.style.use(['science', 'ieee'])
        matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

    # =========================================================================
    # CONFIGURATION: Algorithm default hyperparameters (shared across examples)
    # =========================================================================
    
    # GSF defaults (LGSF/UGSF)
    GSF_DEFAULT = {
        "num_components": 1000,
        "num_iterations": 1,
    }
    UGSF_PARAMS = {
        "alpha": 1.0,
        "beta": 0.0,
        "kappa": 0.0,
    }
    
    # AGSF defaults (LAGSF/UAGSF)
    M, N, L = 200, 5, 5
    LAGSF_DEFAULT = {
        "M": M,
        "N": N,
        "L": L,
        "autocov1": 'prop',
        "autocov2": 'auto',
        "r1": 0.9,
        "r2": 0.9,
        "num_iterations": 1,
    }
    UAGSF_DEFAULT = {
        "M": M,
        "N": N,
        "L": L,
        "autocov1": 'prop',
        "autocov2": 'prop',
        "r1": 0.9,
        "r2": 0.9,
        "num_iterations": 1,
    }
    
    # Particle filter defaults (BPF/APF)
    PF_DEFAULT = {
        "num_particles": 100,
        "ess_threshold": 0.5,
    }

    # =========================================================================
    
    print("=" * 80)
    print("STANDALONE FILTER RUNNER")
    print("=" * 80)

    # =========================================================================
    # CONFIGURATION: Model setup (set these directly)
    # =========================================================================
    MODEL_NAME = "MTT"
    NUM_TIMESTEPS = 500
    SEED1 = random.randint(0, 10000)
    SEED2 = random.randint(0, 10000)
    SEED3 = random.randint(0, 10000)
    COMPARE_SEEDS = [SEED1, SEED2, SEED3]
    COMPARE_COUNTS = [100, 500, 1000]
    MTT_CONFIG = MTTConfig(
        Q_scale=1e-6,
        R_scale=25*1e-1,
        acc = 0.05
    )

    LMSV_CONFIG = LMSVConfig(
        state_dim=1,
        emission_dim=1)
    
    MODEL_CONFIG = MTT_CONFIG if MODEL_NAME == "MTT" else LMSV_CONFIG

    if MODEL_NAME == "MTT":

        COMPARE_DIMS = (0, 2)
        COMPARE_FILENAME = f"comparison_scatter_{SEED1}_{MODEL_CONFIG.acc}_{MODEL_CONFIG.R_scale}.pdf"

    COMPARISON_KEYS = [None] #["LGSF", "LAGSF", "BPF"] # Only for LMSV
    AUTOCOV_FILTERS = {"LAGSF", "UAGSF"}
    SLICE_TIMES = {
        "LGSF": [50, 250, 450],
        "UGSF": [50, 250, 450],
        "LAGSF": [50, 100],
        "UAGSF": [50, 100],
        "BPF": [50, 100],
        "APF": [50, 100],
    }

    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {SEED1}")

    runner = FilterRunner(
        MODEL_CONFIG,
        num_timesteps=NUM_TIMESTEPS,
        seed=SEED1,
    )

    results: Dict[str, Dict] = {}

    filter_specs = [
        ("LGSF", RUN_LGSF, runner.run_lgsf, GSF_DEFAULT),
        ("UGSF", RUN_UGSF, lambda **kw: runner.run_ugsf(**kw, **UGSF_PARAMS), GSF_DEFAULT),
        ("LAGSF", RUN_LAGSF, runner.run_lagsf, LAGSF_DEFAULT),
        ("UAGSF", RUN_UAGSF, lambda **kw: runner.run_uagsf(**kw, **UGSF_PARAMS), UAGSF_DEFAULT),
        ("BPF", RUN_BPF, runner.run_bpf, PF_DEFAULT),
        ("APF", RUN_APF, runner.run_apf, PF_DEFAULT),
    ]

    for key, enabled, run_fn, params in filter_specs:
        if not enabled:
            continue

        print(f"\n--- Running {key} ---")
        if key in ("BPF", "APF"):

            result, _ = run_fn(**params)

        else:

            result = run_fn(**params)

        results[key] = result

        runner.print_summary(result, key)

        if not PLOT_ONLY_COMPARISON_SCATTER:

            runner.plot_results(result)
            runner.plot_errors(result, key)

            if key in AUTOCOV_FILTERS:

                runner.plot_autocov_diagnostics(result, key)

            if PLOT_PAIRWISE_SCATTER:

                runner.plot_pairwise_scatter(result)

            if PLOT_SLICES:

                for t in SLICE_TIMES.get(key, []):

                    runner.plot_slice(result, t=t)

    if PLOT_COMPARISON_SCATTER:
        comparison_algorithms = [
            key for key, enabled, _, _ in filter_specs
            if enabled and key in results
        ]
        if comparison_algorithms and runner.states.shape[1] > max(COMPARE_DIMS):
            compare_rows = []
            for compare_seed, compare_count in zip(COMPARE_SEEDS, COMPARE_COUNTS):
                row_runner = FilterRunner(
                    MODEL_CONFIG,
                    num_timesteps=NUM_TIMESTEPS,
                    seed=compare_seed,
                )
                row_results = {}
                for algo_name in comparison_algorithms:
                    if algo_name == "LGSF":
                        row_results[algo_name] = row_runner.run_lgsf(
                            num_components=compare_count,
                            num_iterations=GSF_DEFAULT["num_iterations"],
                        )
                    elif algo_name == "UGSF":
                        row_results[algo_name] = row_runner.run_ugsf(
                            num_components=compare_count,
                            num_iterations=GSF_DEFAULT["num_iterations"],
                            alpha=UGSF_PARAMS["alpha"],
                            beta=UGSF_PARAMS["beta"],
                            kappa=UGSF_PARAMS["kappa"],
                        )
                    elif algo_name == "LAGSF":
                        row_results[algo_name] = row_runner.run_lagsf(
                            M=compare_count,
                            N=LAGSF_DEFAULT["N"],
                            L=LAGSF_DEFAULT["L"],
                            autocov1=LAGSF_DEFAULT["autocov1"],
                            r1=LAGSF_DEFAULT["r1"],
                            autocov2=LAGSF_DEFAULT["autocov2"],
                            r2=LAGSF_DEFAULT["r2"],
                            num_iterations=LAGSF_DEFAULT["num_iterations"],
                        )
                    elif algo_name == "UAGSF":
                        row_results[algo_name] = row_runner.run_uagsf(
                            M=compare_count,
                            N=UAGSF_DEFAULT["N"],
                            L=UAGSF_DEFAULT["L"],
                            alpha=UGSF_PARAMS["alpha"],
                            beta=UGSF_PARAMS["beta"],
                            kappa=UGSF_PARAMS["kappa"],
                            autocov1=UAGSF_DEFAULT["autocov1"],
                            r1=UAGSF_DEFAULT["r1"],
                            autocov2=UAGSF_DEFAULT["autocov2"],
                            r2=UAGSF_DEFAULT["r2"],
                            num_iterations=UAGSF_DEFAULT["num_iterations"],
                        )
                    elif algo_name == "BPF":
                        row_results[algo_name], _ = row_runner.run_bpf(
                            num_particles=compare_count,
                            ess_threshold=PF_DEFAULT["ess_threshold"],
                        )
                    elif algo_name == "APF":
                        row_results[algo_name], _ = row_runner.run_apf(
                            num_particles=compare_count,
                            ess_threshold=PF_DEFAULT["ess_threshold"],
                        )

                compare_rows.append(
                    {
                        "states": row_runner.states,
                        "results": row_results,
                        "seed": compare_seed,
                        "count": compare_count,
                    }
                )

            os.makedirs(COMPARISON_SCATTER_DIR, exist_ok=True)
            save_path = os.path.join(COMPARISON_SCATTER_DIR, COMPARE_FILENAME)
            runner.plot_three_panel_scatter(
                [row["results"] for row in compare_rows],
                title=None,
                dims=COMPARE_DIMS,
                save_path=save_path,
                rasterize_points=True,
                dpi=300,
                show=SHOW_COMPARISON_SCATTER,
                states_list=[row["states"] for row in compare_rows],
                row_titles=None,
                algorithm_order=comparison_algorithms,
            )

    if not PLOT_ONLY_COMPARISON_SCATTER:
        comparison_results = {
            key: results[key]
            for key in COMPARISON_KEYS
            if key in results
        }
        if comparison_results:
            runner.plot_error_comparison(comparison_results, "Error Comparison")
            runner.plot_mean_comparison(comparison_results, "Mean Comparison")

    plt.show()