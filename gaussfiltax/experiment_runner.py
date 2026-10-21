# This module is adapted from the repository ([https://github.com/gpapamak/snl.git]),
# authored by George Papamakarios under the MIT License
import jax # type: ignore
from jax import random as jr # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as onp # type: ignore
import os
import gc
import time
import util.io
import experiment_descriptor as ed
import misc
import matplotlib.pyplot as plt # type: ignore
import matplotlib_inline # type: ignore
from util.param import sample_prior, to_train_array
from functools import partial

plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def _plot_states_observations(states, observations, exp_dir, prefix='observations'):
    """Plot state and observation summaries for arbitrary dimensions."""

    st = onp.asarray(states)
    obs = onp.asarray(observations)

    if st.ndim == 1:

        st = st[:, None]

    if obs.ndim == 1:

        obs = obs[:, None]

    state_dim = st.shape[1]
    obs_dim = obs.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # Phase plot (first two state dims if available, else dim 0 line)

    if state_dim >= 2:

        axes[0].plot(st[:, 0], st[:, 1], linewidth=0.6, alpha=0.8)
        axes[0].scatter(st[0, 0], st[0, 1], c='red', s=15, label='start')
        axes[0].scatter(st[-1, 0], st[-1, 1], c='black', s=15, label='end')
        axes[0].set_xlabel('state dim 0')
        axes[0].set_ylabel('state dim 1')

    else:

        axes[0].plot(st[:, 0], linewidth=0.8, alpha=0.9)
        axes[0].set_xlabel('time')
        axes[0].set_ylabel('state dim 0')

    axes[0].set_title('State trajectory')
    axes[0].grid(True, alpha=0.3)

    # State time series
    for d in range(state_dim):

        axes[1].plot(st[:, d], label=f'x{d}', linewidth=0.8)

    axes[1].set_xlabel('time')
    axes[1].set_ylabel('state value')

    if state_dim <= 8:

        axes[1].legend()

    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('States over time')

    # Observation time series
    for d in range(obs_dim):

        axes[2].plot(obs[:, d], label=f'y{d}', linewidth=0.8)

    axes[2].set_xlabel('time')
    axes[2].set_ylabel('observation value')

    if obs_dim <= 8:

        axes[2].legend()

    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Observations over time')

    out_path = os.path.join(exp_dir, f'{prefix}.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


class ExperimentRunner:
    """
    Runs experiments on likelihood-free inference of simulator models.
    """

    def __init__(self, exp_desc):
        """
        :param exp_desc: an experiment descriptor object
        """

        assert isinstance(exp_desc, ed.ExperimentDescriptor)

        self.exp_desc = exp_desc
        self.exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        self.sim = misc.get_simulator(exp_desc.sim)

    def run(self, trial=0, sample_emissions = True, plot_sims=False, n_post_samples=1000, key=jr.PRNGKey(0), seed=0):
        """
        Runs the experiment.
        :param rng: random number generator to use
        """

        print('\n' + '-' * 80)
        print('RUNNING EXPERIMENT, TRIAL {0}:\n'.format(trial))
        print(self.exp_desc.pprint())

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if os.path.exists(exp_dir):
            raise misc.AlreadyExistingExperiment(self.exp_desc)

        util.io.make_folder(exp_dir)

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        param_info = self.sim.get_param_info()

        if sim_desc.name == 'covinfl':
      
            a = sim_desc.a
            b = sim_desc.b
            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, a, b, sim_desc.input_dim)

        elif sim_desc.name == 'mtt':

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.a, sim_desc.sigma2)

        else:

            sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim)

        ssm = sim_setup['ssm']
        props = sim_setup['props']

        try:

            if isinstance(self.exp_desc.inf, ed.LGSF_Descriptor):

                self._run_lgsf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.UGSF_Descriptor):

                self._run_ugsf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.LAGSF_Descriptor):

                self._run_lagsf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.UAGSF_Descriptor):

                self._run_uagsf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.BPF_Descriptor):

                self._run_bpf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)

            elif isinstance(self.exp_desc.inf, ed.APF_Descriptor):

                self._run_apf(ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed)
            else:
                raise TypeError('unknown inference descriptor')

        except:

            print('EXPERIMENT FAILED')

            raise

    def _run_lgsf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):
        """
        Runs the ABC experiments.
        """

        from filters import gsf

        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0] 
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = gsf.LGSF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results = filter.run(
                            true_ps, 
                            observations,
                            inf_desc.n_comps,
                            inf_desc.n_iters,
                            subkey,
                            inputs)
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save((states, observations, true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))


            del results
            runner = None
            jax.clear_caches()
            gc.collect()

    def _run_ugsf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):

        """
        Runs the ABC experiments.
        """

        from filters import gsf

        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = gsf.UGSF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results = filter.run(
                            key,
                            true_ps, 
                            observations,
                            inf_desc.n_comps,
                            inf_desc.n_iters,
                            inf_desc.alpha,
                            inf_desc.beta,
                            inf_desc.kappa,
                            inputs)
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save((states, observations,true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))


            del results
            runner = None
            jax.clear_caches()
            gc.collect()

    def _run_lagsf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):
        """
        Runs the ABC experiments.
        """

        from filters import agsf

        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = agsf.LAGSF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results, aux_out = filter.run(
                                subkey,
                                true_ps,
                                observations,
                                inf_desc.ac1,
                                inf_desc.r1,
                                inf_desc.ac2,
                                inf_desc.r2,
                                inf_desc.M,
                                inf_desc.N,
                                inf_desc.L,
                                inf_desc.n_iters,
                                inputs
            )
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save(aux_out, os.path.join(exp_dir, 'aux_outputs'))
            util.io.save((states, observations,true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))

            del results
            runner = None
            jax.clear_caches()
            gc.collect()

    def _run_uagsf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):
        """
        Runs the ABC experiments.
        """

        from filters import agsf

        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = agsf.UAGSF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results, aux_out = filter.run(
                                subkey,
                                true_ps,
                                observations,
                                inf_desc.ac1,
                                inf_desc.r1,
                                inf_desc.ac2,
                                inf_desc.r2,
                                inf_desc.M,
                                inf_desc.N,
                                inf_desc.L,
                                inf_desc.n_iters,
                                inf_desc.alpha,
                                inf_desc.beta,
                                inf_desc.kappa,
                                inputs
            )
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save(aux_out, os.path.join(exp_dir, 'aux_outputs'))
            util.io.save((states, observations, true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))

            del results
            runner = None
            jax.clear_caches()
            gc.collect()

    def _run_bpf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):

        """
        Runs the ABC experiments.
        """

        from filters import pf
    
        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)
            
        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = pf.BPF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results = filter.run(
                            true_ps,
                            observations,
                            inf_desc.num_prt,
                            subkey,
                            inputs,
                            inf_desc.ess
                            )
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save((states, observations, true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))


            del results
            runner = None
            jax.clear_caches()
            gc.collect()

    def _run_apf(self, ssm, sim_desc, inf_desc, props, exp_dir, plot_sims, key, seed):

        from filters import pf

        key, subkey = jr.split(key)
        inputs = self.sim.get_inputs(subkey, sim_desc.num_timesteps)
        true_ps = sample_prior(jr.PRNGKey(0), props, 1)[0]
        key, subkey = jr.split(key)
        states, observations = ssm.simulate(subkey, true_ps, sim_desc.num_timesteps, inputs)

        if plot_sims:
            _plot_states_observations(states, observations, exp_dir)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            filter = pf.APF(props, ssm)
            key, subkey = jr.split(key)
            
            start_time = time.time()
            results = filter.run(
                            true_ps,
                            observations,
                            inf_desc.num_prt,
                            subkey,
                            inputs,
                            inf_desc.ess
                            )
            elapsed_time = time.time() - start_time
            
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(f"Runtime: {elapsed_time:.6f} seconds\n", os.path.join(exp_dir, 'runtime.txt'))
            util.io.save((states, observations,true_ps), os.path.join(exp_dir, 'gt'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'exp_desc.txt'))


            del results
            runner = None
            jax.clear_caches()
            gc.collect()