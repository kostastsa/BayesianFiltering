# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License
import os
import numpy as np
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp

from flax import nnx

import util.plot
import util.io
import util.math

import experiment_descriptor as ed
from util.param import to_train_array
import misc



class ExperimentViewer:
    """
    Shows the results of a previously run experiment.
    """

    def __init__(self, exp_desc, overwrite):
        """
        :param exp_desc: an experiment descriptor object
        """

        assert isinstance(exp_desc, ed.ExperimentDescriptor)

        self.exp_desc = exp_desc
        self.overwrite = overwrite
        self.exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        self.sim = misc.get_simulator(exp_desc.sim)

    def print_log(self, trial=0):
        """
        Prints the log of the experiment.
        """

        print('\n' + '-' * 80)
        print('PRINTING LOG:\n')
        print(self.exp_desc.pprint())

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(self.exp_desc)

        assert util.io.load_txt(os.path.join(exp_dir, 'info.txt')) == self.exp_desc.pprint()

        print(util.io.load_txt(os.path.join(exp_dir, 'out.log')))

    def view_results(self, trial=0, block=False):
        """
        Shows the results of the experiment.
        :param block: whether to block execution after showing results
        """

        print('\n' + '-' * 80)
        print('VIEWING RESULTS:\n')
        print(self.exp_desc.pprint())

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(self.exp_desc)

        assert util.io.load_txt(os.path.join(exp_dir, 'info.txt')) == self.exp_desc.pprint()

        if isinstance(self.exp_desc.inf, ed.ABC_Descriptor):
            self._view_abc(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.BPF_MCMC_Descriptor):
            self._view_mcmc(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.SNL_Descriptor):
            self._view_snl(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.TSNL_Descriptor):
            self._view_snl(exp_dir)

        else:
            raise TypeError('unknown inference descriptor')

    def _view_abc(self, exp_dir):
        """
        View the results for ABC,
        """

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        props = sim_setup['props']
        [true_ps, true_cps], obs_ys = util.io.load(os.path.join(exp_dir, 'gt'))
        results = util.io.load(os.path.join(exp_dir, 'results'))

        if isinstance(inf_desc, ed.SMC_ABC_Descriptor):

            all_samples, all_weights, all_eps, all_log_ess, all_n_sims, avg_acc_dists, is_nan = results

            if is_nan:
                print('ABORT: NaNs encountered')
                return None

            # posterior histograms
            skip = int(max(1, len(all_eps) / 5))
            for samples, weights, eps in zip(all_samples[:-1:skip], all_weights[:-1:skip], all_eps[:-1:skip]):
                fig = util.plot.plot_hist_marginals(samples, weights, lims=self.sim.get_disp_lims(), gt=true_cps)
                fig.suptitle('SMC ABC, eps = {0:.2}'.format(eps))
            fig = util.plot.plot_hist_marginals(all_samples[-1], all_weights[-1], lims=self.sim.get_disp_lims(), gt=true_cps)
            fig.suptitle('SMC ABC, eps = {0:.2}'.format(all_eps[-1]))

            # effective sample size vs iteration
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.exp(all_log_ess) * 100, ':o')
            ax.plot(ax.get_xlim(), 0.5 * np.ones(2) * 100, 'r--')
            ax.set_xlabel('iteration')
            ax.set_ylabel('effective sample size [%]')
            ax.set_title('SMC ABC')

            # sims vs eps
            fig, ax = plt.subplots(1, 1)
            ax.plot(all_eps, all_n_sims, ':o')
            ax.set_xlabel('eps')
            ax.set_ylabel('num_sims')
            ax.set_title('SMC ABC')

            # acc_dist vs vs eps
            fig, ax = plt.subplots(1, 1)
            ax.plot(all_eps, ':o', label='eps')
            ax.plot(avg_acc_dists, ':o', label='dist')
            ax.legend()
            ax.set_xlabel('iteration')
            ax.set_ylabel('distance')
            ax.set_title('SMC ABC')

            fig_dir = os.path.join(exp_dir, 'figures')

            if not os.path.exists(fig_dir):

                util.io.make_folder(fig_dir)

                figs = [fig for fig in plt.get_fignums()]

                for i, fig in enumerate(figs):

                    plt.figure(fig)
                    plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

                plt.close('all')  # Closes all open figures

            else:

                if self.overwrite:

                    for file in os.listdir(fig_dir):

                        file_path = os.path.join(fig_dir, file)

                        if os.path.isfile(file_path):

                            os.unlink(file_path)

                        elif os.path.isdir(file_path):

                            os.rmdir(file_path)
            
                    figs = [fig for fig in plt.get_fignums()]

                    for i, fig in enumerate(figs):

                        plt.figure(fig)
                        plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

                    plt.close('all')  # Closes all open figures

                else: 

                    print('Figures already exist, use --overwrite to replace them')

        else:

            raise TypeError('unknown ABC descriptor')
        
        error, num_simulations = util.io.load(os.path.join(exp_dir, 'error'))
        return error, num_simulations
        
    def _view_mcmc(self, exp_dir):
        """
        View the results for ABC,
        """

        import jax.scipy.stats as jss

        inf_desc = self.exp_desc.inf
        sim_desc = self.exp_desc.sim
        sim_setup = self.sim.setup(sim_desc.state_dim, sim_desc.emission_dim, sim_desc.input_dim, sim_desc.target_vars)
        props = sim_setup['props']
        [true_ps, true_cps], _ = util.io.load(os.path.join(exp_dir, 'gt'))
        results = util.io.load(os.path.join(exp_dir, 'results'))

        if isinstance(inf_desc, ed.PRT_MCMC_Descriptor):

            positions, lls = results.position, results.logdensity
            
            fig = util.plot.plot_hist_marginals(positions, lims=self.sim.get_disp_lims(), gt=true_cps)
            fig.suptitle('BPF MCMC')
            fig, ax = plt.subplots(1, 1)
            ax.plot(lls)
            ax.set_title('Logdensity evolution')

            fig_dir = os.path.join(exp_dir, 'figures')

            if not os.path.exists(fig_dir):

                util.io.make_folder(fig_dir)

                figs = [fig for fig in plt.get_fignums()]

                for i, fig in enumerate(figs):

                    plt.figure(fig)
                    plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

                plt.close('all')  # Closes all open figures

            else:

                if self.overwrite:

                    for file in os.listdir(fig_dir):

                        file_path = os.path.join(fig_dir, file)

                        if os.path.isfile(file_path):

                            os.unlink(file_path)

                        elif os.path.isdir(file_path):

                            os.rmdir(file_path)
            
                    figs = [fig for fig in plt.get_fignums()]

                    for i, fig in enumerate(figs):

                        plt.figure(fig)
                        plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

                    plt.close('all')  # Closes all open figures

                else: 

                    print('Figures already exist, use --overwrite to overwrite them')

        else:
            raise TypeError('unknown ABC descriptor')
        
        error, num_simulations = util.io.load(os.path.join(exp_dir, 'error'))
        return error, num_simulations
      
    def _view_snl(self, exp_dir):
        """
        Shows the results of learning the likelihood with MCMC.
        """

        model_id = self.exp_desc.inf.model.get_id(' ')
        train_on = self.exp_desc.inf.train_on
        n_rounds = self.exp_desc.inf.n_rounds
        n_samples = self.exp_desc.inf.n_samples

        [_, true_cond_ps], obs_ys = util.io.load(os.path.join(exp_dir, 'gt'))
        all_cond_params = util.io.load(os.path.join(exp_dir, 'all_cond_params'))
        all_emissions = util.io.load(os.path.join(exp_dir, 'all_emissions'))
        posterior = util.io.load(os.path.join(exp_dir, 'posterior'))
        all_dist = util.io.load(os.path.join(exp_dir, 'all_dists'))
        avg_dist = np.mean(all_dist, axis=1)

        # show distances
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(all_dist)
        ax.set_xlabel('round')
        ax.set_title('SNL on {0}, {1}, distances'.format(train_on, model_id))

        fig, ax = plt.subplots(1, 1)
        ax.plot(avg_dist)
        ax.set_xlabel('round')
        ax.set_title('SNL on {0}, {1}, avg distance'.format(train_on, model_id))

        # show proposed parameters
        for i, ps in enumerate(all_cond_params.reshape((n_rounds, n_samples, -1))):
            fig = util.plot.plot_hist_marginals(ps, lims=self.sim.get_disp_lims(), gt=true_cond_ps)
            fig.suptitle('SNL on {0}, {1}, proposed params round {2}'.format(train_on, model_id, i+1))

        _, post_cps = posterior
        fig = util.plot.plot_hist_marginals(post_cps, lims=self.sim.get_disp_lims(), gt=true_cond_ps)
        fig.suptitle(f'SNL posterior, {post_cps.shape[0]} samples')

        fig_dir = os.path.join(exp_dir, 'figures')

        if not os.path.exists(fig_dir):

            util.io.make_folder(fig_dir)

            figs = [fig for fig in plt.get_fignums()]

            for i, fig in enumerate(figs):

                plt.figure(fig)
                plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

            plt.close('all')  # Closes all open figures

        else:

            if self.overwrite:

                for file in os.listdir(fig_dir):

                    file_path = os.path.join(fig_dir, file)

                    if os.path.isfile(file_path):

                        os.unlink(file_path)

                    elif os.path.isdir(file_path):

                        os.rmdir(file_path)
        
                figs = [fig for fig in plt.get_fignums()]

                for i, fig in enumerate(figs):

                    plt.figure(fig)
                    plt.savefig(os.path.join(fig_dir, 'fig_{0}.png'.format(i)))

                plt.close('all')  # Closes all open figures

            else: 

                print('Figures already exist, use --overwrite to overwrite them')

        error, num_simulations = util.io.load(os.path.join(exp_dir, 'error'))
        return error, num_simulations

    def _create_model(self, n_inputs, n_cond, rng):
        """
        Given input and output sizes, creates and returns the model for the NDE experiments.
        """

        model_desc = self.exp_desc.inf.model

        if isinstance(model_desc, ed.MAF_Descriptor):

            return MAF(
                din = n_inputs,
                nmade = model_desc.nmades,
                dhidden=model_desc.dhidden,
                nhidden=model_desc.nhidden,
                act_fun=model_desc.act_fun,
                dcond = n_cond,
                rngs = nnx.Rngs(rng),
                random_order = model_desc.random_order,
                reverse=model_desc.reverse,
                dropout=model_desc.dropout,
                batch_norm=model_desc.batch_norm
            )

        else:
            raise TypeError('unknown model descriptor')


def get_dist_disp_lim(sim_desc):
    """
    Given a simulator descriptor, returns the upper display limit for the distances histogram.
    """

    dist_disp_lim = {
        'gauss': float('inf'),
        'mg1': 1.0,
        'lotka_volterra': float('inf'),
        'hodgkin_huxley': float('inf')
    }

    return dist_disp_lim[sim_desc]
