# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License
import argparse
import os
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore

from flax import nnx  # type: ignore

import gc
import zlib
import util.io
from util.plot import get_acf_plot
from functools import partial

import os
import time
import subprocess

from experiment_runner import ExperimentRunner
import experiment_descriptor as ed
import experiment_results as er
import jax.random as jr  # type: ignore
import misc

from util.numerics import log_prob_error, min_error, bias, get_num_sims, nees, mse_x1, mse_rest
# from util.sample import sim_emissions, generate_emissions
# from util.train import find_mle, loglik_fn
# from inference.diagnostics.two_sample import sq_maximum_mean_discrepancy as mmd
import matplotlib.pyplot as plt  # type: ignore
import matplotlib_inline # type: ignore
import scienceplots # type: ignore

plt.style.use(['science', 'ieee'])
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

fig_dir = '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/figures'

# Seed of the bootstrap in 'errors' (see eval_and_plot_errors).
BOOTSTRAP_SEED = 0

def parse_args():
    """
    Returns an object describing the command line.
    """

    def int_or_str(value):
        """Try converting to int, otherwise return as string."""

        try:

            return int(value)

        except ValueError:

            return value

    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments.')
    subparsers = parser.add_subparsers()

    parser_trials = subparsers.add_parser('trials', help='run multiple experiment trials')
    parser_trials.add_argument('seed', type=int_or_str, help='seed for PRNG')
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_trials)

    parser_view = subparsers.add_parser('view', help='view results')
    parser_view.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_view.add_argument('-b', '--block', action='store_true', help='block execution after viewing each experiment')
    parser_view.add_argument('-o', '--overwrite', type=str)
    parser_view.add_argument('-t', '--trial', type=int, default=0, help='trial to view (default is 0)')
    parser_view.set_defaults(func=view_results)

    parser_ensemble = subparsers.add_parser('ensemble', help='view ensemble results')
    parser_ensemble.add_argument('start', type=int, help='# of first trial')
    parser_ensemble.add_argument('end', type=int, help='# of last trial')
    parser_ensemble.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_ensemble.set_defaults(func=view_ensemble)

    parser_errors = subparsers.add_parser('errors', help='plot error by variable and inference method')
    parser_errors.add_argument('start', type=int, help='# of first trial')
    parser_errors.add_argument('end', type=int, help='# of last trial')
    parser_errors.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_errors.add_argument('--no-plots', action='store_true', help='compute the metrics and the summary, but do not make the histograms')
    parser_errors.set_defaults(func=eval_and_plot_errors)

    parser_table = subparsers.add_parser('table', help='Markdown table of the Error summary.txt files (run errors first)')
    parser_table.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_table.add_argument('-o', '--out', type=str, default=None, help='write the table to this file (default: print it)')
    parser_table.set_defaults(func=make_table)

    parser_time = subparsers.add_parser('time', help='plot the MSE over time and the rho2 of the AGSF, mean over trials (run errors first)')
    parser_time.add_argument('start', type=int, help='# of first trial')
    parser_time.add_argument('end', type=int, help='# of last trial')
    parser_time.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_time.add_argument('-o', '--out', type=str, default=None, help='output file (default: figures/time_<sim>.pdf)')
    parser_time.set_defaults(func=plot_time)

    parser_cost = subparsers.add_parser('cost', help='MSE and LPE of the table rows against the run time of the timing runs (task A3)')
    parser_cost.add_argument('src', type=str, help="data source of the timing runs (the SRC of 'create_exps.py timing')")
    parser_cost.add_argument('files', nargs='+', type=str, help='descriptor files of the table rows (one simulator setting)')
    parser_cost.add_argument('-o', '--out', type=str, default=None, help='output figure (default: figures/cost_<sim>.pdf)')
    parser_cost.add_argument('--md', type=str, default=None, help='also write a Markdown table of the run times to this file')
    parser_cost.set_defaults(func=plot_cost)

    parser_dim = subparsers.add_parser('dim', help='sinlin: eigenvalues of the AGSF split and split errors against the state dimension (task A10)')
    parser_dim.add_argument('start', type=int, help='# of first trial')
    parser_dim.add_argument('end', type=int, help='# of last trial')
    parser_dim.add_argument('files', nargs='+', type=str, help='file(s) describing experiments (several state dimensions)')
    parser_dim.add_argument('-o', '--out', type=str, default=None, help='output figure (default: figures/dim_<sim>.pdf)')
    parser_dim.add_argument('--md', type=str, default=None, help='also write a Markdown table to this file')
    parser_dim.set_defaults(func=plot_dim)

    return parser.parse_args()


def run_trials(args):
    """
    Runs experiments for multiple trials with random ground truth.
    """

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')
    
    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    seed = int(time.time() * 1000) if args.seed=='r' else args.seed  
    key = jr.PRNGKey(seed)
    
    for exp_desc in exp_descs:

        for trial in range(args.start, args.end + 1):
            
            exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
            trial_dir = os.path.join(exp_dir, str(trial))
            # one split for every trial, also for a trial that is skipped: a trial gets the same key after a restart
            key, subkey = jr.split(key)

            if os.path.exists(trial_dir + '/results.pkl'):

                print('EXPERIMENT ALREADY EXISTS')
                jax.clear_caches()
                gc.collect()
                
                continue

            runner = ExperimentRunner(exp_desc)

            try:

                if os.path.exists(trial_dir):
                    
                    for root, _, files in os.walk(trial_dir, topdown=False):

                        for name in files:

                            os.remove(os.path.join(root, name))

                    os.rmdir(trial_dir)

                out = runner.run(trial=trial, plot_sims=True, key=subkey, seed=seed)

            except misc.AlreadyExistingExperiment:

                print('RUNNER FAILED')

            out = None
            jax.clear_caches()
            gc.collect()

    print('ALL DONE')


def view_results(args):
    """
    Views experiments.
    """

    from experiment_viewer import ExperimentViewer, plt

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    if args.overwrite=='false':

        overwrite = False

    else:

        overwrite = True

    for exp_desc in exp_descs:

        try:

            ExperimentViewer(exp_desc, overwrite).view_results(trial=args.trial, block=args.block)

        except misc.NonExistentExperiment:

            print('EXPERIMENT DOES NOT EXIST')


def view_ensemble(args, show=False):
    """
    Takes a set of experiments and line-plots the errors vs num_sims of each algorithm together.
    """

    from experiment_viewer import plt

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    lgsf = er.LGSF_Results()
    ugsf = er.UGSF_Results()
    lagsf = er.LAGSF_Results()
    uagsf = er.UAGSF_Results()
    bpf = er.BPF_Results()
    apf = er.APF_Results()
    rpf = er.RPF_Results()
    upf = er.UPF_Results()


    for exp_desc in exp_descs:
        
        lpe_array = []
        bs_array = []
        me_array = []

        for trial in range(args.start, args.end + 1):

            try:

                print(trial)
                exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir += '/' + str(trial)

                try:

                    print(exp_desc.pprint())

                    lpe, lpe_t = util.io.load(os.path.join(exp_dir, 'lpe'))
                    lpe_array.append(lpe)

                    bs, bs_t = util.io.load(os.path.join(exp_dir, 'bs'))
                    bs_array.append(bs)

                    me, me_t = util.io.load(os.path.join(exp_dir, 'me'))
                    me_array.append(me)

                    num_sims = util.io.load(os.path.join(exp_dir, 'num_sims'))

                except FileNotFoundError:

                    print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:

                print('TRIAL DOES NOT EXIST')

        B = 100
        # The same fixed bootstrap key for each experiment as in 'errors' (BOOTSTRAP_SEED and a hash of the folder)
        key = jr.fold_in(jr.PRNGKey(BOOTSTRAP_SEED), zlib.crc32(exp_desc.get_dir().encode()) & 0x7fffffff)
        key, subkey = jr.split(key)
        lpe_array = jnp.array(lpe_array)
        lpe_array = lpe_array[~(jnp.isnan(lpe_array) + jnp.isinf(lpe_array))]
        bootstrap_lpe = util.numerics.bootstrap(subkey, lpe_array, B)
        mean_lpe = jnp.mean(bootstrap_lpe)
        std_lpe = jnp.std(bootstrap_lpe)

        key, subkey = jr.split(key)
        bs_array = jnp.array(bs_array)
        bs_array = bs_array[~(jnp.isnan(bs_array) + jnp.isinf(bs_array))]
        bootstrap_bs = util.numerics.bootstrap(subkey, bs_array, B)
        mean_bs = jnp.mean(bootstrap_bs)
        std_bs = jnp.std(bootstrap_bs)

        key, subkey = jr.split(key)
        me_array = jnp.array(me_array)
        me_array = me_array[~(jnp.isnan(me_array) + jnp.isinf(me_array))]
        bootstrap_me = util.numerics.bootstrap(subkey, me_array, B)
        mean_me = jnp.mean(bootstrap_me)
        std_me = jnp.std(bootstrap_me)

        num_sims = jnp.log(num_sims)
            
        if isinstance(exp_desc.inf, ed.LGSF_Descriptor):
            
            lgsf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])
        
        elif isinstance(exp_desc.inf, ed.UGSF_Descriptor):

            ugsf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.LAGSF_Descriptor):

            lagsf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.UAGSF_Descriptor):

            uagsf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.BPF_Descriptor):

            bpf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.APF_Descriptor):

            apf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.RPF_Descriptor):

            rpf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

        elif isinstance(exp_desc.inf, ed.UPF_Descriptor):

            upf.results.append([num_sims, mean_lpe, std_lpe, mean_bs, std_bs, mean_me, std_me])

    lgsf.make_jnp()
    ugsf.make_jnp()
    lagsf.make_jnp()
    uagsf.make_jnp()
    bpf.make_jnp()
    apf.make_jnp()
    rpf.make_jnp()
    upf.make_jnp()

    fig, ax = plt.subplots(3, 1, figsize=(5, 5))

    for alg in [lgsf, ugsf, lagsf, uagsf, bpf, apf, rpf, upf]:

        try:

            ax[0].plot(alg.results[:, 0], alg.results[:, 1], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[0].fill_between(alg.results[:, 0], alg.results[:,1]-alg.results[:, 2], alg.results[:, 1] + alg.results[:, 2], color=alg.color, alpha=0.05)
            ax[0].set_ylabel('$\mathcal{E}_{LPE}$')
            ax[0].legend(prop={'size': 5})

            ax[1].plot(alg.results[:, 0], alg.results[:, 3], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[1].fill_between(alg.results[:, 0], alg.results[:,3]-alg.results[:, 4], alg.results[:, 3] + alg.results[:, 4], color=alg.color, alpha=0.05)
            ax[1].set_ylabel('Bias')
            ax[1].legend(prop={'size': 5})

            ax[2].plot(alg.results[:, 0], alg.results[:, 5], marker=alg.marker, linestyle='dotted', color=alg.color, markersize=3, label=f'{alg.name}')
            ax[2].fill_between(alg.results[:, 0], alg.results[:,5]-alg.results[:, 6], alg.results[:, 5] + alg.results[:, 6], color=alg.color, alpha=0.05)
            ax[2].set_ylabel('$\mathcal{E}_{min}$')
            ax[2].legend(prop={'size': 5})
            ax[2].set_xlabel('log Number of simulations')

        except IndexError as e:
            
            print(f'Error plotting {alg.name}: {e}')
            print(f'No {alg.name} results to plot')

        except TypeError as e:
            
            print(f'Error plotting {alg.name}: {e}')
            print(f'{alg.name} has no results')

    fig.savefig(os.path.join(fig_dir, f'ensemble_{exp_desc.sim.name}_' + '.pdf'), format="pdf", dpi=800)

    if show:

        plt.show()


def read_error_summary(path):
    """Values of an 'Error summary.txt' file of 'main.py errors': the number of trials, the number of trials with MSE
    above 10, and (bootstrap mean, bootstrap sd, median) of each metric."""

    import re
    out = {}
    for line in util.io.load_txt(path).splitlines():
        if line.startswith('trials:'):
            out['trials'] = int(line.split(':')[1])
        elif line.startswith('trials with mse > 10:'):
            out['lost'] = int(line.split(':')[1])
        elif line.startswith('trials with > 20 % of steps outside:'):
            out['empty'] = line.split(':')[1].strip()
        else:
            m = re.match(r'(lpe|bias|minerr|mse_x1|mse_rest|mse|outside): (\S+) \+- (\S+), (\S+)', line)
            if m:
                out[m.group(1)] = tuple(float(v) for v in m.groups()[1:])
    return out


# Results class of each filter (name, marker and color of the plots).
RESULTS_CLASSES = {'lgsf': er.LGSF_Results, 'ugsf': er.UGSF_Results, 'lagsf': er.LAGSF_Results, 'uagsf': er.UAGSF_Results,
                   'bpf': er.BPF_Results, 'apf': er.APF_Results, 'rpf': er.RPF_Results, 'upf': er.UPF_Results}
# Simulators with a fixed input sequence (get_inputs does not use the key): 'time' shades the steps with input 1.
FIXED_INPUT_SIMS = ('lmsv',)


def _setting_label(inf_desc):
    """Short label of the setting of a filter, e.g. 'M = 100' or 'M = 100, N = L = 5'."""
    name = inf_desc.name
    if name in ('lgsf', 'ugsf'):
        return f'$M={inf_desc.n_comps}$'
    if name in ('lagsf', 'uagsf'):
        nl = f'N=L={inf_desc.N}' if inf_desc.N == inf_desc.L else f'N={inf_desc.N},L={inf_desc.L}'
        return f'$M={inf_desc.M},{nl}$'
    return f'$M={inf_desc.num_prt}$'


def plot_time(args):
    """
    Plots the mean over trials of the MSE over time (mse.pkl of 'errors', util.numerics.mse) of each experiment.
    If an experiment is an AGSF with aux_outputs.pkl, a second panel shows its rho2 over time: the mean over the
    components, with the mean and the 10 % and 90 % quantiles over the trials that have the file.
    Trials with a non-finite MSE are left out of the mean, and their number is printed.
    """
    import numpy as onp

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    mse_curves, rho_curves = [], []
    for exp_desc in exp_descs:
        folder = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        res = RESULTS_CLASSES[exp_desc.inf.name]()
        label = f'{res.name} ({_setting_label(exp_desc.inf)})'
        mse_t, rho_t = [], []
        for trial in range(args.start, args.end + 1):
            trial_dir = os.path.join(folder, str(trial))
            mse_t.append(onp.asarray(util.io.load(os.path.join(trial_dir, 'mse'))[1]))
            aux = os.path.join(trial_dir, 'aux_outputs')
            if exp_desc.inf.name in ('lagsf', 'uagsf') and os.path.exists(aux + '.pkl'):
                rho2s = onp.asarray(util.io.load(aux)['rho2s'])
                rho_t.append(rho2s.reshape(rho2s.shape[0], -1).mean(axis=1))
        mse_t = onp.array(mse_t)
        ok = onp.all(onp.isfinite(mse_t), axis=1)
        print(f'{label}: {ok.sum()} of {len(ok)} trials finite, time mean of the MSE {mse_t[ok].mean():.3f}')
        mse_curves.append((res, label, mse_t[ok].mean(axis=0)))
        if rho_t:
            rho_t = onp.array(rho_t)
            print(f'{label}: rho2 from {len(rho_t)} trials, min {rho_t.mean(axis=0).min():.3f}, max {rho_t.mean(axis=0).max():.3f}')
            rho_curves.append((res, label, rho_t))

    T = mse_curves[0][2].shape[0]
    sim = exp_descs[0].sim
    shade = None
    if sim.name in FIXED_INPUT_SIMS:
        shade = onp.asarray(misc.get_simulator(sim).get_inputs(jr.PRNGKey(0), T)) > 0.5

    n_rows = 2 if rho_curves else 1
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(3.5, 1.6 * n_rows + 0.7), squeeze=False)
    axes = axes[:, 0]
    ts = onp.arange(1, T + 1)
    every = max(T // 10, 1)
    for ax in axes:
        if shade is not None:
            for t in onp.flatnonzero(shade & ~onp.concatenate([[False], shade[:-1]])):
                end = t + onp.argmin(onp.concatenate([shade[t:], [False]]))
                ax.axvspan(t + 0.5, end + 0.5, color='0.9', lw=0, zorder=0)
    for res, label, curve in mse_curves:
        axes[0].plot(ts, curve, color=res.color, marker=res.marker, markevery=every, ms=2.5, lw=0.8, label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('MSE')
    # Legend above the top panel, so that it does not cover the curves
    axes[0].legend(fontsize=5, ncol=2, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', frameon=False)
    if rho_curves:
        styles = ['-', '--', ':', '-.']
        for k, (res, label, rho_t) in enumerate(rho_curves):
            axes[1].fill_between(ts, onp.quantile(rho_t, 0.1, axis=0), onp.quantile(rho_t, 0.9, axis=0), color=res.color, alpha=0.2, lw=0)
            axes[1].plot(ts, rho_t.mean(axis=0), color=res.color, ls=styles[k % 4], marker=res.marker, markevery=every, ms=2.5, lw=0.8, label=label)
        axes[1].set_ylabel(r'$\rho_2$')
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=5, loc='lower left', frameon=True)
    axes[-1].set_xlabel('time step $t$')
    axes[-1].set_xlim(1, T)
    fig.tight_layout()
    out = args.out or os.path.join(fig_dir, f'time_{sim.name}.pdf')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print('saved', out)


def _read_runtimes(folder):
    """Run times of the trials in folder (runtime.txt): arrays of the timed call and of the first call (NaN if absent)."""
    import re
    import numpy as onp
    run, first = [], []
    for t in sorted(int(d) for d in os.listdir(folder) if d.isdigit()) if os.path.isdir(folder) else []:
        path = os.path.join(folder, str(t), 'runtime.txt')
        if not os.path.exists(path):
            continue
        nums = [float(x) for x in re.findall(r'([\d.]+) seconds', util.io.load_txt(path))]
        run.append(nums[0])
        first.append(nums[1] if len(nums) > 1 else float('nan'))
    return onp.array(run), onp.array(first)


def plot_cost(args):
    """
    Task A3. For each table row in the files: the MSE and the LPE (bootstrap mean and sd of 'Error summary.txt') against
    the mean run time of its timing run. The timing run is the same filter on the stored data of the row 'src'
    (create_exps.py timing SRC, run_timing.zsh), and its runtime.txt has the time without the JIT compile (GAUSSFILTAX_WARM_TIME=1).
    Also writes a Markdown table with the run time, the time of the first call and the number of simulations (get_num_sims).
    """
    import numpy as onp

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    sims = {e.sim.get_dir() for e in exp_descs}
    assert len(sims) == 1, f'cost: give the files of one simulator setting, not {sims}'
    rows, curves = [], {}
    for e in exp_descs:
        row_dir = os.path.join(misc.get_root(), 'experiments', e.get_dir())
        timing_dir = os.path.join(misc.get_root(), 'experiments', e.sim.get_dir(),
                                  'data_' + args.src.replace('/', '_'), e.inf.get_dir())
        run, first = _read_runtimes(timing_dir)
        summ = os.path.join(row_dir, 'Error summary.txt')
        if run.size == 0 or not os.path.exists(summ):
            print(f'cost: no timing runs or no summary for {e.get_dir()}, left out')
            continue
        v = read_error_summary(summ)
        res = RESULTS_CLASSES[e.inf.name]()
        n_sims = int(util.numerics.get_num_sims(e))
        rows.append((res.name, _setting_label(e.inf), run.mean(), run.std(), onp.nanmean(first), run.size, n_sims, v['mse'], v['lpe'], v['trials']))
        curves.setdefault(e.inf.name, (res, []))[1].append((run.mean(), v['mse'][0], v['mse'][1], v['lpe'][0], v['lpe'][1]))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.6))
    for name, (res, pts) in curves.items():
        pts = onp.array(sorted(pts))
        for ax, k in ((axes[0], 1), (axes[1], 3)):
            ax.errorbar(pts[:, 0], pts[:, k], yerr=pts[:, k + 1], color=res.color, marker=res.marker, ms=3, lw=0.8,
                        ls='dotted', capsize=1.5, label=res.name)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('MSE')
    axes[1].set_yscale('symlog', linthresh=10)
    lo = min(min(p[3] - p[4] for p in pts) for _, pts in curves.values())
    axes[1].set_ylim(bottom=min(lo, 0) - 2)   # symlog: no empty negative decades below the data
    axes[1].set_ylabel('LPE')
    axes[1].set_xlabel('run time without compile (s)')
    axes[0].legend(fontsize=5, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', frameon=False)
    fig.tight_layout()
    sim_name = exp_descs[0].sim.name
    out = args.out or os.path.join(fig_dir, f'cost_{sim_name}.pdf')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print('saved', out)

    fmt = lambda x: f'{x:.2e}' if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2) else f'{x:.2f}'
    import platform
    machine = f'{platform.platform()}, {platform.processor() or platform.machine()}, JAX {jax.__version__} on {jax.devices()[0].platform}'
    text = (f'### {sims.pop()}\n\nRun time: mean and sd over the timing runs (without the JIT compile). First call: with the '
            f'compile. MSE and LPE: bootstrap mean and sd of the table row. Timing data: {args.src}. Machine: {machine}.\n\n'
            '| Filter | Setting | Run time (s) | First call (s) | Timing runs | Simulations | MSE | LPE | Trials |\n'
            '|---|---|---|---|---|---|---|---|---|\n')
    for name, lab, rm, rs, fc, nr, ns, mse_v, lpe_v, nt in rows:
        text += (f'| {name} | {lab} | {rm:.4f} ± {rs:.4f} | {fc:.3f} | {nr} | {ns:.2e} | {fmt(mse_v[0])} ± {fmt(mse_v[1])} '
                 f'| {fmt(lpe_v[0])} ± {fmt(lpe_v[1])} | {nt} |\n')
    if args.md:
        util.io.save_txt(text, args.md)
    print(text)


def _save_eig_summary(exp_dir, inf_name):
    """AGSF with aux_outputs.pkl: save eig2_mm.pkl, the means over the components of the smallest and the largest
    eigenvalue of Sigma^-1 Lambda (eig2s, sorted) at each step, as (lam_min_t, lam_max_t), each of shape (T,)."""
    import numpy as onp
    aux = os.path.join(exp_dir, 'aux_outputs')
    if inf_name not in ('lagsf', 'uagsf') or not os.path.exists(aux + '.pkl'):
        return
    aux_out = util.io.load(aux)
    if 'eig2s' not in aux_out:
        return
    eig = onp.asarray(aux_out['eig2s'])
    util.io.save((eig[..., 0].mean(axis=1), eig[..., -1].mean(axis=1)), os.path.join(exp_dir, 'eig2_mm'))


def _filter_label(inf_desc):
    """Label of a filter with its setting and, for the AGSF, the split rule of the update (e.g. 'L-AGSF (M=10,N=L=5) sdp 0.4')."""
    label = f'{RESULTS_CLASSES[inf_desc.name]().name} ({_setting_label(inf_desc)})'
    if inf_desc.name in ('lagsf', 'uagsf'):
        label += f' {inf_desc.ac2} {inf_desc.r2}'
    return label


def plot_dim(args):
    """
    Task A10 (sinlin, D-42). For each filter and state dimension d of the experiments in the files:
    - the RMSE of x_1 and of x_2..x_d: square roots of the bootstrap means of mse_x1 and mse_rest ('Error summary.txt');
    - the mean LPE and the NEES value of the summary;
    - AGSF: the mean over trials, steps and components of the smallest and the largest eigenvalue of Sigma^-1 Lambda
      (eig2_mm.pkl of 'errors'). For a scalar rule (prop, auto) both are rho2.
    The figure shows the settings with rot = 0 against d: the eigenvalues of the rules auto and sdp, and the RMSE of x_2..x_d.
    The Markdown table has all settings.
    """
    import numpy as onp

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    rows = {}   # (rot, filter label) -> list of (d, values)
    for e in exp_descs:
        folder = os.path.join(misc.get_root(), 'experiments', e.get_dir())
        summ = os.path.join(folder, 'Error summary.txt')
        if not os.path.exists(summ):
            print(f'dim: no summary for {e.get_dir()}, left out')
            continue
        v = read_error_summary(summ)
        val = {'res': RESULTS_CLASSES[e.inf.name](), 'trials': v['trials'], 'lpe': v['lpe'][0],
               'rmse_x1': onp.sqrt(v['mse_x1'][0]), 'rmse_rest': onp.sqrt(v['mse_rest'][0]),
               'outside': v['outside'][0] if 'outside' in v else float('nan'), 'rule': getattr(e.inf, 'ac2', None)}
        if e.inf.name in ('lagsf', 'uagsf'):
            mm = [util.io.load(os.path.join(folder, str(t), 'eig2_mm')) for t in range(args.start, args.end + 1)
                  if os.path.exists(os.path.join(folder, str(t), 'eig2_mm.pkl'))]
            if mm:
                val['lam_min'] = float(onp.mean([m[0] for m in mm]))
                val['lam_max'] = float(onp.mean([m[1] for m in mm]))
                val['eig_trials'] = len(mm)
        rows.setdefault((e.sim.rot, _filter_label(e.inf)), []).append((e.sim.state_dim, val))

    def short(label, val):
        # Legend label without the setting (the caption gives it); a GSF with one component is the EKF or the UKF
        name = val['res'].name
        if name in ('L-GSF', 'U-GSF') and '$M=1$' in label:
            return {'L-GSF': 'EKF', 'U-GSF': 'UKF'}[name]
        return name + (label[label.index(')') + 1:] if ')' in label else '')

    def style(val, label):
        rule = val['rule']
        if rule == 'prop' and label.endswith(' 0.001'):
            return '-.'
        return {'sdp': '-', 'auto': '--'}.get(rule, ':')

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 4.0))
    for (rot, label), pts in rows.items():
        if rot != 0:
            continue
        pts = sorted(pts, key=lambda p: p[0])
        ds = [p[0] for p in pts]
        v0 = pts[0][1]
        kw = dict(color=v0['res'].color, marker=v0['res'].marker, ms=3, lw=0.8, ls=style(v0, label))
        if v0['rule'] in ('auto', 'sdp') and 'lam_max' in v0:
            lab = short(label, v0) + (': largest' if v0['rule'] == 'sdp' else r' ($\rho_2$)')
            axes[0].plot(ds, [p[1]['lam_max'] for p in pts], label=lab, **kw)
            if v0['rule'] == 'sdp':
                axes[0].plot(ds, [p[1]['lam_min'] for p in pts], label=short(label, v0) + ': smallest', **{**kw, 'marker': 'v'})
        axes[1].plot(ds, [p[1]['rmse_rest'] for p in pts], label=short(label, v0), **kw)
    axes[0].set_ylabel(r'eigenvalue of $\Sigma^{-1}\Lambda$')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=5, loc='lower right', frameon=False)
    axes[1].set_ylabel(r'RMSE $x_2,\ldots,x_d$')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=5, ncol=2, loc='upper left', frameon=False)
    ds_all = sorted({p[0] for pts in rows.values() for p in pts})
    axes[1].set_xticks(ds_all)
    axes[1].set_xticklabels([str(d) for d in ds_all])
    axes[1].xaxis.set_minor_locator(plt.NullLocator())
    axes[1].set_xlabel('state dimension $d$')
    fig.tight_layout()
    sim_name = exp_descs[0].sim.name
    out = args.out or os.path.join(fig_dir, f'dim_{sim_name}.pdf')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print('saved', out)

    text = ''
    for rot in sorted({k[0] for k in rows}):
        for d in sorted({p[0] for (r, _), pts in rows.items() if r == rot for p in pts}):
            text += (f'### d = {d}' + (', rotated direction a' if rot else ', a = e_1') + '\n\n'
                     '| Filter | RMSE $x_1$ | RMSE $x_2..x_d$ | LPE | NEES outside | lam min | lam max | Trials |\n'
                     '|---|---|---|---|---|---|---|---|\n')
            for (r, label), pts in rows.items():
                if r != rot:
                    continue
                for dd, val in pts:
                    if dd != d:
                        continue
                    lam = f"{val['lam_min']:.3f} | {val['lam_max']:.3f}" if 'lam_min' in val else '- | -'
                    text += (f"| {label} | {val['rmse_x1']:.3f} | {val['rmse_rest']:.3f} | {val['lpe']:.4g} | "
                             f"{100 * val['outside']:.1f} % | {lam} | {val['trials']} |\n")
            text += '\n'
    if args.md:
        util.io.save_txt(text, args.md)
    print(text)


def make_table(args):
    """Markdown rows (MSE, LPE, lost tracks) of the experiments in the files, one table for each simulator setting.
    The values come from the 'Error summary.txt' files that 'main.py errors' writes (bootstrap mean +- sd)."""

    names = {'lgsf': 'L-GSF', 'ugsf': 'U-GSF', 'lagsf': 'L-AGSF', 'uagsf': 'U-AGSF',
             'bpf': 'BPF', 'apf': 'APF', 'rpf': 'RPF', 'upf': 'UPF'}
    size = lambda inf: getattr(inf, 'num_prt', None) or getattr(inf, 'n_comps', None) or getattr(inf, 'M', None)
    # AGSF: also the split rule of the update step (for example the rho2 sweep of task A6: '100, rho2 prop 0.3')
    base_size = size
    size = lambda inf: f'{inf.M}, rho2 {inf.ac2} {inf.r2}' if inf.name in ('lagsf', 'uagsf') else base_size(inf)
    fmt = lambda x: f'{x:.2e}' if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2) else f'{x:.2f}'

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])
    tables = {}
    for exp_desc in exp_descs:
        path = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir(), 'Error summary.txt')
        rows = tables.setdefault(exp_desc.sim.get_dir(), [])
        if not os.path.exists(path):
            rows.append(f'| {names[exp_desc.inf.name]} | {size(exp_desc.inf)} | no summary | | | |')
            continue
        v = read_error_summary(path)
        outside = f'{100 * v["outside"][0]:.1f} % ({v["empty"]})' if 'outside' in v and v['outside'][0] == v['outside'][0] else 'n/a'
        rows.append(f'| {names[exp_desc.inf.name]} | {size(exp_desc.inf)} | {fmt(v["mse"][0])} ± {fmt(v["mse"][1])} '
                    f'| {fmt(v["lpe"][0])} ± {fmt(v["lpe"][1])} | {outside} | {v["lost"]} of {v["trials"]} |')

    text = ''
    for sim_dir, rows in tables.items():
        text += (f'### {sim_dir}\n\n| Filter | M | MSE | LPE | NEES: outside 99.9 % (trials > 20 %) | Lost |\n'
                 f'|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n\n')
    if args.out:
        util.io.save_txt(text, args.out)
    print(text)


def eval_and_plot_errors(args, show=False):
    """
    Takes a set of experiments and line-plots the errors of each algorithm together.
    """

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        # Fixed bootstrap key for each experiment: BOOTSTRAP_SEED and a hash of the results folder. Thus the summary of an
        # experiment is the same at each run, whatever other descriptor files are in the call. (Before 02-Nov-2026 the
        # seed came from the clock, and the last digits changed at each run.)
        key = jr.fold_in(jr.PRNGKey(BOOTSTRAP_SEED), zlib.crc32(exp_desc.get_dir().encode()) & 0x7fffffff)

        lpe_trials = []
        bs_trials = []
        me_trials = []
        mse_trials = []
        x1_trials = []     # mse_x1 and mse_rest (split error, sinlin); NaN if a trial has neither results nor the saved files
        rest_trials = []
        out_trials = []    # fraction of steps with NEES above chi2_d(0.999) (nees.pkl); NaN if a trial has neither results nor nees.pkl
        inf_desc = exp_desc.inf
        sim_desc = exp_desc.sim

        for trial in range(args.start, args.end + 1):
            
            print(f'Working on trial {trial}')
            print(exp_desc.pprint())

            try:

                exp_root = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
                exp_dir = exp_root + '/' + str(trial)

                try:

                    states, observations, params = util.io.load(os.path.join(exp_dir, 'gt'))
                    results = util.io.load(os.path.join(exp_dir, 'results'))

                    lpe, lpe_t = log_prob_error(results, inf_desc.name, states)
                    bs, bs_t = bias(results, inf_desc.name, states)
                    me, me_t = min_error(results, inf_desc.name, states)
                    mse, mse_t = util.numerics.mse(results, inf_desc.name, states)
                    out, nees_t = nees(results, inf_desc.name, states)
                    x1, x1_t = mse_x1(results, inf_desc.name, states)
                    rest, rest_t = mse_rest(results, inf_desc.name, states)
                    num_simulations = get_num_sims(exp_desc)

                    util.io.save((lpe, lpe_t), os.path.join(exp_dir, 'lpe'))
                    util.io.save((bs, bs_t), os.path.join(exp_dir, 'bs'))
                    util.io.save((me, me_t), os.path.join(exp_dir, 'me'))
                    util.io.save((mse, mse_t), os.path.join(exp_dir, 'mse'))
                    util.io.save((out, nees_t), os.path.join(exp_dir, 'nees'))
                    util.io.save((x1, x1_t), os.path.join(exp_dir, 'mse_x1'))
                    util.io.save((rest, rest_t), os.path.join(exp_dir, 'mse_rest'))
                    util.io.save(num_simulations, os.path.join(exp_dir, 'num_sims'))
                    _save_eig_summary(exp_dir, inf_desc.name)

                    lpe_trials.append(lpe)
                    bs_trials.append(bs)
                    me_trials.append(me)
                    mse_trials.append(mse)
                    out_trials.append(out)
                    x1_trials.append(x1)
                    rest_trials.append(rest)

                except FileNotFoundError:

                    # The results of large runs can be deleted after 'errors' has saved the metrics. Then use the saved metrics.
                    try:

                        lpe_trials.append(util.io.load(os.path.join(exp_dir, 'lpe'))[0])
                        bs_trials.append(util.io.load(os.path.join(exp_dir, 'bs'))[0])
                        me_trials.append(util.io.load(os.path.join(exp_dir, 'me'))[0])
                        mse_trials.append(util.io.load(os.path.join(exp_dir, 'mse'))[0])
                        try:
                            out_trials.append(util.io.load(os.path.join(exp_dir, 'nees'))[0])
                        except FileNotFoundError:
                            out_trials.append(float('nan'))    # runs from before the NEES metric, with deleted results
                        try:
                            x1_trials.append(util.io.load(os.path.join(exp_dir, 'mse_x1'))[0])
                            rest_trials.append(util.io.load(os.path.join(exp_dir, 'mse_rest'))[0])
                        except FileNotFoundError:
                            x1_trials.append(float('nan'))     # runs from before the split error, with deleted results
                            rest_trials.append(float('nan'))
                        print('Results not found, saved metrics used')

                    except FileNotFoundError:

                        print('ERROR FILE NOT FOUND')

            except misc.AlreadyExistingExperiment:
                print('TRIAL DOES NOT EXIST')

        try:
            
            lpe_trials = jnp.array(lpe_trials)
            bs_trials = jnp.array(bs_trials)
            me_trials = jnp.array(me_trials)
            mse_trials = jnp.array(mse_trials)

            B = 100
            key, subkey = jr.split(key)
            bootstrap_lpe = util.numerics.bootstrap(subkey, lpe_trials, B)
            lpe_avg = jnp.mean(bootstrap_lpe)
            lpe_std = jnp.std(bootstrap_lpe)

            key, subkey = jr.split(key)
            bootstrap_bs = util.numerics.bootstrap(subkey, bs_trials, B)
            bs_avg = jnp.mean(bootstrap_bs)
            bs_std = jnp.std(bootstrap_bs)

            key, subkey = jr.split(key)
            bootstrap_me = util.numerics.bootstrap(subkey, me_trials, B)
            me_avg = jnp.mean(bootstrap_me)
            me_std = jnp.std(bootstrap_me)

            key, subkey = jr.split(key)
            bootstrap_mse = util.numerics.bootstrap(subkey, mse_trials, B)
            mse_avg = jnp.mean(bootstrap_mse)
            mse_std = jnp.std(bootstrap_mse)

            # NEES: bootstrap over the trials that have it (see out_trials)
            out_trials = jnp.array(out_trials)
            out_ok = out_trials[~jnp.isnan(out_trials)]
            key, subkey = jr.split(key)
            if out_ok.size:
                bootstrap_out = util.numerics.bootstrap(subkey, out_ok, B)
                out_line = f'outside: {jnp.mean(bootstrap_out):.6g} +- {jnp.std(bootstrap_out):.6g}, {jnp.median(out_ok):.6g}\n'
            else:
                out_line = 'outside: nan +- nan, nan\n'

            # Split error (mse_x1, mse_rest): bootstrap over the trials that have it
            split_lines = ''
            for lab, arr in (('mse_x1', x1_trials), ('mse_rest', rest_trials)):
                arr = jnp.array(arr)
                arr = arr[~jnp.isnan(arr)]
                key, subkey = jr.split(key)
                if arr.size:
                    bs_arr = util.numerics.bootstrap(subkey, arr, B)
                    split_lines += f'{lab}: {jnp.mean(bs_arr):.6g} +- {jnp.std(bs_arr):.6g}, {jnp.median(arr):.6g}\n'

            # Summary over the trials, printed and saved next to the histograms. The tables of the paper are made from this file.
            n_bad = lambda x: int(jnp.sum(jnp.isnan(x) + jnp.isinf(x)))
            summary = (f'{exp_desc.get_dir()}\n'
                       f'trials: {len(mse_trials)}\n'
                       f'non-finite trials (lpe, bias, minerr, mse): {n_bad(lpe_trials)}, {n_bad(bs_trials)}, {n_bad(me_trials)}, {n_bad(mse_trials)}\n'
                       f'trials with mse > 10: {int(jnp.sum(mse_trials > 10))}\n'
                       f'bootstrap mean +- sd (B={B}), median over trials\n'
                       f'lpe: {lpe_avg:.6g} +- {lpe_std:.6g}, {jnp.nanmedian(lpe_trials):.6g}\n'
                       f'bias: {bs_avg:.6g} +- {bs_std:.6g}, {jnp.nanmedian(bs_trials):.6g}\n'
                       f'minerr: {me_avg:.6g} +- {me_std:.6g}, {jnp.nanmedian(me_trials):.6g}\n'
                       f'mse: {mse_avg:.6g} +- {mse_std:.6g}, {jnp.nanmedian(mse_trials):.6g}\n'
                       + split_lines +
                       f'NEES: fraction of steps with NEES > chi2_d(0.999), the truth outside the 99.9 % region of the filter\n'
                       + out_line +
                       f'trials with > 20 % of steps outside: {int(jnp.sum(out_ok > 0.2))} of {out_ok.size}\n'
                       f'trials without NEES (no results.pkl and no nees.pkl): {int(out_trials.size - out_ok.size)}\n')
            print(summary)
            util.io.save_txt(summary, os.path.join(exp_root, 'Error summary.txt'))

            # remove nans
            nans_infs = jnp.isnan(lpe_trials) + jnp.isinf(lpe_trials)
            lpe_trials = lpe_trials[~nans_infs]
            nans_infs = jnp.isnan(bs_trials) + jnp.isinf(bs_trials)
            bs_trials = bs_trials[~nans_infs]
            nans_infs = jnp.isnan(me_trials) + jnp.isinf(me_trials)
            me_trials = me_trials[~nans_infs]
            nans_infs = jnp.isnan(mse_trials) + jnp.isinf(mse_trials)
            mse_trials = mse_trials[~nans_infs]

            if args.no_plots:

                continue

            try:

                fig, ax = plt.subplots(2, 2, figsize=(20, 10))

                ax[0, 0].hist(lpe_trials, alpha=0.5, label=f'{inf_desc.name}', density=True)
                ax[0, 0].axvline(lpe_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[0, 0].axvline(lpe_avg - lpe_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 0].axvline(lpe_avg + lpe_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 0].set_xlabel('-logprob')
                ax[0, 0].set_ylabel('count')
                ax[0, 0].set_title(f'{inf_desc.name} - Error histogram %')
                ax[0, 0].legend(prop={'size': 5})

                ax[0, 1].hist(bs_trials, alpha=0.5, label=f'{inf_desc.name}', density=True)
                ax[0, 1].axvline(bs_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[0, 1].axvline(bs_avg - bs_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 1].axvline(bs_avg + bs_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[0, 1].set_xlabel('bias')
                ax[0, 1].set_ylabel('count')
                ax[0, 1].set_title(f'{inf_desc.name} - Bias histogram %')
                ax[0, 1].legend(prop={'size': 5})

                ax[1, 0].hist(me_trials, alpha=0.5, label=f'{inf_desc.name}', density=True)
                ax[1, 0].axvline(me_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[1, 0].axvline(me_avg - me_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 0].axvline(me_avg + me_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 0].set_xlabel('minerr')
                ax[1, 0].set_ylabel('count')
                ax[1, 0].set_title(f'{inf_desc.name} - MinError histogram %')
                ax[1, 0].legend(prop={'size': 5})

                ax[1, 1].hist(mse_trials, alpha=0.5, label=f'{inf_desc.name}', density=True)
                ax[1, 1].axvline(mse_avg, color='green', linestyle='dashed', linewidth=0.5)
                ax[1, 1].axvline(mse_avg - mse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 1].axvline(mse_avg + mse_std, color='gray', linestyle='dotted', linewidth=0.25)
                ax[1, 1].set_xlabel('mse')
                ax[1, 1].set_ylabel('count')
                ax[1, 1].set_title(f'{inf_desc.name} - MSE histogram %')
                ax[1, 1].legend(prop={'size': 5})

                fig.savefig(os.path.join(f'{exp_root}', f'Error histograms.png'))

                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(lpe_trials, label='lpe')
                ax.plot(bs_trials, label='bias')
                ax.plot(me_trials, label='minerr')
                ax.plot(mse_trials, label='mse')

                ax.set_xlabel('Trial')
                ax.set_ylabel('Error')
                ax.set_title(f'{inf_desc.name} - Errors vs Trial')
                ax.legend(prop={'size': 5})

                fig.savefig(os.path.join(f'{exp_root}', f'Errors vs Trials.png'))

            except ValueError as e:

                print(f'Error plotting {inf_desc.name}: {e}')
                print(f'No {inf_desc.name} results to plot')


        except FileNotFoundError:

            print('Experiment Empty')
    
    if show: 
        plt.show()


def main():

    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()