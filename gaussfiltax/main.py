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

from util.numerics import log_prob_error, min_error, bias, get_num_sims, nees
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
            m = re.match(r'(lpe|bias|minerr|mse|outside): (\S+) \+- (\S+), (\S+)', line)
            if m:
                out[m.group(1)] = tuple(float(v) for v in m.groups()[1:])
    return out


def make_table(args):
    """Markdown rows (MSE, LPE, lost tracks) of the experiments in the files, one table for each simulator setting.
    The values come from the 'Error summary.txt' files that 'main.py errors' writes (bootstrap mean +- sd)."""

    names = {'lgsf': 'L-GSF', 'ugsf': 'U-GSF', 'lagsf': 'L-AGSF', 'uagsf': 'U-AGSF',
             'bpf': 'BPF', 'apf': 'APF', 'rpf': 'RPF', 'upf': 'UPF'}
    size = lambda inf: getattr(inf, 'num_prt', None) or getattr(inf, 'n_comps', None) or getattr(inf, 'M', None)
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
                    f'| {fmt(v["lpe"][0])} ± {fmt(v["lpe"][1])} | {v["lost"]} of {v["trials"]} | {outside} |')

    text = ''
    for sim_dir, rows in tables.items():
        text += (f'### {sim_dir}\n\n| Filter | M | MSE | LPE | Lost | Outside 99.9 % (trials > 20 %) |\n'
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
                    num_simulations = get_num_sims(exp_desc)

                    util.io.save((lpe, lpe_t), os.path.join(exp_dir, 'lpe'))
                    util.io.save((bs, bs_t), os.path.join(exp_dir, 'bs'))
                    util.io.save((me, me_t), os.path.join(exp_dir, 'me'))
                    util.io.save((mse, mse_t), os.path.join(exp_dir, 'mse'))
                    util.io.save((out, nees_t), os.path.join(exp_dir, 'nees'))
                    util.io.save(num_simulations, os.path.join(exp_dir, 'num_sims'))

                    lpe_trials.append(lpe)
                    bs_trials.append(bs)
                    me_trials.append(me)
                    mse_trials.append(mse)
                    out_trials.append(out)

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