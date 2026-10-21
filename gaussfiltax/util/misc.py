from io import StringIO
import re
import numpy as np # type: ignore
import jax.numpy as jnp # type: ignore
from jax import lax, vmap, random as jr # type: ignore
from jax.tree_util import tree_map # type: ignore
import util.io
import pandas as pd
import os
import experiment_descriptor as ed
from typing import NamedTuple
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN # type: ignore


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r'\s+')
    return p.sub('', str)


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, y, one_datapoint


def look_up(name):

    lu_field = {'i': '0', 'd': '1', 'e': '2'}
    field_cd = lu_field[name[0]]
    param_cd = str(int(name[1]) - 1)
    
    return field_cd + param_cd


def get_bool_tree(target_vars, param_names):
    r"""Returns a tree of booleans with the same structure as target_vars.
    """
    is_target = []
    target_vars = list(map(lambda var: look_up(var), target_vars))
    for i, field in enumerate(param_names):
        sub_is_target = []
        for j, _ in enumerate(field):
            if str(i) + str(j) in target_vars:
                sub_is_target.append(True)
            else:
                sub_is_target.append(False)
        is_target.append(sub_is_target)
    return is_target


def get_prior_fields(init_vals, param_dists, is_target):    
    r"""Sample parameters from the prior distribution. When prior field is tfd.Distribution,
        sample num_samples values from it, and set the corresponding values of is_constrained to False
        (we consider that parameters are sampled always in unconstrained form). Otherwise set the value 
        equal to the provided value and is_constrained to True.
    """
    return tree_map(lambda b, l1, l2: l2 if b else l1, is_target, init_vals, param_dists)


def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))


def get_exp_dir(inf,
            sim,
            state_dim,
            emission_dim,
            num_timesteps,
            a = None,
            b = None,
            n_comps = None, 
            M = None,
            N = None,
            L = None,
            ac1 = None, 
            r1 = None,
            ac2 = None,
            r2 = None,
            n_iters = None,
            alpha=None,
            beta=None,
            kappa=None,
            ess=None):
    
    data_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/data.nosync/experiments/'

    if inf == 'lgsf':

        inf_dir = f'gsf/lgsf/ncomps_{n_comps}_iters_{n_iters}'

    elif inf == 'ugsf':

        inf_dir = f'gsf/ugsf/ncomps_{n_comps}_iters_{n_iters}_alpha_{alpha}_beta_{beta}_kappa_{kappa}'

    elif inf == 'lagsf':

        inf_dir = f'agsf/lagsf/M_{M}_N_{N}_L_{L}_iters_{n_iters}_ac1_{ac1}_r1_{r1}_ac2_{ac2}_r2_{r2}'

    elif inf == 'uagsf':

        inf_dir = f'agsf/uagsf/M_{M}_N_{N}_L_{L}_iters_{n_iters}_ac1_{ac1}_r1_{r1}_ac2_{ac2}_r2_{r2}'

    elif inf == 'bpf':

        inf_dir = f'pf/bpf/numprt_{M}_ess_{ess}'

    elif inf == 'apf':

        inf_dir = f'pf/apf/numprt_{M}_ess_{ess}'

    sim_dir = data_root + f'{sim}/state-dim_{state_dim}_emission-dim_{emission_dim}_num-timesteps_{num_timesteps}/'

    if sim == 'covinfl':

        sim_dir = data_root + f'{sim}/emission-dim_{emission_dim}_num-timesteps_{num_timesteps}_a_{a}_b_{b}/'

    exp_dir = sim_dir + inf_dir

    return exp_dir


def get_exp_data(exp_dir, start_trial, end_trial):

    exp_data = {
        'results': [],
        'gt': [],
        'lpe': [],
        'mse': [],
        'bs': [],
        'me': [],
        'aux_outputs': []
        }
    
    for trial in range(start_trial, end_trial+1):

        for entry in exp_data.keys():
            
            try:

                exp_data[entry].append(util.io.load(os.path.join(exp_dir + f'/{trial}', entry)))

            except FileNotFoundError:

                continue

    return exp_data


def get_data_for(alg, sim_args, inf_args, start_trial =1, end_trial=20):

    ssm = sim_args['ssm']
    state_dim = sim_args['state_dim']
    emission_dim = sim_args['emission_dim']
    num_timesteps = sim_args['num_timesteps']
    a = sim_args.get('a', None)
    b = sim_args.get('b', None)
    
    if alg == 'lgsf':

        n_comps = inf_args['n_comps']
        n_iters = inf_args['n_iters']

        exp_dir = get_exp_dir('lgsf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                n_comps=n_comps, 
                                n_iters=n_iters,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    elif alg == 'ugsf':

        n_comps = inf_args['n_comps']
        n_iters = inf_args['n_iters']
        alpha = inf_args['alpha']
        beta = inf_args['beta']
        kappa = inf_args['kappa']

        exp_dir = get_exp_dir('ugsf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                n_comps=n_comps, 
                                n_iters=n_iters,
                                alpha=alpha,
                                beta=beta,
                                kappa=kappa,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    elif alg == 'lagsf':

        M = inf_args['M']
        N = inf_args['N']
        L = inf_args['L']
        n_iters = inf_args['n_iters']
        ac1 = inf_args['ac1']
        r1 = inf_args['r1']
        ac2 = inf_args['ac2']
        r2 = inf_args['r2']

        exp_dir = get_exp_dir('lagsf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                M=M, 
                                N=N, 
                                L=L, 
                                n_iters=n_iters,
                                ac1=ac1,
                                r1=r1,
                                ac2=ac2,
                                r2=r2,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    elif alg == 'uagsf':

        M = inf_args['M']
        N = inf_args['N']
        L = inf_args['L']
        n_iters = inf_args['n_iters']
        ac1 = inf_args['ac1']
        r1 = inf_args['r1']
        ac2 = inf_args['ac2']
        r2 = inf_args['r2']

        exp_dir = get_exp_dir('uagsf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                M=M, 
                                N=N, 
                                L=L, 
                                n_iters=n_iters,
                                ac1=ac1,
                                r1=r1,
                                ac2=ac2,
                                r2=r2,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    elif alg == 'bpf':

        M = inf_args['M']
        ess = inf_args['ess']

        exp_dir = get_exp_dir('bpf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                M=M, 
                                ess=ess,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    elif alg == 'apf':

        M = inf_args['M']
        ess = inf_args['ess']

        exp_dir = get_exp_dir('apf', 
                                ssm, 
                                state_dim=state_dim, 
                                emission_dim=emission_dim, 
                                num_timesteps=num_timesteps, 
                                M=M, 
                                ess=ess,
                                a = a,
                                b = b)

        out = get_exp_data(exp_dir, start_trial, end_trial)

    ave_path = os.path.join(exp_dir, 'averaged_errors.txt')

    if os.path.isfile(ave_path):

        with open(ave_path, 'r') as f:

            text = f.read()

        df = pd.read_csv(StringIO(text), header=None)

        err_names = df[0].values
        err_means = df[1].values
        err_stds = df[2].values

        err_dict = {str(name): (float(mean), float(std))
                for name, mean, std in zip(err_names, err_means, err_stds)}

        out['averaged_errors'] = err_dict

    return out


def str_to_bool(str):

    """
    Converts a string to a boolean value.
    :param str: string to convert
    :return: boolean value
    """

    if str.lower() == 'true':

        return True

    elif str.lower() == 'false':

        return False

    else:
        raise ValueError(f"Invalid boolean string: {str}")
    

def print_table(rows):

    # Find max width for each column
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    row_names = ['ABC', 'MCMC', 'SNL', 'TSNL']
    
    for row_name, row in zip(row_names, rows):
        print(row_name.ljust(5) + " | " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))


def get_params(x, dim, t):
    return x[t] if x.ndim == dim + 1 else x


def process_fn(f, u):
    return (lambda x, y: f(x)) if u is None else f


def process_input(x, y):
    return jnp.zeros((y, 1)) if x is None else x


def MVN_log_prob(mean, cov, y):
    return MVN(mean, cov).log_prob(jnp.atleast_1d(y))


def swap_axes_on_values(outputs, axis1=0, axis2=1):
    return dict(map(lambda x: (x[0], jnp.swapaxes(x[1], axis1, axis2)), outputs.items()))
