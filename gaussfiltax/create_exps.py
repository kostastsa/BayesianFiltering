import experiment_descriptor as ed
import sys
import os
import itertools
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as onp # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Default parameter configurations for each algorithm
# Format: {param_name: list_of_values_to_scan}
# Fixed parameters should not be in this dict, only scanned ones
DEFAULT_SCAN_PARAMS = {
    'lgsf': {
        'n_comps': [1, 10, 100],
        'n_iters': [1],
    },
    'ugsf': {
        'n_comps': [1, 10, 100],
        'n_iters': [1],
        'alpha': [1.0],
        'beta': [0.0],
        'kappa': [0.0],
    },
    'lagsf': {
        'M': [2, 10, 100],
        'N': [5],
        'L': [5],
        'n_iters': [1],
        'ac1': ['prop'],
        'r1': [0.95],
        'ac2': ['prop'],
        'r2': [0.95],
    },
    'uagsf': {
        'M': [2, 10, 100],
        'N': [5],
        'L': [5],
        'n_iters': [1],
        'alpha': [1.0],
        'beta': [0.0],
        'kappa': [0.0],
        'ac1': ['prop'],
        'r1': [0.95],
        'ac2': ['prop'],
        'r2': [0.95],
    },
    'bpf': {
        'num_prt': [100, 500, 1000],
        'ess': [0.5],
    },
    'apf': {
        'num_prt': [100, 500, 1000],
        'ess': [0.5],
    },
}


def create_descs(
    sim_desc,
    alg,
    scan_params=None,
):
    """
    Create experiment descriptors for a given algorithm with flexible parameter scanning.
    
    Parameters
    ----------
    sim_desc : str
        Simulator descriptor
    alg : str
        Algorithm name ('lgsf', 'ugsf', 'lagsf', 'uagsf', 'bpf', 'apf')
    scan_params : dict, optional
        Dictionary specifying parameters to scan. Format: {param_name: [list of values]}.
        If None, uses DEFAULT_SCAN_PARAMS[alg].
        
    Returns
    -------
    exp_descs : str
        Concatenated experiment descriptors
    scan_info : tuple
        (first_param_name, init_value, fin_value) for naming purposes
    """
    
    if scan_params is None:

        scan_params = DEFAULT_SCAN_PARAMS.get(alg, {})
    
    exp_descs = ''
    inf = ed.InferenceDescriptor().get_descriptor(alg)
    
    # Get parameter names and values
    param_names = list(scan_params.keys())
    param_values = [scan_params[name] for name in param_names]
    
    # Generate all combinations
    for combination in itertools.product(*param_values):

        params_dict = dict(zip(param_names, combination))
        
        # Call the appropriate create_desc method based on algorithm
        if alg == 'lgsf':

            inf_desc = inf.create_desc(
                params_dict['n_comps'],
                params_dict['n_iters']
            )

        elif alg == 'ugsf':

            inf_desc = inf.create_desc(
                params_dict['n_comps'],
                params_dict['n_iters'],
                params_dict['alpha'],
                params_dict['beta'],
                params_dict['kappa']
            )

        elif alg == 'lagsf':

            inf_desc = inf.create_desc(
                params_dict['M'],
                params_dict['N'],
                params_dict['L'],
                params_dict['n_iters'],
                params_dict['ac1'],
                params_dict['r1'],
                params_dict['ac2'],
                params_dict['r2']
            )

        elif alg == 'uagsf':

            inf_desc = inf.create_desc(
                params_dict['M'],
                params_dict['N'],
                params_dict['L'],
                params_dict['n_iters'],
                params_dict['alpha'],
                params_dict['beta'],
                params_dict['kappa'],
                params_dict['ac1'],
                params_dict['r1'],
                params_dict['ac2'],
                params_dict['r2']
            )

        elif alg in ['bpf', 'apf']:

            inf_desc = inf.create_desc(
                params_dict['num_prt'],
                params_dict['ess']
            )

        else:

            print(f'unknown algorithm: {alg}')
            return '', ('', None, None)
        
        exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc)
        exp_descs += '\n'
    
    # Determine scan info for naming: use first scanned parameter
    if param_names:

        first_param = param_names[0]
        scan_init = scan_params[first_param][0]
        scan_fin = scan_params[first_param][-1]
        scan_info = (first_param, scan_init, scan_fin)

    else:

        scan_info = ('', None, None)
    
    return exp_descs, scan_info

if __name__ == '__main__':

    exp_root = '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/exps'
    ssm = 'mtt'
    sim = ed.SimulatorDescriptor().get_descriptor(ssm)

    # Files that are written. 'apf_ess2' is the APF that resamples the ancestors at every step (ess 2.0: the ESS is always below 2 M),
    # the rule of the first version of the paper.
    algs_to_write = ['bpf', 'apf', 'apf_ess2']

    # mtt only: the (a, sigma2) settings. Each setting has its own folder of descriptor files and its own results folder.
    # (None, None) keeps the defaults of simulators/mtt.py (a = 0.05, sigma2 = 2.5) and writes to the folder state_dim_4.
    mtt_settings = [(a, s) for a in [0.5, 0.05] for s in [2.5, 0.025, 2.5e-5]]

    if ssm == 'covinfl':

        state_dim = 1
        emission_dim = 1
        num_timesteps = 100
        a = 10 
        b = 0.1
        sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, a, b)

    elif ssm == 'mtt':
        
        state_dim = 4
        emission_dim = 2
        num_timesteps = 500
        sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps)

    elif ssm == 'lmsv':

        state_dim = 4
        emission_dim = 4
        num_timesteps = 200
        sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps)

    else: 

        print('unknown ssm')

    # Generate experiments with r1 and r2 scanning for lagsf and uagsf
    r_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    custom_scan_params_lgsf = {
        'n_comps': [10, 100, 1000],
        'n_iters': [1],
    }

    custom_scan_params_ugsf = {
        'n_comps': [10, 100, 1000],
        'n_iters': [1],
        'alpha': [1.0],
        'beta': [0.0],
        'kappa': [0.0],
    }
    
    custom_scan_params_lagsf = {
        'M': [10, 100, 1000],
        'N': [5],
        'L': [5],
        'n_iters': [1],
        'ac1': ['prop'],
        'r1': [0.95],
        'ac2': ['auto'],
        'r2': [2/5],
    }
    
    custom_scan_params_uagsf = {
        'M': [10, 100, 1000],
        'N': [5],
        'L': [5],
        'n_iters': [1],
        'alpha': [1.0],
        'beta': [0.0],
        'kappa': [0.0],
        'ac1': ['prop'],
        'r1': [0.95],
        'ac2': ['auto'],
        'r2': [2/5],
    }
    
    custom_scan_params_bpf = {
        'num_prt': [100, 1000, 10000, 100000],
        'ess': [0.5],
    }

    for a, sigma2 in (mtt_settings if ssm == 'mtt' else [(None, None)]):

        out_dir = exp_root + f'/{ssm}/state_dim_{state_dim}'

        if ssm == 'mtt' and a is not None:

            sim_desc = sim.create_desc(state_dim, emission_dim, num_timesteps, a, sigma2)
            out_dir += f'/a_{a}_sigma2_{sigma2}'

        for name, alg, scan_params in [('lgsf', 'lgsf', custom_scan_params_lgsf),
                                       ('ugsf', 'ugsf', custom_scan_params_ugsf),
                                       ('lagsf', 'lagsf', custom_scan_params_lagsf),
                                       ('uagsf', 'uagsf', custom_scan_params_uagsf),
                                       ('bpf', 'bpf', custom_scan_params_bpf),
                                       ('apf', 'apf', custom_scan_params_bpf),
                                       ('apf_ess2', 'apf', {'num_prt': [1000], 'ess': [2.0]})]:

            if name not in algs_to_write:

                continue

            exp_descs, (scan_param, scan_init, scan_fin) = create_descs(
                sim_desc=sim_desc,
                alg=alg,
                scan_params=scan_params
            )

            os.makedirs(out_dir, exist_ok=True)

            with open(out_dir + f'/{ssm}_{name}_scan_num_prt.txt', 'w') as f:

                f.write(exp_descs)