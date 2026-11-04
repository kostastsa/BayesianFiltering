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
    'rpf': {
        'num_prt': [100, 500, 1000],
        'ess': [0.5],
    },
    'upf': {
        'num_prt': [100, 500, 1000],
        'ess': [0.5],
        'alpha': [1.0],
        'beta': [0.0],
        'kappa': [0.0],
    },
}


def create_descs(
    sim_desc,
    alg,
    scan_params=None,
    data_desc=None,
):
    """
    Create experiment descriptors for a given algorithm with flexible parameter scanning.
    
    Parameters
    ----------
    sim_desc : str
        Simulator descriptor
    alg : str
        Algorithm name ('lgsf', 'ugsf', 'lagsf', 'uagsf', 'bpf', 'apf', 'rpf', 'upf')
    scan_params : dict, optional
        Dictionary specifying parameters to scan. Format: {param_name: [list of values]}.
        If None, uses DEFAULT_SCAN_PARAMS[alg].
    data_desc : str, optional
        Data block (ed.DataDescriptor().create_desc): run on the stored data of another experiment.
        
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

        elif alg in ['bpf', 'apf', 'rpf']:

            inf_desc = inf.create_desc(
                params_dict['num_prt'],
                params_dict['ess']
            )

        elif alg == 'upf':

            inf_desc = inf.create_desc(
                params_dict['num_prt'],
                params_dict['ess'],
                params_dict['alpha'],
                params_dict['beta'],
                params_dict['kappa']
            )

        else:

            print(f'unknown algorithm: {alg}')
            return '', ('', None, None)
        
        exp_descs += ed.ExperimentDescriptor().create_desc(sim_desc, inf_desc, data_desc)
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
    # The simulator is the first argument (default mtt): python gaussfiltax/create_exps.py lmsv
    ssm = sys.argv[1] if len(sys.argv) > 1 else 'mtt'
    sim = ed.SimulatorDescriptor().get_descriptor(ssm)

    # Files that are written. 'apf_ess2' is the APF that resamples the ancestors at every step (ess 2.0: the ESS is always below 2 M),
    # the rule of the first version of the paper.
    # 'bpf_samedata' and 'apf_samedata' run again on the data of the 30 trials of the runs with 1000 particles (data block):
    # trials 1..30 R give R runs on each data set (check of the implementation against the particles library).
    # '*_table' are the GSF and AGSF rows of Tables I and III of the paper (task T1).
    # 'rpf' and 'upf' are the new baselines for Reviewer 2 (task A1, D-26, D-27, D-29).
    # '*_wrap': the same rows with the wrapped bearing (sim key 'wrap: 1', pilot of D-30). Same filters in the same order
    # as the files without '_wrap', thus with the same SEED_BASE every trial gets the same key and the same states.
    # Earlier sets: ['bpf', 'apf', 'apf_ess2', 'bpf_samedata', 'apf_samedata'], ['lgsf_table', 'ugsf_table', 'lagsf_table', 'uagsf_table'],
    # ['rpf', 'upf'].
    # D-30 decided (c): the tables use the wrapped bearing. The GSF and AGSF rows are the '*_table_wrap' files (the pilot);
    # the PF rows are 'bpf_wrap', 'apf_wrap' (SEED_BASE 1000, as A2) and 'rpf_wrap', 'upf_wrap' (SEED_BASE 3000, as A1).
    algs_to_write = ['bpf_wrap', 'apf_wrap', 'rpf_wrap', 'upf_wrap']
    # Table R1 with the wrapped bearing (D-33): runs on the 30 data sets of the wrapped BPF/APF runs with 1000 particles.
    algs_to_write = ['bpf_wrap_samedata', 'apf_wrap_samedata']
    # Exp. B (lmsv), Table II of the paper: '*_tableII' (task T2 and A1, D-32).
    if ssm == 'lmsv':
        algs_to_write = ['lgsf_tableII', 'ugsf_tableII', 'lagsf_tableII', 'uagsf_tableII',
                         'bpf_tableII', 'apf_tableII', 'rpf_tableII', 'upf_tableII']
        # A6 (D-35): sweep of a fixed rho2 (rule 'prop'), M = 100, on the 100 data sets of the adaptive Table II rows (data block).
        algs_to_write = ['lagsf_rho2sweep', 'uagsf_rho2sweep']
        # D-38: the rule 'auto' in the update with r2 in {0.1, 0.2, 0.8} (Table II uses 0.4), on the same 100 data sets as A6.
        algs_to_write = ['lagsf_r2auto', 'uagsf_r2auto']

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

    # RPF and UPF rows of Tables I and III (Exp. A, task A1, D-27): 100, 1000 and 10000 particles.
    table_params_rpf = {'num_prt': [100, 1000, 10000], 'ess': [0.5]}
    table_params_upf = {**table_params_rpf, 'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0]}

    # Table II (Exp. B, lmsv, D-32): M = 10, 100, 1000 for the GSF and AGSF, 100, 1000, 10000 particles for the PFs.
    # AGSF: N = L = 5, rho1 = 0.9 fixed (rule 'prop'), rho2 from the automatic rule with r2 = 2 / L = 0.4 (Sec. V-B).
    t2_gsf = {'n_comps': [10, 100, 1000], 'n_iters': [1]}
    t2_ukf = {'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0]}
    t2_agsf = {'M': [10, 100, 1000], 'N': [5], 'L': [5], 'n_iters': [1]}
    t2_split = {'ac1': ['prop'], 'r1': [0.9], 'ac2': ['auto'], 'r2': [0.4]}
    t2_pf = {'num_prt': [100, 1000, 10000], 'ess': [0.5]}
    # A6 (D-35): fixed rho2 in {0.1, 0.3, 0.5, 0.7, 0.9}, M = 100, N = L = 5, rho1 = 0.9. Each run uses the data of the
    # same trial of the adaptive row with M = 100 of Table II (T2_AGSF_SRC), so the comparison is paired.
    t2_sweep = {'M': [100], 'N': [5], 'L': [5], 'n_iters': [1], 'ac1': ['prop'], 'r1': [0.9], 'ac2': ['prop'], 'r2': [0.1, 0.3, 0.5, 0.7, 0.9]}
    T2_AGSF_SRC = 'agsf/{}/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4'
    # D-38: as A6, but with the rule 'auto' in the update and other values of r2.
    t2_r2auto = {**t2_sweep, 'ac2': ['auto'], 'r2': [0.1, 0.2, 0.8]}

    # GSF and AGSF rows of Tables I and III (Exp. A): M = 1, 100, 1000 components for the GSF, and
    # M = 2, 10, 100 with N = L = 5 and rho1 = rho2 = 0.9 (rule 'prop') for the AGSF.
    table_params_gsf = {'n_comps': [1, 100, 1000], 'n_iters': [1]}
    table_params_ugsf = {**table_params_gsf, 'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0]}
    table_params_agsf = {'M': [2, 10, 100], 'N': [5], 'L': [5], 'n_iters': [1]}
    table_split = {'ac1': ['prop'], 'r1': [0.9], 'ac2': ['prop'], 'r2': [0.9]}
    table_params_lagsf = {**table_params_agsf, **table_split}
    table_params_uagsf = {**table_params_agsf, 'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0], **table_split}

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
                                       ('apf_ess2', 'apf', {'num_prt': [1000], 'ess': [2.0]}),
                                       ('bpf_samedata', 'bpf', {'num_prt': [1000], 'ess': [0.5]}),
                                       ('apf_samedata', 'apf', {'num_prt': [1000], 'ess': [0.5]}),
                                       ('rpf', 'rpf', table_params_rpf),
                                       ('upf', 'upf', table_params_upf),
                                       ('lgsf_table', 'lgsf', table_params_gsf),
                                       ('ugsf_table', 'ugsf', table_params_ugsf),
                                       ('lagsf_table', 'lagsf', table_params_lagsf),
                                       ('uagsf_table', 'uagsf', table_params_uagsf),
                                       ('lgsf_table_wrap', 'lgsf', table_params_gsf),
                                       ('ugsf_table_wrap', 'ugsf', table_params_ugsf),
                                       ('lagsf_table_wrap', 'lagsf', table_params_lagsf),
                                       ('uagsf_table_wrap', 'uagsf', table_params_uagsf),
                                       ('bpf_wrap', 'bpf', custom_scan_params_bpf),
                                       ('apf_wrap', 'apf', custom_scan_params_bpf),
                                       ('rpf_wrap', 'rpf', table_params_rpf),
                                       ('upf_wrap', 'upf', table_params_upf),
                                       ('bpf_wrap_samedata', 'bpf', {'num_prt': [1000], 'ess': [0.5]}),
                                       ('apf_wrap_samedata', 'apf', {'num_prt': [1000], 'ess': [0.5]}),
                                       ('lgsf_tableII', 'lgsf', t2_gsf),
                                       ('ugsf_tableII', 'ugsf', {**t2_gsf, **t2_ukf}),
                                       ('lagsf_tableII', 'lagsf', {**t2_agsf, **t2_split}),
                                       ('uagsf_tableII', 'uagsf', {**t2_agsf, **t2_ukf, **t2_split}),
                                       ('bpf_tableII', 'bpf', t2_pf),
                                       ('apf_tableII', 'apf', t2_pf),
                                       ('rpf_tableII', 'rpf', t2_pf),
                                       ('upf_tableII', 'upf', {**t2_pf, **t2_ukf}),
                                       ('lagsf_rho2sweep', 'lagsf', t2_sweep),
                                       ('uagsf_rho2sweep', 'uagsf', {**t2_sweep, **t2_ukf}),
                                       ('lagsf_r2auto', 'lagsf', t2_r2auto),
                                       ('uagsf_r2auto', 'uagsf', {**t2_r2auto, **t2_ukf})]:

            if name not in algs_to_write:

                continue

            data_desc = ed.DataDescriptor().create_desc(f'pf/{alg}/numprt_1000_ess_0.5', 30) if name.endswith('_samedata') else None
            if name.endswith(('_rho2sweep', '_r2auto')):
                data_desc = ed.DataDescriptor().create_desc(T2_AGSF_SRC.format(alg), 100)
            exp_descs, (scan_param, scan_init, scan_fin) = create_descs(
                sim_desc=sim.create_desc(state_dim, emission_dim, num_timesteps, a, sigma2, wrap=True) if '_wrap' in name else sim_desc,
                alg=alg,
                scan_params=scan_params,
                data_desc=data_desc
            )

            os.makedirs(out_dir, exist_ok=True)

            with open(out_dir + f'/{ssm}_{name}_scan_num_prt.txt', 'w') as f:

                f.write(exp_descs)