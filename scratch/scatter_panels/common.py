"""Scatter panels of Exp. A (side figure, CLAUDE.md rule 5): the settings, filters and folders that the scripts share.

For each (a, sigma2) setting of Tables I and III: one figure, 8 columns (the filters at the middle sizes), 3 rows (the
best, median and worst run by MSE).
- Version 1 (v1): the table runs. Each filter has its own data, so each cell can show a different trajectory.
- Version 2 (v2): all filters on the same 40 stored data sets (data block, source SRC). Each row shows one trajectory.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))

SETTINGS = [(a, s) for a in ('0.5', '0.05') for s in ('2.5', '0.025', '2.5e-05')]
TRIALS = 100
REPRO_ROOT = os.path.join(ROOT, 'data.nosync', 'repro')      # GAUSSFILTAX_ROOT of the exact repeats of version 1
SRC = 'gsf/ugsf/ncomps_1_iters_1_alpha_1.0_beta_0.0_kappa_0.0'  # data source of version 2 (the U-GSF row with M = 1)
V2_SEED_BASE = 9500
V2_TRIALS = 40   # the user chose 40 paired data sets (05-Nov-2026), not 100

# (label, inference folder, group). Group: the name list and the SEED_BASE of the table runs (run_tables_I_III.zsh).
FILTERS = [
    ('L-GSF', 'gsf/lgsf/ncomps_100_iters_1', 'gsf'),
    ('U-GSF', 'gsf/ugsf/ncomps_100_iters_1_alpha_1.0_beta_0.0_kappa_0.0', 'gsf'),
    ('L-AGSF', 'agsf/lagsf/M_10_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9', 'gsf'),
    ('U-AGSF', 'agsf/uagsf/M_10_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9', 'gsf'),
    ('BPF', 'pf/bpf/numprt_1000_ess_0.5', 'pf1'),
    ('APF', 'pf/apf/numprt_1000_ess_0.5', 'pf1'),
    ('RPF', 'pf/rpf/numprt_1000_ess_0.5', 'pf2'),
    ('UPF', 'pf/upf/numprt_1000_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0', 'pf2'),
]
GROUPS = {'gsf': (['lgsf_table_wrap', 'ugsf_table_wrap', 'lagsf_table_wrap', 'uagsf_table_wrap'], 2000),
          'pf1': (['bpf_wrap', 'apf_wrap'], 1000),
          'pf2': (['rpf_wrap', 'upf_wrap'], 3000)}
NAME = {inf: lab for lab, inf, _ in FILTERS}


def sim_dir(a, s):
    return f'mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}_wrap'


def table_dir(a, s, inf):
    return os.path.join(ROOT, 'data.nosync', 'experiments', sim_dir(a, s), inf)


def v2_dir(a, s, inf):
    return os.path.join(ROOT, 'data.nosync', 'experiments', sim_dir(a, s), 'data_' + SRC.replace('/', '_'), inf)


def group_files(group, a):
    """The descriptor files of one call of run_tables_I_III.zsh, in its order: for each name, the sigma2 folders in
    the sort order of the zsh glob."""
    names, _ = GROUPS[group]
    out = []
    for n in names:
        out += sorted(glob.glob(os.path.join(ROOT, f'exps/mtt/state_dim_4/a_{a}_sigma2_*/mtt_{n}_scan_num_prt.txt')))
    return out


def v2_file(a, s):
    return os.path.join(HERE, 'descs', f'paired_a{a}_s{s}.txt')


def mse_of(folder, t):
    import util.io
    return float(util.io.load(os.path.join(folder, str(t), 'mse'))[0])


def best_median_worst(values):
    """(trial of the lowest, median and highest value). values: dict trial -> value. Median: the lower middle value."""
    order = sorted(values, key=lambda t: values[t])
    return order[0], order[(len(order) - 1) // 2], order[-1]
