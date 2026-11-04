"""Showcase of plan step 6 (side question, CLAUDE.md rule 5): a nonlinearity in one direction of a large state.

Simulator sinlin (gaussfiltax/simulators/sinlin.py): x' = 0.9 x + q, y_1 = kappa sin(omega a^T x) + r_1, y_j = x_j + r_j
(j = 2..d), Q = R = 0.1 I, x_0 ~ N(0, I), kappa = 1, omega = 2, T = 100.

For each d in DIMS (rot = 0) and each d in ROT_DIMS (rot = 1), this script writes two descriptor files in descs/:
  src_<tag>.txt:     the EKF (L-GSF with one component). It simulates the data of each trial.
  filters_<tag>.txt: the other filters, on the stored data of the EKF run (data block). Thus the comparison is paired.
    - UKF (U-GSF with one component, alpha = 1, beta = 0, kappa = 0)
    - L-AGSF, M = 10, N = L = 5, the same rule in both steps: prop 0.9, auto 0.4 (Eq. (47), r = 2/N), sdp 0.4
      (program (45)), and prop 0.001 (Delta close to 0: the Monte Carlo limit of the AGSF)
    - U-AGSF, M = 10, N = L = 5, alpha = 1, beta = 0, kappa = 0: prop 0.9 and auto 0.4 (task A10, D-42; the fair
      comparison with the UKF)
    - BPF with 1000 particles, ess 0.5 (a Monte Carlo reference)
<tag> is d<d> (rot = 0) or d<d>_rot (rot = 1).

Usage (repository root, jaxenv):  python scratch/sinlin_showcase/make_descs.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gaussfiltax'))
import experiment_descriptor as ed
from create_exps import create_descs

DIMS = [5, 10, 20, 50]           # rot = 0
ROT_DIMS = [10, 20]              # rot = 1 (the variant with a random unit direction a)
T, KAPPA, OMEGA = 100, 1.0, 2.0
TRIALS = 100                     # trials of the data source (the data block repeats them after this number)
M, N = 10, 5
RULES = [('prop', 0.9), ('auto', 0.4), ('sdp', 0.4), ('prop', 0.001)]
U_RULES = [('prop', 0.9), ('auto', 0.4)]   # U-AGSF (A10). No sdp: it doubles the longest runs (d = 50).
SRC = 'gsf/lgsf/ncomps_1_iters_1'


def tag(d, rot):
    return f'd{d}' + ('_rot' if rot else '')


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'descs'), exist_ok=True)
    for rot, dims in ((0, DIMS), (1, ROT_DIMS)):
        for d in dims:
            sim_desc = ed.SimulatorDescriptor.get_descriptor('sinlin').create_desc(d, d, T, KAPPA, OMEGA, rot)
            src = create_descs(sim_desc=sim_desc, alg='lgsf', scan_params={'n_comps': [1], 'n_iters': [1]})[0]
            data_desc = ed.DataDescriptor().create_desc(SRC, TRIALS)
            text = create_descs(sim_desc=sim_desc, alg='ugsf', data_desc=data_desc,
                                scan_params={'n_comps': [1], 'n_iters': [1], 'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0]})[0]
            for ac, r in RULES:
                text += create_descs(sim_desc=sim_desc, alg='lagsf', data_desc=data_desc,
                                     scan_params={'M': [M], 'N': [N], 'L': [N], 'n_iters': [1],
                                                  'ac1': [ac], 'r1': [r], 'ac2': [ac], 'r2': [r]})[0]
            for ac, r in U_RULES:
                text += create_descs(sim_desc=sim_desc, alg='uagsf', data_desc=data_desc,
                                     scan_params={'M': [M], 'N': [N], 'L': [N], 'n_iters': [1], 'alpha': [1.0], 'beta': [0.0],
                                                  'kappa': [0.0], 'ac1': [ac], 'r1': [r], 'ac2': [ac], 'r2': [r]})[0]
            text += create_descs(sim_desc=sim_desc, alg='bpf', data_desc=data_desc, scan_params={'num_prt': [1000], 'ess': [0.5]})[0]
            for name, body in (('src', src), ('filters', text)):
                path = os.path.join(HERE, 'descs', f'{name}_{tag(d, rot)}.txt')
                open(path, 'w').write(body)
                print(path, len(ed.parse(body)), 'experiments')
