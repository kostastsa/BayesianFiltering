"""Pilot (side question, CLAUDE.md rule 5): the adaptive AGSF in Exp. A (mtt, wrapped bearing, a = 0.5, Table I).

Writes one descriptor file for each sigma2 in this folder (descs/pilot_sigma2_<sigma2>.txt), with create_exps.create_descs.
Each file has the L-AGSF and the U-AGSF with M = 100, N = L = 5, and four sparse settings of the split rules:
  prop 0.9 / auto 0.2,  prop 0.9 / auto 0.4 (the adaptive variant of Sec. V-B),  prop 0.9 / auto 0.8,  auto 0.4 / auto 0.4.
'auto r' is the automatic rule of filters/agsf.py: rho = min(0.95, r Tr(P J^T J) / sum_i Tr(H_i P)^2).
Each run uses the stored data of the same trial of the Table I row of the same filter with M = 100 and prop 0.9 / prop 0.9
(data block, the reference). Thus the comparison with the table row is paired.

Usage (repository root, jaxenv):  python scratch/agsf_adaptive_expA/make_descs.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gaussfiltax'))
import experiment_descriptor as ed
from create_exps import create_descs

A, SIGMAS = 0.5, [2.5, 0.025, 2.5e-5]
REF = 'agsf/{}/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9'
REF_TRIALS = 100
SETTINGS = [('prop', 0.9, 'auto', 0.2), ('prop', 0.9, 'auto', 0.4), ('prop', 0.9, 'auto', 0.8), ('auto', 0.4, 'auto', 0.4)]
UKF = {'alpha': [1.0], 'beta': [0.0], 'kappa': [0.0]}

os.makedirs(os.path.join(HERE, 'descs'), exist_ok=True)
sim = ed.SimulatorDescriptor().get_descriptor('mtt')
for s in SIGMAS:
    sim_desc = sim.create_desc(4, 2, 500, A, s, wrap=True)
    text = ''
    for alg in ('lagsf', 'uagsf'):
        data_desc = ed.DataDescriptor().create_desc(REF.format(alg), REF_TRIALS)
        for ac1, r1, ac2, r2 in SETTINGS:
            params = {'M': [100], 'N': [5], 'L': [5], 'n_iters': [1], **(UKF if alg == 'uagsf' else {}),
                      'ac1': [ac1], 'r1': [r1], 'ac2': [ac2], 'r2': [r2]}
            text += create_descs(sim_desc=sim_desc, alg=alg, scan_params=params, data_desc=data_desc)[0]
    path = os.path.join(HERE, 'descs', f'pilot_sigma2_{s}.txt')
    open(path, 'w').write(text)
    print(path, len(ed.parse(text)), 'experiments')
