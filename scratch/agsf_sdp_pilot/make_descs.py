"""Pilot A8 (D-39, side question, CLAUDE.md rule 5): the split rules 'prop', 'auto' and 'sdp' for different N.

Writes two descriptor files in this folder (descs/pilot_expA.txt, descs/pilot_expB.txt), with create_exps.create_descs.
Each file has the L-AGSF with M = 10, N = L in {5, 20, 100}, and three rules, the same rule in both steps:
  prop 0.9,  auto 2/N (Eq. (47)),  sdp 2/N (program (45)).
- Exp. A: mtt, wrapped bearing, a = 0.5, sigma2 = 0.025. Data of the Table I row (L-AGSF, M = 100, prop 0.9 / prop 0.9).
- Exp. B: lmsv. Data of the Table II row (L-AGSF, M = 100, prop 0.9 / auto 0.4).
Each run uses the stored data of the same trial of the reference row (data block). Thus the comparison is paired.

Usage (repository root, jaxenv):  python scratch/agsf_sdp_pilot/make_descs.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gaussfiltax'))
import experiment_descriptor as ed
from create_exps import create_descs

M, NS = 10, [5, 20, 100]
REF_TRIALS = 100
EXPS = {
    'expA': (('mtt', (4, 2, 500, 0.5, 0.025), {'wrap': True}), 'agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9'),
    'expB': (('lmsv', (4, 4, 200), {}), 'agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4'),
}


def rules(n):
    return [('prop', 0.9), ('auto', round(2 / n, 6)), ('sdp', round(2 / n, 6))]


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'descs'), exist_ok=True)
    for name, ((sim_name, args, kw), ref) in EXPS.items():
        sim_desc = ed.SimulatorDescriptor().get_descriptor(sim_name).create_desc(*args, **kw)
        data_desc = ed.DataDescriptor().create_desc(ref, REF_TRIALS)
        text = ''
        for n in NS:
            for ac, r in rules(n):
                params = {'M': [M], 'N': [n], 'L': [n], 'n_iters': [1], 'ac1': [ac], 'r1': [r], 'ac2': [ac], 'r2': [r]}
                text += create_descs(sim_desc=sim_desc, alg='lagsf', scan_params=params, data_desc=data_desc)[0]
        path = os.path.join(HERE, 'descs', f'pilot_{name}.txt')
        open(path, 'w').write(text)
        print(path, len(ed.parse(text)), 'experiments')
