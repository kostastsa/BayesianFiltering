"""Pilot of the adaptive AGSF in Exp. A (see make_descs.py): paired comparison with the Table I row on the same trials.

For each sigma2, filter and setting, over trials 1 to N (the same data sets as trials 1 to N of the reference row):
- mean and median of the MSE, the number of lost tracks (MSE > 10, as in Error summary.txt), the mean LPE and the mean NEES
  value (fraction of steps outside the 99.9 % region), from the mse.pkl, lpe.pkl and nees.pkl that 'main.py errors' saved;
- the number of trials in which the setting has a lower MSE than the reference.
The reference is the Table I row of the same filter with M = 100 and prop 0.9 / prop 0.9.

Usage (repository root, jaxenv):  python scratch/agsf_adaptive_expA/compare.py [N]     (default 12; writes out/compare.md)
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
import numpy as onp
import util.io

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
A, SIGMAS = '0.5', ['2.5', '0.025', '2.5e-05']
REF = 'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9'
SETTINGS = [('prop', 0.9, 'auto', 0.2), ('prop', 0.9, 'auto', 0.4), ('prop', 0.9, 'auto', 0.8), ('auto', 0.4, 'auto', 0.4)]
NAME = {'lagsf': 'L-AGSF', 'uagsf': 'U-AGSF'}


def metrics(folder):
    get = lambda k: onp.array([float(util.io.load(os.path.join(folder, str(t), k))[0]) for t in range(1, N + 1)])
    return {k: get(k) for k in ('mse', 'lpe', 'nees')}


lines = [f'Exp. A (mtt, wrapped bearing), a = {A}, M = 100, N = L = 5. Trials 1 to {N} of the Table I data sets (paired). '
         f'Lost: MSE > 10. NEES: mean fraction of steps outside the 99.9 % region.', '']
for s in SIGMAS:
    sim = os.path.join(ROOT, 'data.nosync', 'experiments', 'mtt', f'state-dim_4_emission-dim_2_num-timesteps_500_a_{A}_sigma2_{s}_wrap')
    lines += [f'### sigma2 = {s}', '', '| Filter | Rules (prediction / update) | Mean MSE | Median MSE | Lost | Mean LPE | NEES | MSE below the reference |',
              '|---|---|---|---|---|---|---|---|']
    for f in ('lagsf', 'uagsf'):
        ref = metrics(os.path.join(sim, 'agsf', f, REF))
        rows = [('prop 0.9 / prop 0.9 (Table I)', ref, None)]
        for ac1, r1, ac2, r2 in SETTINGS:
            d = os.path.join(sim, f'data_agsf_{f}_{REF}', 'agsf', f, f'M_100_N_5_L_5_iters_1_ac1_{ac1}_r1_{r1}_ac2_{ac2}_r2_{r2}')
            m = metrics(d)
            rows.append((f'{ac1} {r1} / {ac2} {r2}', m, int((m['mse'] < ref['mse']).sum())))
        for lab, m, below in rows:
            lines.append(f"| {NAME[f]} | {lab} | {m['mse'].mean():.3f} | {onp.median(m['mse']):.3f} | {int((m['mse'] > 10).sum())} of {N} "
                         f"| {m['lpe'].mean():.2f} | {100 * m['nees'].mean():.1f} % | {'-' if below is None else f'{below} of {N}'} |")
    lines.append('')

text = '\n'.join(lines)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
open(os.path.join(HERE, 'out', 'compare.md'), 'w').write(text + '\n')
print(text)
