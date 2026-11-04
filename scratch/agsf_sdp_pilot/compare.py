"""Pilot A8 (D-39, see make_descs.py): paired comparison of the rules prop 0.9, auto 2/N and sdp 2/N.

For each experiment, N and rule, over trials 1 to K (the same data sets as trials 1 to K of the reference row):
- mean and median of the MSE, the number of lost tracks (MSE > 10, as in Error summary.txt) and the mean LPE,
  from the mse.pkl and lpe.pkl that 'main.py errors' saved;
- the mean rho1 (prediction) and rho2 (update) over trials, steps and components, from aux_outputs.pkl;
- the number of trials in which the rule has a lower MSE (LPE) than prop 0.9 with the same N.
The first row of each table is the reference row (M = 100, the data source), for scale.

Usage (repository root, jaxenv):  python scratch/agsf_sdp_pilot/compare.py [K]     (default 5; writes out/compare.md)
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
sys.path.insert(0, HERE)
import numpy as onp
import util.io
from make_descs import M, NS, rules

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EXP = ROOT + '/data.nosync/experiments/'
SIMS = {'expA': ('Exp. A (mtt, wrapped bearing, a = 0.5, sigma2 = 0.025)', 'mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_0.5_sigma2_0.025_wrap',
                 'agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9'),
        'expB': ('Exp. B (lmsv)', 'lmsv/state-dim_4_emission-dim_4_num-timesteps_200',
                 'agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4')}


def load_metrics(folder, aux=True):
    get = lambda k: onp.array([float(util.io.load(os.path.join(folder, str(t), k))[0]) for t in range(1, K + 1)])
    out = {k: get(k) for k in ('mse', 'lpe')}
    if aux:
        a = [util.io.load(os.path.join(folder, str(t), 'aux_outputs')) for t in range(1, K + 1)]
        out['rho1'] = onp.mean([onp.mean(x['rho1s']) for x in a])
        out['rho2'] = onp.mean([onp.mean(x['rho2s']) for x in a])
    return out


def fmt(lab, m, base=None):
    rho = f"{m['rho1']:.3f} | {m['rho2']:.3f}" if 'rho1' in m else '- | -'
    below = '- | -' if base is None else f"{int((m['mse'] < base['mse']).sum())} of {K} | {int((m['lpe'] < base['lpe']).sum())} of {K}"
    return (f"| {lab} | {m['mse'].mean():.4g} | {onp.median(m['mse']):.4g} | {int((m['mse'] > 10).sum())} | "
            f"{m['lpe'].mean():.3f} | {rho} | {below} |")


lines = [f'Pilot A8: L-AGSF, M = {M}, the same rule in both steps. Trials 1 to {K} of the stored data of the reference row (paired). '
         'Lost: MSE > 10. "Below prop": number of trials with a lower value than prop 0.9 at the same N.', '']
for key, (title, sim, ref) in SIMS.items():
    data = os.path.join(EXP, sim, 'data_' + ref.replace('/', '_'), 'agsf', 'lagsf')
    lines += [f'### {title}', '', '| N = L | Rule | Mean MSE | Median MSE | Lost | Mean LPE | rho1 | rho2 | MSE below prop | LPE below prop |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines.append('| 5 | ' + fmt('reference row, M = 100', load_metrics(os.path.join(EXP, sim, ref), aux=False))[2:])
    for n in NS:
        base = None
        for ac, r in rules(n):
            m = load_metrics(os.path.join(data, f'M_{M}_N_{n}_L_{n}_iters_1_ac1_{ac}_r1_{r}_ac2_{ac}_r2_{r}'))
            lines.append(f'| {n} | ' + fmt(f'{ac} {r}', m, base)[2:])
            base = m if ac == 'prop' else base
    lines.append('')

text = '\n'.join(lines)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
open(os.path.join(HERE, 'out', 'compare.md'), 'w').write(text + '\n')
print(text)
