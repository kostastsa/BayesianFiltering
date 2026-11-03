"""Task A6/A7, side check (CLAUDE.md rule 5): the curves of Fig. 3 (Exp. B) for the A6 page, as JSON.

- MSE over time: the mean over the 100 trials of the MSE over time that 'main.py errors' saved (mse.pkl, util/numerics.mse),
  for the Fig. 3 filters (L-GSF and U-GSF with M = 1000, BPF with M = 100, L-AGSF and U-AGSF with M = 100 and the adaptive rho2)
  and for the fixed-rho2 runs of A6.
- rho2 over time: the mean over the components and over trials 1 to 20 (aux_outputs.pkl) of the adaptive rho2, with the
  10 % and 90 % quantiles over these trials.

Usage (repository root, jaxenv):  python scratch/a6_rho2/export_fig3.py   (writes scratch/a6_rho2/out/fig3.json)
"""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
import numpy as onp
import util.io

SIM = os.path.join(ROOT, 'data.nosync', 'experiments', 'lmsv', 'state-dim_4_emission-dim_4_num-timesteps_200')
ADAPT = 'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4'
N, T = 100, 200


def mse_t(folder):
    return onp.array([onp.asarray(util.io.load(os.path.join(folder, str(t), 'mse'))[1]) for t in range(1, N + 1)]).mean(axis=0)


out = {'mse': {}, 'rho2': {}}
out['mse']['L-GSF M=1000'] = mse_t(os.path.join(SIM, 'gsf', 'lgsf', 'ncomps_1000_iters_1'))
out['mse']['U-GSF M=1000'] = mse_t(os.path.join(SIM, 'gsf', 'ugsf', 'ncomps_1000_iters_1_alpha_1.0_beta_0.0_kappa_0.0'))
out['mse']['BPF M=100'] = mse_t(os.path.join(SIM, 'pf', 'bpf', 'numprt_100_ess_0.5'))
for f, name in (('lagsf', 'L-AGSF'), ('uagsf', 'U-AGSF')):
    ref = os.path.join(SIM, 'agsf', f, ADAPT)
    out['mse'][f'{name} adaptive'] = mse_t(ref)
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
        out['mse'][f'{name} {r}'] = mse_t(os.path.join(SIM, f'data_agsf_{f}_{ADAPT}', 'agsf', f,
                                                     f'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_{r}'))
    rho = []
    for t in range(1, N + 1):
        p = os.path.join(ref, str(t), 'aux_outputs')
        if os.path.exists(p + '.pkl'):
            rho.append(onp.asarray(util.io.load(p)['rho2s']).reshape(T, -1).mean(axis=1))
    rho = onp.array(rho)
    out['rho2'][name] = {'mean': rho.mean(axis=0), 'q10': onp.quantile(rho, 0.1, axis=0), 'q90': onp.quantile(rho, 0.9, axis=0),
                         'trials': len(rho)}

r4 = lambda a: [round(float(v), 4) for v in a]
js = {'mse': {k: r4(v) for k, v in out['mse'].items()},
      'rho2': {k: {kk: (r4(vv) if kk != 'trials' else vv) for kk, vv in v.items()} for k, v in out['rho2'].items()}}
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(js, open(os.path.join(HERE, 'out', 'fig3.json'), 'w'))
for k, v in js['mse'].items():
    print(f'{k:20s} mean {onp.mean(v):7.3f}  max {onp.max(v):7.3f}')
for k, v in js['rho2'].items():
    print(k, 'rho2 trials', v['trials'], 'min', min(v['mean']), 'max', max(v['mean']))
