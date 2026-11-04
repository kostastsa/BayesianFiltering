"""Showcase of plan step 6 (see make_descs.py): paired comparison of the filters on the sinlin data.

For each setting (d, rot) and filter, over trials 1 to K (the same data sets for all filters):
- RMSE of x_1 and of x_2..x_d (per coordinate): the square root of the mean over trials of mse_x1.pkl and mse_rest.pkl;
- the mean LPE (lpe.pkl) and the mean NEES value (nees.pkl: fraction of steps outside the 99.9 % region);
- the mean run time (runtime.txt, with the JIT compile time);
- AGSF only: the eigenvalues of Sigma^-1 Lambda in the update (aux_outputs.pkl, eig2s). 'lam min' and 'lam max' are the
  means over trials, steps and components of the smallest and the largest eigenvalue.
- the number of trials in which L-AGSF sdp has a lower mse_x1 than L-AGSF auto.
All metric files come from 'main.py errors'. This script only averages them.

Usage (repository root, jaxenv):  python scratch/sinlin_showcase/compare.py [K]
Writes out/compare.md and out/summary.json (the numbers of the page).
"""
import json
import os
import re
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
sys.path.insert(0, HERE)
import numpy as onp
import util.io
from make_descs import DIMS, ROT_DIMS, T, KAPPA, OMEGA, M, N, SRC

K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
EXP = os.path.join(ROOT, 'data.nosync', 'experiments', 'sinlin')
AG = f'agsf/lagsf/M_{M}_N_{N}_L_{N}_iters_1_ac1_{{0}}_r1_{{1}}_ac2_{{0}}_r2_{{1}}'
UAG = AG.replace('agsf/lagsf', 'agsf/uagsf')
FILTERS = [  # (key, label, folder relative to the sim folder; None: the data source)
    ('ekf', 'EKF (L-GSF, 1 comp.)', SRC),
    ('ukf', 'UKF (U-GSF, 1 comp.)', 'D/gsf/ugsf/ncomps_1_iters_1_alpha_1.0_beta_0.0_kappa_0.0'),
    ('prop', 'L-AGSF prop 0.9', 'D/' + AG.format('prop', 0.9)),
    ('auto', 'L-AGSF auto 2/N (Eq. 47)', 'D/' + AG.format('auto', 0.4)),
    ('sdp', 'L-AGSF sdp 2/N (program 45)', 'D/' + AG.format('sdp', 0.4)),
    ('mc', 'L-AGSF prop 0.001 (Delta near 0)', 'D/' + AG.format('prop', 0.001)),
    ('uprop', 'U-AGSF prop 0.9', 'D/' + UAG.format('prop', 0.9)),
    ('uauto', 'U-AGSF auto 2/N (Eq. 47)', 'D/' + UAG.format('auto', 0.4)),
    ('bpf', 'BPF, 1000 particles', 'D/pf/bpf/numprt_1000_ess_0.5'),
]


def sim_dir(d, rot):
    return os.path.join(EXP, f'state-dim_{d}_emission-dim_{d}_num-timesteps_{T}_kappa_{KAPPA}_omega_{OMEGA}' + ('_rot' if rot else ''))


def load(folder, agsf):
    get = lambda k: onp.array([float(util.io.load(os.path.join(folder, str(t), k))[0]) for t in range(1, K + 1)])
    out = {k: get(k) for k in ('mse_x1', 'mse_rest', 'lpe', 'nees')}
    out['runtime'] = onp.array([float(re.search(r'([\d.]+)', open(os.path.join(folder, str(t), 'runtime.txt')).read()).group(1))
                                for t in range(1, K + 1)])
    if agsf:
        # eig2_mm.pkl (shrink.py): the means over the components of the smallest and largest eigenvalue, each (T,).
        # Without it, read eig2s (T, MN, d) of aux_outputs.pkl.
        def mm(t):
            p = os.path.join(folder, str(t))
            if os.path.exists(os.path.join(p, 'eig2_mm.pkl')):
                return util.io.load(os.path.join(p, 'eig2_mm'))
            e = onp.asarray(util.io.load(os.path.join(p, 'aux_outputs'))['eig2s'])
            return e[..., 0].mean(axis=1), e[..., -1].mean(axis=1)
        lo, hi = zip(*[mm(t) for t in range(1, K + 1)])
        out['lam_min'] = float(onp.mean(lo))
        out['lam_max'] = float(onp.mean(hi))
        out['lam_min_t'] = onp.mean(lo, axis=0).tolist()   # mean over trials and components
        out['lam_max_t'] = onp.mean(hi, axis=0).tolist()
    return out


settings = [(d, 0) for d in DIMS] + [(d, 1) for d in ROT_DIMS]
summary = {'K': K, 'settings': []}
lines = [f'sinlin showcase: T = {T}, kappa = {KAPPA}, omega = {OMEGA}, L-AGSF M = {M}, N = L = {N}. Trials 1 to {K}, the same data '
         'for all filters (data block). RMSE: square root of the mean over trials. "sdp below auto": trials in which sdp has the '
         'lower mse_x1.', '']
for d, rot in settings:
    sd = sim_dir(d, rot)
    data_dir = os.path.join(sd, 'data_' + SRC.replace('/', '_'))
    rows, res = [], {}
    for key, lab, rel in FILTERS:
        folder = os.path.join(sd, rel) if not rel.startswith('D/') else os.path.join(data_dir, rel[2:])
        try:
            res[key] = load(folder, key in ('prop', 'auto', 'sdp', 'mc', 'uprop', 'uauto'))
        except FileNotFoundError:
            print(f'missing: d = {d}, rot = {rot}, {key} (trials 1 to {K} not complete)')
            continue
    entry = {'d': d, 'rot': rot, 'filters': []}
    lines += [f'### d = {d}' + (', rotated direction a' if rot else ', a = e_1'), '',
              '| Filter | RMSE x_1 | RMSE x_2..x_d | Mean LPE | NEES outside | lam min | lam max | Run time (s) |', '|---|---|---|---|---|---|---|---|']
    for key, lab, _ in FILTERS:
        if key not in res:
            continue
        m = res[key]
        f = {'key': key, 'label': lab, 'rmse_x1': float(onp.sqrt(m['mse_x1'].mean())), 'rmse_rest': float(onp.sqrt(m['mse_rest'].mean())),
             'med_x1': float(onp.sqrt(onp.median(m['mse_x1']))), 'lpe': float(m['lpe'].mean()), 'lpe_med': float(onp.median(m['lpe'])),
             'nees': float(m['nees'].mean()), 'runtime': float(m['runtime'].mean()),
             'mse_x1_trials': m['mse_x1'].tolist()}
        for k in ('lam_min', 'lam_max', 'lam_min_t', 'lam_max_t'):
            if k in m:
                f[k] = m[k]
        entry['filters'].append(f)
        lam = f"{f['lam_min']:.3f} | {f['lam_max']:.3f}" if 'lam_min' in f else '- | -'
        lines.append(f"| {lab} | {f['rmse_x1']:.3f} | {f['rmse_rest']:.3f} | {f['lpe']:.4g} | {100 * f['nees']:.1f} % | {lam} | {f['runtime']:.1f} |")
    if 'sdp' in res and 'auto' in res:
        entry['sdp_below_auto'] = int((res['sdp']['mse_x1'] < res['auto']['mse_x1']).sum())
        lines += ['', f"sdp below auto (mse_x1): {entry['sdp_below_auto']} of {K}"]
    lines.append('')
    summary['settings'].append(entry)

os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
open(os.path.join(HERE, 'out', 'compare.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(summary, open(os.path.join(HERE, 'out', 'summary.json'), 'w'))
print('\n'.join(lines))
