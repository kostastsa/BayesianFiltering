"""Task A6 (D-35), side check (CLAUDE.md rule 5): paired comparison of a fixed rho2 with the adaptive rho2 in Exp. B (lmsv).

Each fixed-rho2 run (descriptor files lmsv_{lagsf,uagsf}_rho2sweep_scan_num_prt.txt) uses the data of the same trial of the
adaptive Table II row with M = 100 (data block). Thus trial t of the two rows has the same states and observations.
The values are the per-trial metrics that 'main.py errors' saved (mse.pkl, lpe.pkl, from util/numerics.py).
The reported rows of the table come from 'main.py table'. This script only adds the paired statistics:
- median over the trials of (fixed - adaptive), for the MSE and the LPE;
- the number of trials in which the fixed rho2 has the lower value;
- the two-sided Wilcoxon signed-rank p value.
It also prints the mean MSE over the linear steps (u = 0) and the MSV steps (u = 1), from the MSE over time in mse.pkl,
and the mean adaptive rho2 in the linear (u = 0) and the MSV (u = 1) steps, from aux_outputs.pkl (trials 1 to 20 have it).

Usage (repository root, jaxenv):  python scratch/a6_rho2/paired.py [-o scratch/a6_rho2/out/paired.md]
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
import numpy as onp
from scipy.stats import wilcoxon
import util.io

SIM = os.path.join(ROOT, 'data.nosync', 'experiments', 'lmsv', 'state-dim_4_emission-dim_4_num-timesteps_200')
ADAPT = 'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4'
RHO2 = [0.1, 0.3, 0.5, 0.7, 0.9]
N = 100
T = 200


def load(folder, metric):
    return onp.array([float(util.io.load(os.path.join(folder, str(t), metric))[0]) for t in range(1, N + 1)])


lines = []
for f, name in (('lagsf', 'L-AGSF'), ('uagsf', 'U-AGSF')):
    ref = os.path.join(SIM, 'agsf', f, ADAPT)
    ref_m = {k: load(ref, k) for k in ('mse', 'lpe')}
    lines += [f'### {name}, M = 100: fixed rho2 against the adaptive rho2 (same {N} data sets)', '',
              f'Adaptive: median MSE {onp.median(ref_m["mse"]):.2f}, median LPE {onp.median(ref_m["lpe"]):.3f}.', '',
              '| rho2 | Median MSE | Median (MSE fixed - adaptive) | Trials fixed < adaptive (MSE) | p (MSE) '
              '| Median LPE | Median (LPE fixed - adaptive) | Trials fixed < adaptive (LPE) | p (LPE) |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in RHO2:
        d = os.path.join(SIM, f'data_agsf_{f}_{ADAPT}', 'agsf', f, f'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_{r}')
        row = [f'{r}']
        for k in ('mse', 'lpe'):
            x = load(d, k)
            diff = x - ref_m[k]
            row += [f'{onp.median(x):.3f}', f'{onp.median(diff):+.3f}', f'{int((diff < 0).sum())} of {N}', f'{wilcoxon(diff).pvalue:.2g}']
        lines.append('| ' + ' | '.join(row) + ' |')
    # mean MSE in the two regimes (input switches every 20 steps, u_1 = 0), mean over the trials of the MSE over time
    u = (onp.arange(T) // 20) % 2
    per_t = lambda folder: onp.array([onp.asarray(util.io.load(os.path.join(folder, str(t), 'mse'))[1]) for t in range(1, N + 1)])
    rows = [('adaptive', per_t(ref))] + [(f'{r}', per_t(os.path.join(SIM, f'data_agsf_{f}_{ADAPT}', 'agsf', f,
                                                                   f'M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_{r}'))) for r in RHO2]
    lines += ['', '| rho2 | Mean MSE, linear steps (u = 0) | Mean MSE, MSV steps (u = 1) |', '|---|---|---|']
    lines += [f'| {n} | {x[:, u == 0].mean():.3f} | {x[:, u == 1].mean():.3f} |' for n, x in rows]
    rho = []
    for t in range(1, N + 1):
        p = os.path.join(ref, str(t), 'aux_outputs.pkl')
        if os.path.exists(p):
            rho.append(onp.asarray(util.io.load(p[:-4])['rho2s']).reshape(T, -1).mean(axis=1))
    if rho:
        rho = onp.array(rho)
        lines += ['', f'Adaptive rho2 ({len(rho)} trials with aux_outputs.pkl): mean {rho[:, u == 0].mean():.3f} in the linear steps (u = 0), '
                      f'{rho[:, u == 1].mean():.3f} in the MSV steps (u = 1).']
    lines.append('')

text = '\n'.join(lines)
print(text)
if len(sys.argv) > 2 and sys.argv[1] == '-o':
    os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
    open(sys.argv[2], 'w').write(text)
