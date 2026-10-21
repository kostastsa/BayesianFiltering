"""BPF and APF rows of Tables I (a = 0.5) and III (a = 0.05) of the paper, from the files that 'main.py errors' writes.

Usage (repository root):  python exps/apf_study/tables_I_III.py
It reads data.nosync/experiments/mtt/<setting>/pf/{bpf,apf}/numprt_<M>_ess_0.5/Error summary.txt. It runs no filter and computes no metric.
Output: exps/apf_study/tables_I_III_rows.tex (the rows in the format of the paper) and a text summary (median MSE, lost trials) on the screen.
"""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data.nosync', 'experiments', 'mtt')
SIGMA2 = [2.5, 0.025, 2.5e-5]
PARTICLES = [100, 1000, 10000, 100000]
FILTERS = [('bpf', 'BPF'), ('apf', 'APF')]


def read_summary(a, sigma2, flt, m):
    path = os.path.join(ROOT, f'state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{sigma2}', 'pf', flt, f'numprt_{m}_ess_0.5', 'Error summary.txt')
    text = open(path).read()
    out = {'trials': int(re.search(r'trials: (\d+)', text).group(1)),
           'lost': int(re.search(r'trials with mse > 10: (\d+)', text).group(1)),
           'nonfinite': [int(x) for x in re.search(r'non-finite trials \(lpe, bias, minerr, mse\): (\d+), (\d+), (\d+), (\d+)', text).groups()]}

    for name in ('lpe', 'mse'):
        mean, sd, med = re.search(rf'^{name}: (\S+) \+- (\S+), (\S+)$', text, re.M).groups()
        out[name] = (float(mean), float(sd), float(med))

    return out


def num(x):
    """The number format of the paper: 2 decimals, or a x 10^n when the number is below 0.01 or above 999."""
    if x != x:
        return r'\textsc{nan}'
    if x == 0 or 0.01 <= abs(x) < 1000:
        return f'{x:.2f}'
    mant, exp = f'{x:.2e}'.split('e')
    return f'{mant}\\times 10^{{{int(exp)}}}'


def cell(mean, sd):
    return f'${num(mean)}\\pm{num(sd)}$'


rows, text = [], []
for a, table in ((0.5, 'I'), (0.05, 'III')):
    rows.append(f'% Table {table}, a = {a}')
    text.append(f'Table {table} (a = {a}): median MSE (trials with MSE > 10, non-finite trials of MSE)  [trials]')

    for flt, label in FILTERS:
        for i, m in enumerate(PARTICLES):
            head = f'\\multirow{{{len(PARTICLES)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{label}}}}} & ' if i == 0 else '  & '
            cells, tcells = [], []

            for s in SIGMA2:
                r = read_summary(a, s, flt, m)
                cells.append(f'{cell(*r["mse"][:2])} & {cell(*r["lpe"][:2])}')
                tcells.append(f'{r["mse"][2]:.3g} ({r["lost"]}, {r["nonfinite"][3]}) [{r["trials"]}]')

            rows.append(head + f'$M={m}$ & ' + ' & '.join(cells) + r' \\')
            text.append(f'  {label} M={m:6d}: ' + '   '.join(tcells))

        if flt != FILTERS[-1][0]:
            rows.append(r'\midrule')

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables_I_III_rows.tex'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(text))
