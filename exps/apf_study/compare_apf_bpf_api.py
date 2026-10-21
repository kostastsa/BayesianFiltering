"""APF against BPF in the runs of Tables I and III (30 trials for each setting), from the per-trial MSE that 'main.py errors' saved.

Usage (repository root):  python exps/apf_study/compare_apf_bpf_api.py
The BPF and the APF see different data (main.py gives every descriptor its own random key), so the tests are for two independent samples:
  - Mann-Whitney U test (two-sided) on the MSE of the 30 trials,
  - Fisher exact test (two-sided) on the number of lost trials (MSE > 10).
Under the hypothesis that the two filters are the same, about 5% of the tests are below 0.05.
Output: apf_vs_bpf_api.tex (rows: median MSE and lost trials of each filter, with the two p-values) and a text summary.
"""
import os
import sys
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', 'data.nosync', 'experiments', 'mtt')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gaussfiltax'))
import util.io

SIGMA2 = [2.5, 0.025, 2.5e-5]
PARTICLES = [100, 1000, 10000, 100000]
SIG = {2.5: r'25\times10^{-1}', 0.025: r'25\times10^{-3}', 2.5e-5: r'25\times10^{-6}'}
LOST = 10.0
N_TRIALS = 30


def mse_trials(a, sigma2, flt, m, ess=0.5):
    base = os.path.join(ROOT, f'state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{sigma2}', 'pf', flt, f'numprt_{m}_ess_{ess}')
    return np.array([float(util.io.load(os.path.join(base, str(k), 'mse'))[0]) for k in range(1, N_TRIALS + 1)])


def pstr(p):
    return '<0.001' if p < 1e-3 else f'{p:.3f}'


tex, text = [], []
pooled = {}          # sigma2 group -> [lost bpf, lost apf, n each]
n_cells, n_sig_u, n_sig_f = {}, {}, {}
for a in (0.5, 0.05):
    for s in SIGMA2:
        for m in PARTICLES:
            b, p = mse_trials(a, s, 'bpf', m), mse_trials(a, s, 'apf', m)
            lb, lp = int((b > LOST).sum()), int((p > LOST).sum())
            pu = mannwhitneyu(b, p, alternative='two-sided').pvalue
            pf = fisher_exact([[lb, N_TRIALS - lb], [lp, N_TRIALS - lp]])[1]
            grp = 'high' if s >= 0.025 else 'low'
            t = pooled.setdefault((a, grp), [0, 0, 0]); t[0] += lb; t[1] += lp; t[2] += N_TRIALS
            n_cells[(a, grp)] = n_cells.get((a, grp), 0) + 1
            n_sig_u[(a, grp)] = n_sig_u.get((a, grp), 0) + (pu < 0.05)
            n_sig_f[(a, grp)] = n_sig_f.get((a, grp), 0) + (pf < 0.05)
            tex.append(f'${a:g}$ & ${SIG[s]}$ & {m} & {np.median(b):.2g} ({lb}) & {np.median(p):.2g} ({lp}) & {pstr(pu)} & {pstr(pf)} \\\\')
            text.append(f'a={a:g} sigma2={s:g} M={m:6d}: BPF median {np.median(b):.3g} lost {lb:2d} | APF median {np.median(p):.3g} lost {lp:2d} | MWU p={pstr(pu)} Fisher p={pstr(pf)}')

text.append('')
for (a, grp), (lb, lp, n) in sorted(pooled.items(), reverse=True):
    pf = fisher_exact([[lb, n - lb], [lp, n - lp]])[1]
    name = 'sigma2 in {2.5, 0.025}' if grp == 'high' else 'sigma2 = 2.5e-5'
    text.append(f'a={a:g} {name}: {n_cells[(a, grp)]} cells, tests below 0.05: Mann-Whitney {n_sig_u[(a, grp)]}, Fisher {n_sig_f[(a, grp)]}; pooled lost trials BPF {lb}/{n}, APF {lp}/{n}, Fisher p={pstr(pf)}')

lb = sum(v[0] for (a, g), v in pooled.items() if g == 'high'); lp = sum(v[1] for (a, g), v in pooled.items() if g == 'high'); n = sum(v[2] for (a, g), v in pooled.items() if g == 'high')
text.append(f'both a, sigma2 in {{2.5, 0.025}}: cells {sum(n_cells[k] for k in n_cells if k[1] == "high")}, tests below 0.05: Mann-Whitney {sum(n_sig_u[k] for k in n_sig_u if k[1] == "high")}, Fisher {sum(n_sig_f[k] for k in n_sig_f if k[1] == "high")}; pooled lost BPF {lb}/{n}, APF {lp}/{n}, Fisher p={pstr(fisher_exact([[lb, n - lb], [lp, n - lp]])[1])}')

open(os.path.join(HERE, 'apf_vs_bpf_api.tex'), 'w').write('\n'.join(tex) + '\n')
print('\n'.join(text))

# The APF that resamples the ancestors at every step (ess 2.0, the rule of the first version of the paper), M = 1000.
if os.path.isdir(os.path.join(ROOT, 'state-dim_4_emission-dim_2_num-timesteps_500_a_0.5_sigma2_2.5', 'pf', 'apf', 'numprt_1000_ess_2.0')):
    tex2, text2, lost_all = [], ['', 'M = 1000: APF with the ESS rule against the APF that resamples at every step (APF*)'], [0, 0, 0, 0]
    for a in (0.5, 0.05):
        for s in SIGMA2:
            b, p = mse_trials(a, s, 'bpf', 1000), mse_trials(a, s, 'apf', 1000)
            o = mse_trials(a, s, 'apf', 1000, ess=2.0)
            lb, lp, lo = int((b > LOST).sum()), int((p > LOST).sum()), int((o > LOST).sum())
            pu = mannwhitneyu(o, p, alternative='two-sided').pvalue
            pf = fisher_exact([[lo, N_TRIALS - lo], [lp, N_TRIALS - lp]])[1]
            lost_all = [lost_all[0] + lb, lost_all[1] + lp, lost_all[2] + lo, lost_all[3] + 1]
            tex2.append(f'${a:g}$ & ${SIG[s]}$ & {np.median(b):.2g} ({lb}) & {np.median(p):.2g} ({lp}) & {np.median(o):.2g} ({lo}) & {pstr(pu)} & {pstr(pf)} \\\\')
            text2.append(f'a={a:g} sigma2={s:g}: BPF median {np.median(b):.3g} lost {lb:2d} | APF {np.median(p):.3g} lost {lp:2d} | APF* {np.median(o):.3g} lost {lo:2d} | APF* vs APF: MWU p={pstr(pu)} Fisher p={pstr(pf)}')
    n = N_TRIALS * lost_all[3]
    pf = fisher_exact([[lost_all[2], n - lost_all[2]], [lost_all[1], n - lost_all[1]]])[1]
    text2.append(f'pooled over the six settings: lost BPF {lost_all[0]}/{n}, APF {lost_all[1]}/{n}, APF* {lost_all[2]}/{n}; APF* against APF, Fisher p={pstr(pf)}')
    open(os.path.join(HERE, 'apf_ess2_api.tex'), 'w').write('\n'.join(tex2) + '\n')
    print('\n'.join(text2))
