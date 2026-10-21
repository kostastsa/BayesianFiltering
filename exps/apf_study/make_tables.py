"""Tables of the APF study, from the result files in exps/apf_study/out. Run it in that folder:  python3 ../make_tables.py

Writes apf_tables.tex (LaTeX rows for the supplementary material) and prints the same numbers as text.
Table 1  (our code, M = 1000, 100 data sets): MSE and runs that lose track, BPF, APF and APF with the old rule; paired APF/BPF.
Table 2  (diagnostics, M = 1000, 60 data sets): error of the first-stage weight, ESS of the two filters, fraction of resampling steps.
Table 3  (reference code, M = 1000, 30 data sets x 12 runs): our filters against the particles library on the same data.
"""
import math
import numpy as np

SETTINGS = [(0.5, 2.5), (0.5, 0.025), (0.5, 2.5e-5), (0.05, 2.5), (0.05, 0.025), (0.05, 2.5e-5)]
SIG = {2.5: r'25\times10^{-1}', 0.025: r'25\times10^{-3}', 2.5e-5: r'25\times10^{-6}'}
LOST = 10.0   # an MSE above this value: the filter has lost the target


def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sci(x):
    return f'{x:.2g}'


def pstr(p):
    return '<0.001' if p < 1e-3 else f'{p:.3f}'


tex, txt = [], []

# ---- Table 1
tex.append(r'% Table 1: ours, M = 1000')
txt.append('Table 1  M = 1000: median MSE (runs with MSE > 10 or NaN) of BPF, APF, APF with the old rule, and the paired APF/BPF')
for a, r in SETTINGS:
    d = np.load(f'rerun_a{a:g}_r{r:g}_M1000.npz')
    b, p, o = d['bpf'], d['apf'], d['apf_every']
    n = len(b)
    lost = lambda x: int(((x > LOST) | ~np.isfinite(x)).sum())
    ok = np.isfinite(b) & np.isfinite(p)
    ratio = np.exp(np.median(np.log(p[ok] / b[ok])))
    wins = int((p[ok] < b[ok]).sum())
    pv = sign_p(wins, int(ok.sum()))
    tex.append(f'${a:g}$ & ${SIG[r]}$ & {sci(np.nanmedian(b))} ({lost(b)}) & {sci(np.nanmedian(p))} ({lost(p)}) & {sci(np.nanmedian(o))} ({lost(o)}) & {ratio:.2f} & {pstr(pv)} \\\\')
    txt.append(f'  a={a:g} sigma2={r:g}: BPF {np.nanmedian(b):.3g} ({lost(b)}/{n})  APF {np.nanmedian(p):.3g} ({lost(p)}/{n})  APF old rule {np.nanmedian(o):.3g} ({lost(o)}/{n})  '
               f'ratio {ratio:.2f}  wins {wins}/{int(ok.sum())}  p={pstr(pv)}')

# ---- Table 2
tex.append(r'% Table 2: diagnostics')
txt.append('Table 2  M = 1000, 60 data sets: sd of the log error of the first-stage weight; ESS/M of the first-stage weights (APF) and of the BPF weights; fraction of steps that resample')
for a, r in SETTINGS:
    sv = np.load(f'apf_regime_M1000_n60_T500_a{a:g}.npy', allow_pickle=True).item()[(r, 1e-6)]
    sd_d, sd_ll, e1, e2, rb, ra = np.median(sv['diag'], axis=0)
    tex.append(f'${a:g}$ & ${SIG[r]}$ & {sci(sd_d)} & {e1:.2f} & {e2:.2f} & {rb:.2f} & {ra:.2f} \\\\')
    txt.append(f'  a={a:g} sigma2={r:g}: sd_d {sd_d:.3g}  ess1 {e1:.2f}  ess2 {e2:.2f}  resample BPF {rb:.2f} APF {ra:.2f}')

# ---- Table 3
tex.append(r'% Table 3: ours against the particles library')
txt.append('Table 3  ours against the particles library (same data, M = 1000, 30 data sets x 12 runs): geometric mean of MSE ours / particles (sign-test p) and median APF/BPF')
for a, r in SETTINGS:
    suf = f'_a{a:g}' if a != 0.05 or True else ''
    try:
        o, p = np.load(f'cmp_data_q1e-06_r{r:g}{suf}.npz'), np.load(f'cmp_particles_q1e-06_r{r:g}{suf}.npz')
    except FileNotFoundError:
        continue
    cells = []
    for f in ('bpf', 'apf'):
        mo, mp = o[f'mse_{f}'], p[f'mse_{f}']
        so = np.nanmean(np.log(mo), axis=1)
        sp = np.nanmean(np.log(np.where(np.isfinite(mp), mp, np.nan)), axis=1)
        ok = np.isfinite(so) & np.isfinite(sp)
        wins = int((so[ok] < sp[ok]).sum())
        cells.append((np.exp(np.mean(so[ok] - sp[ok])), sign_p(wins, int(ok.sum()))))
    rat_o = np.nanmedian(o['mse_apf']) / np.nanmedian(o['mse_bpf'])
    rat_p = np.nanmedian(p['mse_apf']) / np.nanmedian(p['mse_bpf'])
    tex.append(f'${a:g}$ & ${SIG[r]}$ & {cells[0][0]:.2f} ({pstr(cells[0][1])}) & {cells[1][0]:.2f} ({pstr(cells[1][1])}) & {rat_o:.2f} & {rat_p:.2f} \\\\')
    txt.append(f'  a={a:g} sigma2={r:g}: BPF {cells[0][0]:.2f} (p={pstr(cells[0][1])})  APF {cells[1][0]:.2f} (p={pstr(cells[1][1])})  median APF/BPF: ours {rat_o:.2f}, particles {rat_p:.2f}')

open('apf_tables.tex', 'w').write('\n'.join(tex) + '\n')
print('\n'.join(txt))
