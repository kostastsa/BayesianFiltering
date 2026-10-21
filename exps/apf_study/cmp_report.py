"""Compare our BPF/APF (pf.py) with the particles library on the same data. Reads cmp_data_*.npz and cmp_particles_*.npz."""
import math
import os
import numpy as np

SUF = os.environ.get('SUF', '')
CONFIGS = [tuple(float(v) for v in c.split(':')) for c in os.environ.get('CONFIGS', '1e-6:2.5,1e-4:0.025,1e-2:0.025').split(',')]
TAGS = [(f'q{q:g}_r{r:g}{SUF}', f'q={q:g}, r={r:g}{SUF}') for q, r in CONFIGS]


def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


for tag, title in TAGS:
    o, p = np.load(f'cmp_data_{tag}.npz'), np.load(f'cmp_particles_{tag}.npz')
    n = p['mse_apf'].shape[0]
    print(f'\n=== {title}: {n} data sets, {p["mse_apf"].shape[1]} runs each, M=1000 ===')
    print(f'{"filter":6s} | {"median MSE ours":>15s} {"particles":>10s} | {"geo. mean ratio":>15s} {"ours<part.":>10s} {"p":>6s} | '
          f'{"logL diff (ours-part.)":>24s} {"se":>6s} {"t":>5s} | {"sd logL ours":>12s} {"particles":>10s}')
    for f in ('bpf', 'apf'):
        mo, mp = o[f'mse_{f}'], p[f'mse_{f}']
        lo, lp = o[f'll_{f}'], p[f'll_{f}']
        mo_s, mp_s = np.nanmean(np.log(mo), axis=1), np.nanmean(np.log(np.where(np.isfinite(mp), mp, np.nan)), axis=1)
        ratio = np.exp(np.mean(mo_s - mp_s))
        wins = int((mo_s < mp_s).sum())
        dl = np.nanmean(lo, axis=1) - np.nanmean(np.where(np.isfinite(lp), lp, np.nan), axis=1)
        se = dl.std(ddof=1) / np.sqrt(n)
        sd_o = np.sqrt(np.nanmean(np.nanvar(lo, axis=1, ddof=1)))
        sd_p = np.sqrt(np.nanmean(np.nanvar(np.where(np.isfinite(lp), lp, np.nan), axis=1, ddof=1)))
        print(f'{f:6s} | {np.nanmedian(mo):15.4g} {np.nanmedian(mp):10.4g} | {ratio:15.3f} {wins:6d}/{n:<3d} {sign_p(wins, n):6.3f} | '
              f'{dl.mean():24.4g} {se:6.2g} {dl.mean() / se:5.1f} | {sd_o:12.3g} {sd_p:10.3g}')
    print('APF/BPF median-MSE ratio:  ours %.3f   particles %.3f' % (
        np.nanmedian(o['mse_apf']) / np.nanmedian(o['mse_bpf']), np.nanmedian(p['mse_apf']) / np.nanmedian(p['mse_bpf'])))
