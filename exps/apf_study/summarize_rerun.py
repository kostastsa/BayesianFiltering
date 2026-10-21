"""Summary of the files rerun_a{a}_r{sigma2}_M{M}.npz that expA_rerun.py writes (run it in exps/apf_study/out).

Usage: python3 ../summarize_rerun.py
For each setting: mean +- sd, median and the number of runs that lose track (MSE > 10) of the BPF and the APF, and the
paired comparison APF against BPF on the same data: median of the ratio MSE_APF / MSE_BPF and the two-sided exact sign test.
The rows "old rule" give the APF that resamples the ancestors at every step (the rule of the submitted tables).
"""
import glob
import math
import re
import numpy as np


def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = []
for fn in glob.glob('rerun_a*_r*_M*.npz'):
    a, r, M = re.match(r'rerun_a(.*)_r(.*)_M(\d+)\.npz', fn).groups()
    rows.append((float(a), -float(r), int(M), fn))
rows.sort()

fmt = lambda x: f'{x:.3g}'
print(f'{"a":>5s} {"sigma2":>8s} {"M":>7s} {"n":>4s} | {"BPF mean+-sd":>20s} {"med":>8s} {"lost":>4s} | {"APF mean+-sd":>20s} {"med":>8s} {"lost":>4s} | '
      f'{"ratio":>6s} {"wins":>7s} {"p":>6s} | {"APF old rule: mean":>18s} {"med":>8s} {"lost":>4s}')
for a, nr, M, fn in rows:
    d = np.load(fn)
    b, p = d['bpf'], d['apf']
    n = len(b)
    wins = int((p < b).sum())
    ratio = np.exp(np.median(np.log(p / b)))
    old = d['apf_every']
    print(f'{a:5g} {-nr:8.3g} {M:7d} {n:4d} | {fmt(b.mean()) + " +- " + fmt(b.std(ddof=1)):>20s} {fmt(np.median(b)):>8s} {int((b > 10).sum()):4d} | '
          f'{fmt(p.mean()) + " +- " + fmt(p.std(ddof=1)):>20s} {fmt(np.median(p)):>8s} {int((p > 10).sum()):4d} | '
          f'{ratio:6.2f} {wins:3d}/{n:<3d} {sign_p(wins, n):6.3f} | {fmt(old.mean()) + " +- " + fmt(old.std(ddof=1)):>18s} {fmt(np.median(old)):>8s} {int((old > 10).sum()):4d}')
