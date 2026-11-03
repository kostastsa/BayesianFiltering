"""Step 3 (jaxenv): table of our BPF/APF (main.py runs) against the particles library on the same data sets, for each setting.

Usage (repository root):  python3 scratch/particles_check/make_table.py A SIGMA2[,SIGMA2...]     e.g.  ... make_table.py 0.5 2.5,0.025,2.5e-05
Ours: the MSE that main.py errors saved in mse.pkl (util.numerics.mse). If the runs on the stored data (descriptor files
mtt_<filter>_samedata_scan_num_prt.txt, run_samedata.zsh) are complete, they give R runs per data set. Otherwise the table uses
the 30 trials of the runs with 1000 particles (one run per data set).
particles: the filter means of run_particles.py, with the MSE from util.numerics.mse (N_RUNS runs per data set).
Columns:
- median MSE over all runs of each library;
- lost tracks: the runs with MSE > 10 (the threshold of Error summary.txt). In a lost track the filter does not follow the target.
  p (lost): Cochran-Mantel-Haenszel test of equal lost-track rates, with one 2 x 2 table (library x lost) for each data set;
- sign test: in how many data sets our median MSE is below the particles median MSE (1/2 if the two filters are the same
  algorithm), two-sided binomial p.
Output: the table on the screen, scratch/particles_check/out/table.md and table.tex.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
import numpy as onp
import jax.numpy as jnp
from jax import vmap
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import StratifiedTable
import util.io
import util.numerics

A, SIGMAS = sys.argv[1], sys.argv[2].split(',')
W = '_wrap' if len(sys.argv) > 3 and sys.argv[3] == 'wrap' else ''    # wrapped bearing (D-33)
M, N_DATA, LOST = 1000, 30, 10.0


def particles_mse(means, states):
    """util.numerics.mse of a filter mean (T, 4): one 'particle' with weight 1 gives the mean itself."""
    data = ({'particles': means[None], 'weights': jnp.ones((1, means.shape[0]))}, None)
    return util.numerics.mse(data, 'bpf', states)[0]


def ours_mse(sim_dir, f):
    """MSE of our runs, shape (N_DATA, R): the runs on the stored data if they are complete, else the single runs."""
    rep_dir = os.path.join(sim_dir, f'data_pf_{f}_numprt_{M}_ess_0.5', 'pf', f, f'numprt_{M}_ess_0.5')
    if os.path.isdir(rep_dir):
        trials = sorted(int(t) for t in os.listdir(rep_dir) if t.isdigit() and os.path.exists(os.path.join(rep_dir, t, 'mse.pkl')))
        n = len(trials) // N_DATA * N_DATA
        if n >= N_DATA and trials[:n] == list(range(1, n + 1)):
            mse = onp.array([util.io.load(os.path.join(rep_dir, str(t), 'mse'))[0] for t in range(1, n + 1)])
            return mse.reshape(n // N_DATA, N_DATA).T     # trial t uses data set ((t - 1) mod N_DATA) + 1
    single = os.path.join(sim_dir, 'pf', f, f'numprt_{M}_ess_0.5')
    return onp.array([util.io.load(os.path.join(single, str(t), 'mse'))[0] for t in range(1, N_DATA + 1)])[:, None]


def cmh_p(ours, part):
    tables = [onp.array([[(o > LOST).sum(), (o <= LOST).sum()], [(q > LOST).sum(), (q <= LOST).sum()]]) for o, q in zip(ours, part)]
    tables = [t for t in tables if t[:, 0].sum() > 0 and t[:, 1].sum() > 0]    # data sets with lost and tracked runs
    if not tables:
        return onp.nan
    return StratifiedTable([t.astype(float) for t in tables]).test_null_odds(correction=True).pvalue


rows = []
for s in SIGMAS:
    sim_dir = os.path.join(ROOT, 'data.nosync', 'experiments', 'mtt', f'state-dim_4_emission-dim_2_num-timesteps_500_a_{A}_sigma2_{s}{W}')
    d = onp.load(os.path.join(HERE, 'out', f'gt_a{A}_sigma2_{s}{W}.npz'))
    p = onp.load(os.path.join(HERE, 'out', f'particles_a{A}_sigma2_{s}{W}.npz'))
    for f in ('bpf', 'apf'):
        ours = ours_mse(sim_dir, f)
        part = onp.array(vmap(lambda ms, st: vmap(lambda m: particles_mse(m, st))(ms))(jnp.array(p[f'means_{f}']), jnp.array(d[f'states_{f}'])))
        below = int((onp.median(ours, axis=1) < onp.median(part, axis=1)).sum())
        rows.append(dict(sigma2=s, filter=f.upper(), r_ours=ours.shape[1], r_part=part.shape[1],
                         med_ours=onp.median(ours), med_part=onp.median(part),
                         lost_ours=100 * (ours > LOST).mean(), lost_part=100 * (part > LOST).mean(), p_lost=cmh_p(ours, part),
                         below=below, p_sign=binomtest(below, N_DATA).pvalue))

fmt_p = lambda v: '-' if onp.isnan(v) else f'{v:.2f}'
hdr = ('| sigma2 | Filter | Runs per data set: ours / particles | Median MSE: ours | Median MSE: particles | Lost tracks: ours '
       '| Lost tracks: particles | p (lost) | Data sets: ours < particles | p (sign) |\n|---|---|---|---|---|---|---|---|---|---|')
md = [hdr] + [f"| {r['sigma2']} | {r['filter']} | {r['r_ours']} / {r['r_part']} | {r['med_ours']:.4f} | {r['med_part']:.4f} "
              f"| {r['lost_ours']:.1f}% | {r['lost_part']:.1f}% | {fmt_p(r['p_lost'])} | {r['below']}/{N_DATA} | {r['p_sign']:.2f} |" for r in rows]
note = (f"Exp. A (mtt), a = {A}, M = {M}, {N_DATA} data sets for each setting and filter. Ours: main.py runs. "
        f"particles: run_particles.py. MSE: util.numerics.mse. Lost track: MSE > {LOST:g}.")
open(os.path.join(HERE, 'out', f'table{W}.md'), 'w').write(note + '\n\n' + '\n'.join(md) + '\n')

tex = [r'\begin{tabular}{llcccccc}', r'\toprule',
       r' & & \multicolumn{2}{c}{Median MSE} & \multicolumn{2}{c}{Lost tracks} & & \\',
       r'$\sigma^2$ & Filter & Ours & \texttt{particles} & Ours & \texttt{particles} & $p$ (lost) & $p$ (sign) \\', r'\midrule']
tex += [f"{r['sigma2']} & {r['filter']} & {r['med_ours']:.3f} & {r['med_part']:.3f} & {r['lost_ours']:.1f}\\% & "
        f"{r['lost_part']:.1f}\\% & {fmt_p(r['p_lost'])} & {r['p_sign']:.2f} \\\\" for r in rows]
tex += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(HERE, 'out', f'table{W}.tex'), 'w').write('\n'.join(tex) + '\n')
print(note + '\n\n' + '\n'.join(md))
