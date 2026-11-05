"""Scatter panels of Exp. A (see common.py): one figure for each setting, with 8 columns (the filters) and 3 rows (the best,
median and worst run by MSE). The panels come from FilterRunner.plot_three_panel_scatter (standalone_filter_runner.py):
the true positions (x_1, x_2), the component means (GSF, AGSF) and samples of the posterior, or the particles (PFs).

- v1: the table runs (repeats in data.nosync/repro, run_v1.zsh). Each cell has its own trial, thus its own trajectory.
- v2: the paired runs (run_v2.zsh, run_v2_select.zsh). Each row is one data set for all filters.

Usage (repository root, jaxenv):  python scratch/scatter_panels/plot_panels.py v1|v2
Writes figures/<version>_a<a>_s<s>.{pdf,png} in this folder.
"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import SETTINGS, FILTERS, HERE, ROOT, REPRO_ROOT, sim_dir, v2_dir
sys.path.insert(0, ROOT)
import standalone_filter_runner as sfr
import util.io

version = sys.argv[1]
assert version in ('v1', 'v2'), 'give v1 or v2'
sel = json.load(open(os.path.join(HERE, 'out', f'{version}_selection.json')))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
labels = [lab for lab, _, _ in FILTERS]
runner = sfr.FilterRunner(sfr.MTTConfig(R_scale=2.5, acc=0.5), num_timesteps=10, seed=0)  # only for the plot method


def load(folder, t):
    res = util.io.load(os.path.join(folder, str(t), 'results'))
    res = res[0] if isinstance(res, tuple) else res            # the PFs save (results, log likelihood)
    return res, util.io.load(os.path.join(folder, str(t), 'gt'))[0]


for a, s in SETTINGS:
    key = f'a{a}_s{s}'
    rows, states, titles, row_titles = [], [], [], []
    for r, rank in enumerate(('best', 'median', 'worst')):
        row, st, ti = {}, {}, {}
        for lab, inf, _ in FILTERS:
            if version == 'v1':
                pick = sel[key][lab][r]
                folder = os.path.join(REPRO_ROOT, 'experiments', sim_dir(a, s), inf)
                t, m = pick['trial'], pick['mse']
                ti[lab] = (f'{lab}\n' if r == 0 else '') + f'trial {t}, MSE {m:.3g}'
            else:
                pick = sel[key][r]
                folder = v2_dir(a, s, inf)
                t, m = pick['trial'], pick['mse'][lab]
                ti[lab] = (f'{lab}\n' if r == 0 else '') + f'MSE {m:.3g}'
            row[lab], st[lab] = load(folder, t)
        rows.append(row); states.append(st); titles.append(ti)
        row_titles.append(rank if version == 'v1' else f"{rank}: data set {pick['trial']} (median MSE {pick['median_mse']:.3g})")
    # show=True keeps the figure open (the Agg backend shows nothing), so that the title can change before the save
    runner.plot_three_panel_scatter(rows, dims=(0, 2), states_list=states, panel_titles=titles, row_titles=row_titles,
                                    algorithm_order=labels, rasterize_points=True, show=True)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.title.set_fontsize(10)
    what = 'table runs, each filter ranked by its own MSE' if version == 'v1' else 'paired runs, data sets ranked by the median MSE of the filters'
    fig.suptitle(f'Exp. A, a = {a}, $\\sigma^2$ = {s}: {what}', fontsize=14, y=1.01)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{version}_{key}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(OUT, f'{version}_{key}.{{png,pdf}}'))
