"""Scatter panels of Exp. A (see common.py): one figure for each setting, with 8 columns (the filters) and 3 rows (the best,
median and worst table run of each filter by METRIC). The panels come from FilterRunner.plot_three_panel_scatter
(standalone_filter_runner.py): the true positions (x_1, x_2), the component means (GSF, AGSF) and samples of the
posterior, or the particles (PFs). The runs are the exact repeats of the table trials in data.nosync/repro (run_v1.zsh).
Each cell has its own trial, thus its own trajectory.

Usage (repository root, jaxenv):  python scratch/scatter_panels/plot_panels.py [mse|lpe]
Writes figures/v1_a<a>_s<s>.{pdf,png} (MSE) or figures/v1_lpe_a<a>_s<s>.{pdf,png} (LPE) in this folder.
"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import SETTINGS, FILTERS, HERE, ROOT, REPRO_ROOT, METRICS, sim_dir, v1_selection_file
sys.path.insert(0, ROOT)
import standalone_filter_runner as sfr
import util.io

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'mse'
assert METRIC in METRICS, f'give one of {METRICS}'
sel = json.load(open(v1_selection_file(METRIC)))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
labels = [lab for lab, _, _ in FILTERS]
runner = sfr.FilterRunner(sfr.MTTConfig(R_scale=2.5, acc=0.5), num_timesteps=10, seed=0)  # only for the plot method
prefix = 'v1' if METRIC == 'mse' else f'v1_{METRIC}'


def load(folder, t):
    res = util.io.load(os.path.join(folder, str(t), 'results'))
    res = res[0] if isinstance(res, tuple) else res            # the PFs save (results, log likelihood)
    return res, util.io.load(os.path.join(folder, str(t), 'gt'))[0]


for a, s in SETTINGS:
    key = f'a{a}_s{s}'
    rows, states, titles = [], [], []
    for r in range(3):
        row, st, ti = {}, {}, {}
        for lab, inf, _ in FILTERS:
            pick = sel[key][lab][r]
            t = pick['trial']
            line = f"trial {t}, MSE {pick['mse']:.3g}" if METRIC == 'mse' else f"trial {t}, LPE {pick['lpe']:.3g} (MSE {pick['mse']:.3g})"
            ti[lab] = (f'{lab}\n' if r == 0 else '') + line
            row[lab], st[lab] = load(os.path.join(REPRO_ROOT, 'experiments', sim_dir(a, s), inf), t)
        rows.append(row); states.append(st); titles.append(ti)
    # show=True keeps the figure open (the Agg backend shows nothing), so that the title can change before the save
    runner.plot_three_panel_scatter(rows, dims=(0, 2), states_list=states, panel_titles=titles,
                                    row_titles=['best', 'median', 'worst'], algorithm_order=labels,
                                    rasterize_points=True, show=True)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.title.set_fontsize(10)
    fig.suptitle(f'Exp. A, a = {a}, $\\sigma^2$ = {s}: table runs, each filter ranked by its own {METRIC.upper()}',
                 fontsize=14, y=1.01)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{prefix}_{key}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(OUT, f'{prefix}_{key}.{{png,pdf}}'))
