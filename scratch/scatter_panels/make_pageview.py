"""Write pageview/scatter_panels.md: the scatter panels as a pageview page (~/tools/pageview), with the same content as
the claude.ai page (page/index.html): one tab for each setting, and in each one the tabs Both, MSE and LPE, with the
figure and the table of the selected runs. The data: page/data.json. The images: page/figs/*.jpg.

Usage (repository root):  python scratch/scatter_panels/make_pageview.py   then   pageview scratch/scatter_panels/pageview/scatter_panels.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'page', 'data.json')))
F = ['L-GSF', 'U-GSF', 'L-AGSF', 'U-AGSF', 'BPF', 'APF', 'RPF', 'UPF']
RANK = ['best', 'median', 'worst']
SETTINGS = [('0.5', '2.5', '2.5', 'high'), ('0.5', '0.025', '2.5×10⁻²', 'mid'), ('0.5', '2.5e-05', '2.5×10⁻⁵', 'low'),
            ('0.05', '2.5', '2.5', 'high'), ('0.05', '0.025', '2.5×10⁻²', 'mid'), ('0.05', '2.5e-05', '2.5×10⁻⁵', 'low')]
TEXT = {'mse': ('Ranked by MSE', 'The rows are the trials with the lowest, median and highest MSE of each filter '
                '(the squared error of the weighted mean).'),
        'lpe': ('Ranked by LPE', 'The rows are the trials with the lowest, median and highest LPE of each filter '
                '(the negative log density of the true state under the posterior).')}


def fmt(x):
    """3 significant digits; an exponent only for very large or very small magnitudes (as fmt in page/index.html)."""
    ax = abs(x)
    if ax >= 1e5 or (0 < ax < 1e-3):
        return f'{x:.2e}'.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')
    if ax >= 100:
        return f'{x:.0f}' if ax >= 1000 else f'{x:.1f}'
    return f'{float(f"{x:.3g}"):g}'


def table(key, m):
    rows = D[key][m]
    out = [f'::: {{.table .wide}}',
           '| Run | ' + ' | '.join(F) + ' |', '|---|' + '--:|' * len(F)]
    for r in range(3):
        cells = []
        for trial, mse, lpe in rows[r]:
            v = mse if m == 'mse' else lpe
            c = f'{fmt(v)} (trial {trial}' + (f', MSE {fmt(mse)})' if m == 'lpe' else ')')
            cells.append(f'**{c}** ⚠' if mse > 10 else c)
        out.append(f'| {RANK[r]} | ' + ' | '.join(cells) + ' |')
    out += ['', f'**{TEXT[m][0]}.** Each cell: the {m.upper()} of the run and its trial. ⚠: the MSE is above 10 '
            '(a lost track, the rule of Tables I and III).', ':::']
    return out


def block(key, m):
    title, text = TEXT[m]
    img = f'../page/figs/{"v1_" if m == "mse" else "v1_lpe_"}{key}.jpg'
    return [f'#### {title}', '', text, '', '::: {.figures .wide cols=1}',
            f'![{title}: 8 filters (columns) by the best, median and worst run (rows). Click the image to open it large.]({img})',
            ':::', ''] + table(key, m) + ['']


L = ['---', 'title: Scatter panels of the eight filters',
     'subtitle: Exp. A (mtt, wrapped bearing), table runs, task A12',
     'date: 2026-11-05', 'tags: [6 settings, 8 filters, best / median / worst, 144 exact repeats per ranking]',
     'view: page', 'toc: false', 'width: wide', '---', '',
     'For each setting of Tables I and III: the true position track (red), the component means (gold) and posterior '
     'samples or particles (black), for the best, median and worst table run of each filter, ranked by MSE or by LPE. '
     'Columns: L-GSF and U-GSF with M = 100, L-AGSF and U-AGSF with M = 10 (N = L = 5), and the BPF, APF, RPF and UPF '
     'with 1000 particles. Each filter has its own 100 trials, so each cell shows its own trajectory.', '',
     '::::::: tabs']
for a, s, s_txt, lvl in SETTINGS:
    key = f'a{a}_s{s}'
    L += [f':::::: {{.tab title="a = {a}, σ² = {s_txt} ({lvl} noise)"}}', '', f'### a = {a}, σ² = {s_txt}', '',
          '::::: tabs', ':::: {.tab title="Both"}', ''] + block(key, 'mse') + block(key, 'lpe') + ['::::']
    for m in ('mse', 'lpe'):
        L += [f':::: {{.tab title="{m.upper()}"}}', ''] + block(key, m) + ['::::']
    L += [':::::', '::::::']
L += [':::::::', '',
      '::: {.callout .note title="About the runs"}',
      'The figures show exact repeats of the table trials: each repeat gives the MSE and the LPE of the table. Lower is '
      'better for both metrics, so "best" is the lowest value. The paper uses the MSE ranking (D-45). '
      'Source: `scratch/scatter_panels/`.', ':::', '']
os.makedirs(os.path.join(HERE, 'pageview'), exist_ok=True)
open(os.path.join(HERE, 'pageview', 'scatter_panels.md'), 'w').write('\n'.join(L))
print('written', os.path.join(HERE, 'pageview', 'scatter_panels.md'), len(L), 'lines')
