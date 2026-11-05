"""Version 2 (see common.py): for each setting, rank the V2_TRIALS (40) data sets by the median over the 8 filters of the MSE
(mse.pkl of 'main.py errors'), and select the best, median and worst data set. Writes out/v2_selection.json and prints
one line for each setting: FILE TRIAL TRIAL TRIAL (for run_v2_select.zsh, which runs these trials again for results.pkl).

Usage (repository root, jaxenv):  python scratch/scatter_panels/v2_select.py
"""
import json
import os
import numpy as onp
from common import SETTINGS, FILTERS, V2_TRIALS, HERE, v2_dir, v2_file, mse_of, best_median_worst

sel = {}
for a, s in SETTINGS:
    m = {lab: {t: mse_of(v2_dir(a, s, inf), t) for t in range(1, V2_TRIALS + 1)} for lab, inf, _ in FILTERS}
    med = {t: float(onp.median([m[lab][t] for lab in m])) for t in range(1, V2_TRIALS + 1)}
    picks = best_median_worst(med)
    sel[f'a{a}_s{s}'] = [{'rank': r, 'trial': t, 'median_mse': med[t], 'mse': {lab: m[lab][t] for lab in m}}
                         for r, t in zip(('best', 'median', 'worst'), picks)]
    print(v2_file(a, s), *picks)
json.dump(sel, open(os.path.join(HERE, 'out', 'v2_selection.json'), 'w'), indent=1)
