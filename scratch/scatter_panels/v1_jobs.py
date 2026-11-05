"""Version 1 (see common.py): select the best, median and worst table trial of each filter by its MSE (mse.pkl of
'main.py errors', 100 trials), and print the 'main.py trials' calls that repeat these trials exactly.

A table trial k of a group ran as 'main.py trials SEED_BASE+k k k <files of the group>' (run_tables_I_III.zsh). The same
call with '--only <folder>' and GAUSSFILTAX_ROOT=REPRO_ROOT gives the same keys and writes only into REPRO_ROOT, so the
table folders do not change. Writes out/v1_selection.json (MSE) or out/v1_selection_lpe.json (LPE). Output lines: SEED TRIAL FILES(comma) ONLY(comma).

The rank uses METRIC (mse or lpe, default mse): the best run has the lowest value.

Usage (repository root, jaxenv):  python scratch/scatter_panels/v1_jobs.py [mse|lpe]
"""
import json
import os
import sys
from common import (SETTINGS, FILTERS, GROUPS, TRIALS, HERE, METRICS, sim_dir, table_dir, group_files, metric_of,
                    best_median_worst, v1_selection_file)

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'mse'
assert METRIC in METRICS, f'give one of {METRICS}'

sel, calls = {}, {}
for a, s in SETTINGS:
    key = f'a{a}_s{s}'
    sel[key] = {}
    for lab, inf, group in FILTERS:
        folder = table_dir(a, s, inf)
        vals = {t: metric_of(folder, t, METRIC) for t in range(1, TRIALS + 1)}
        picks = best_median_worst(vals)
        sel[key][lab] = [{'rank': r, 'trial': t, 'mse': metric_of(folder, t, 'mse'), 'lpe': metric_of(folder, t, 'lpe')}
                         for r, t in zip(('best', 'median', 'worst'), picks)]
        for t in picks:
            calls.setdefault((group, a, t), set()).add(f'{sim_dir(a, s)}/{inf}')
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(sel, open(v1_selection_file(METRIC), 'w'), indent=1)
for (group, a, t), only in sorted(calls.items()):
    print(GROUPS[group][1] + t, t, ','.join(group_files(group, a)), ','.join(sorted(only)))
