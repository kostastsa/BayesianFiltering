"""Version 2 (see common.py): for each setting, write descs/paired_a<a>_s<s>.txt with the 8 filters of common.FILTERS.
Each experiment is the table row with a data block: the stored data of trials 1 to 100 of the row SRC. Thus all
filters of one trial run on the same trajectory. Guard: stop if a target folder already has trial folders.

Usage (repository root, jaxenv):  python scratch/scatter_panels/make_v2_descs.py
"""
import os
from common import SETTINGS, FILTERS, GROUPS, TRIALS, SRC, HERE, sim_dir, group_files, v2_dir, v2_file
import experiment_descriptor as ed  # after common, which puts gaussfiltax/ on sys.path
import util.io

os.makedirs(os.path.join(HERE, 'descs'), exist_ok=True)
for a, s in SETTINGS:
    exps = {}
    for g in GROUPS:
        for f in group_files(g, a):
            for e in ed.parse(util.io.load_txt(f)):
                exps[e.get_dir()] = e
    out = []
    for lab, inf, _ in FILTERS:
        e = exps[f'{sim_dir(a, s)}/{inf}']
        e.data = ed.DataDescriptor()
        e.data.src, e.data.trials = SRC, TRIALS
        d = v2_dir(a, s, inf)
        if os.path.isdir(d) and any(x.isdigit() for x in os.listdir(d)):
            raise SystemExit(f'STOP: {d} already has trial folders')
        out.append(e.pprint())
    open(v2_file(a, s), 'w').write(''.join(out))
    print(v2_file(a, s), len(out), 'experiments')
