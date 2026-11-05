"""Version 1 check: each repeat in data.nosync/repro must give the MSE of its table trial (util.numerics.mse on the repeat
against the stored mse.pkl of the table), and the same states. Also the LPE (util.numerics.log_prob_error). Usage (repository root, jaxenv):  python scratch/scatter_panels/check_v1.py [mse|lpe]
"""
import json
import os
import numpy as onp
import sys
from common import SETTINGS, FILTERS, HERE, REPRO_ROOT, sim_dir, table_dir, mse_of, metric_of, v1_selection_file
import util.io
from util.numerics import mse, log_prob_error

sel = json.load(open(v1_selection_file(sys.argv[1] if len(sys.argv) > 1 else 'mse')))
bad, n = [], 0
for a, s in SETTINGS:
    for lab, inf, _ in FILTERS:
        name = inf.split('/')[1]
        for pick in sel[f'a{a}_s{s}'][lab]:
            t = pick['trial']
            rep = os.path.join(REPRO_ROOT, 'experiments', sim_dir(a, s), inf, str(t))
            res = util.io.load(os.path.join(rep, 'results'))
            st = util.io.load(os.path.join(rep, 'gt'))[0]
            st_tab = util.io.load(os.path.join(table_dir(a, s, inf), str(t), 'gt'))[0]
            m = float(mse(res, name, st)[0])
            lp = float(log_prob_error(res, name, st)[0])
            n += 1
            ok = (onp.allclose(st, st_tab) and onp.isclose(m, mse_of(table_dir(a, s, inf), t), rtol=1e-5)
                  and onp.isclose(lp, metric_of(table_dir(a, s, inf), t, 'lpe'), rtol=1e-4))
            if not ok:
                bad.append((a, s, lab, t, m, mse_of(table_dir(a, s, inf), t), lp, metric_of(table_dir(a, s, inf), t, 'lpe')))
print(f'{n} repeats, {len(bad)} differ from the table')
for b in bad:
    print('DIFFERENT', b)
