"""Disk saver of the 100-trial run (task A10, run100.zsh). For one trial k of the experiments of the descriptor files:
- AGSF: save the mean over the components of the smallest and the largest eigenvalue of Sigma^-1 Lambda (eig2s of
  aux_outputs.pkl) to eig2_mm.pkl, as (lam_min_t, lam_max_t), each of shape (T,). compare.py reads this file.
- If k > KEEP (default 20, the trials of A9): delete results.pkl, aux_outputs.pkl and observations.png of the trial.
  The metrics of 'main.py errors' and gt.pkl stay. Run 'main.py errors' for the trial before this script.

Usage (repository root, jaxenv):  python scratch/sinlin_showcase/shrink.py K FILE [FILE ...]
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'gaussfiltax'))
import numpy as onp
import experiment_descriptor as ed
import misc
import util.io

KEEP = int(os.environ.get('KEEP', 20))
k = int(sys.argv[1])
for f in sys.argv[2:]:
    for e in ed.parse(util.io.load_txt(f)):
        d = os.path.join(misc.get_root(), 'experiments', e.get_dir(), str(k))
        if not os.path.exists(os.path.join(d, 'mse_x1.pkl')):
            print(f'STOP: no mse_x1.pkl in {d} (run errors first)')
            sys.exit(1)
        aux = os.path.join(d, 'aux_outputs.pkl')
        if os.path.exists(aux) and not os.path.exists(os.path.join(d, 'eig2_mm.pkl')):
            eig = onp.asarray(util.io.load(aux[:-4])['eig2s'])                    # (T, M N, d), sorted eigenvalues
            util.io.save((eig[..., 0].mean(axis=1), eig[..., -1].mean(axis=1)), os.path.join(d, 'eig2_mm'))
        if k > KEEP:
            for name in ('results.pkl', 'aux_outputs.pkl', 'observations.png'):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    os.remove(p)
