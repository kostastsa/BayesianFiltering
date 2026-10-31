"""Step 1 (jaxenv): export the data of the main.py runs of Exp. A (mtt, M = 1000) at one setting (a, sigma2) for the particles library.

Usage (repository root):  python3 scratch/particles_check/export_gt.py A SIGMA2     e.g.  ... export_gt.py 0.5 2.5e-05
Write SIGMA2 as in the folder names of data.nosync (2.5e-05, not 2.5e-5).
Reads gt.pkl of trials 1..N_TRIALS of the BPF and APF runs. main.py splits the key again for each descriptor, so the BPF and APF
runs of one trial have different data. The particles BPF runs on the BPF data and the particles APF on the APF data.
Output: scratch/particles_check/out/gt_a<A>_sigma2_<SIGMA2>.npz with the states and observations of each filter, the inputs, the model parameters, and
reference values of the dynamics and emission functions to check the numpy model in run_particles.py.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
import numpy as onp
import jax.numpy as jnp
import util.io
from simulators import mtt

A, SIGMA2 = sys.argv[1], sys.argv[2]
M, N_TRIALS, T = 1000, 30, 500
EXP = os.path.join(ROOT, 'data.nosync', 'experiments', 'mtt', f'state-dim_4_emission-dim_2_num-timesteps_{T}_a_{A}_sigma2_{SIGMA2}', 'pf')

data = {}
for f in ('bpf', 'apf'):
    gts = [util.io.load(os.path.join(EXP, f, f'numprt_{M}_ess_0.5', str(trial), 'gt')) for trial in range(1, N_TRIALS + 1)]
    data[f'states_{f}'] = onp.stack([onp.array(g[0]) for g in gts])
    data[f'obs_{f}'] = onp.stack([onp.array(g[1]) for g in gts])
    for g in gts:   # the model parameters are the same in all trials
        for v, w in zip((g[2].initial.mean.value, g[2].dynamics.cov.value, g[2].emissions.cov.value),
                        (gts[0][2].initial.mean.value, gts[0][2].dynamics.cov.value, gts[0][2].emissions.cov.value)):
            assert onp.array_equal(onp.array(v), onp.array(w))
ps = gts[0][2]

out = mtt.setup(4, 2, 0, float(A), float(SIGMA2))
ssm = out['ssm']
inputs = mtt.get_inputs(None, T)
xt = data['states_bpf'][0][[0, 50, 150, 250, 350, 450]]
chk_f = onp.stack([[onp.array(ssm.dynamics_function(ps, jnp.array(x), jnp.zeros(2), u)) for x in xt] for u in (0, 1, 2)])
chk_g = onp.array([ssm.emission_log_prob(ps, jnp.array(data['obs_bpf'][0][10]), jnp.array(x), 0) for x in xt])

os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
onp.savez(os.path.join(HERE, 'out', f'gt_a{A}_sigma2_{SIGMA2}.npz'), **data, inputs=onp.array(inputs),
          m0=onp.array(ps.initial.mean.value), P0=onp.array(ps.initial.cov.value), Q=onp.array(ps.dynamics.cov.value),
          R=onp.array(ps.emissions.cov.value), q_bias=onp.array(ps.dynamics.bias.value), r_bias=onp.array(ps.emissions.bias.value),
          a=float(A), M=M, xt=xt, chk_f=chk_f, chk_g=chk_g, y_chk=data['obs_bpf'][0][10])
print(f'a = {A}, sigma2 = {SIGMA2}: exported {N_TRIALS} trials of the BPF and APF runs. Q = {onp.diag(ps.dynamics.cov.value)}, R = {onp.diag(ps.emissions.cov.value)}')
