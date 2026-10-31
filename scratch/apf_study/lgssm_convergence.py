"""BPF and APF against the Kalman filter on the linear Gaussian model of tests/test_pf_vs_kalman.py, for growing M.

Usage: python3 lgssm_convergence.py [n_runs]
For each M: RMSE of the filtering mean against the Kalman filter, bias of the log-likelihood estimate, and the
log of the mean of exp(ll) over the runs (the likelihood estimate is unbiased, so this should approach the Kalman value).
The RMSE of a Monte Carlo estimate falls as M^(-1/2).
"""
import os
import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'gaussfiltax'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.scipy.special import logsumexp
import test_pf_vs_kalman as t
from filters import pf

n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
params, ssm = t._model()
ys = t._observations()
ms_kf, ll_kf = t._kalman(ys)
print(f'Kalman filter: ll = {float(ll_kf):.3f}, T = {t.T}, runs per M: {n_runs}')
FILTERS = {'BPF': (pf.BPF, 0.5), 'APF (ESS rule)': (pf.APF, 0.5), 'APF (every step)': (pf.APF, 2.0)}
Ms = [100, 300, 1000, 3000, 10000, 30000]
rmse_all = {k: [] for k in FILTERS}
print(f'{"filter":18s} {"M":>6s} | {"RMSE mean":>9s} | {"mean(ll-ll_KF)":>14s} {"sd ll":>7s} | {"log mean exp(ll) - ll_KF":>24s}')
for name, (cls, ess) in FILTERS.items():
    flt = cls(None, ssm)
    for M in Ms:
        def one(k):
            outs, ll = flt.run(params, ys, M, k, None, ess)
            mean = jnp.einsum('it,itd->td', outs['weights'], outs['particles'])
            return jnp.sqrt(jnp.mean((mean - ms_kf) ** 2)), ll
        rm, ll = [np.array(a) for a in jax.jit(vmap(one))(jr.split(jr.PRNGKey(M), n_runs))]
        rmse_all[name].append(rm.mean())
        lme = float(logsumexp(jnp.array(ll)) - jnp.log(n_runs))
        print(f'{name:18s} {M:6d} | {rm.mean():9.4f} | {np.mean(ll) - float(ll_kf):14.4f} {ll.std(ddof=1):7.4f} | {lme - float(ll_kf):24.4f}', flush=True)
for name in FILTERS:
    slope = np.polyfit(np.log(Ms), np.log(rmse_all[name]), 1)[0]
    print(f'{name}: slope of log RMSE against log M = {slope:.2f} (Monte Carlo rate: -0.5)')
