"""Diagnostics of the UPF (filters/pf.py) on the stored data of one trial of Exp. A. Debug only (rule 5).
Usage: python scratch/upf_check/diagnose.py <sigma2> <num_prt> <trial> [a]
Prints the ESS, the number of particles with a non-finite log weight, and the MSE of the UPF and the BPF on the same data."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import jax, jax.numpy as jnp, jax.random as jr
from simulators import mtt
from filters import pf
from util.param import sample_prior
import util.io, util.numerics as nm

sigma2, N, trial = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
s2 = sys.argv[1]
d = (f"data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s2}"
     f"/pf/upf/numprt_{N}_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0/{trial}")
states, obs, true_ps = util.io.load(d + "/gt")
setup = mtt.setup(4, 2, 1, a, sigma2)
ssm, props = setup["ssm"], setup["props"]
inputs = mtt.get_inputs(None, 500)

for name, cls in [("UPF", pf.UPF), ("BPF", pf.BPF)]:
    for seed in [0, 1]:
        out, ll = cls(props, ssm).run(true_ps, obs, N, jr.PRNGKey(seed), inputs, 0.5)
        uw = out["u_weights"]
        n_zero = jnp.sum(uw == 0, axis=0)
        w = uw / jnp.sum(uw, axis=0)
        ess = 1.0 / jnp.sum(w ** 2, axis=0)
        mse, mse_t = nm.mse([out], "upf", states)
        bad = jnp.where(mse_t > 10)[0]
        print(f"{name} seed {seed}: MSE {float(mse):.4f}, ll {float(ll):.1f}, min ESS {float(ess.min()):.1f}, "
              f"median ESS {float(jnp.median(ess)):.0f}, max zero weights {int(n_zero.max())}, "
              f"first step with sq. error > 10: {int(bad[0]) if bad.size else None}, any NaN: {bool(jnp.any(jnp.isnan(out['particles'])))}")
