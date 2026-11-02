"""Range and bearing errors of the weighted mean estimate in the lost runs of lost_runs_scatter.py. Debug only (rule 5).
The range is |(x1, x2)| and the bearing is atan2(x2, x1). Prints the median over time of |range error| and
|bearing error|, and the share of the squared position error that is along the bearing direction."""
import os, sys, glob, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import jax.numpy as jnp
import util.io, util.numerics as nm

PANELS = [l.split() for l in """
0.5 lgsf gsf/lgsf/ncomps_1000_iters_1 5
0.5 ugsf gsf/ugsf/ncomps_1000_iters_1_alpha_1.0_beta_0.0_kappa_0.0 6
0.5 lagsf agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9 6
0.5 uagsf agsf/uagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9 2
0.5 bpf pf/bpf/numprt_1000_ess_0.5 12
0.5 apf pf/apf/numprt_1000_ess_0.5 7
0.5 rpf pf/rpf/numprt_1000_ess_0.5 4
0.5 upf pf/upf/numprt_1000_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0 9
0.05 lgsf gsf/lgsf/ncomps_1000_iters_1 4
0.05 ugsf gsf/ugsf/ncomps_1000_iters_1_alpha_1.0_beta_0.0_kappa_0.0 4
0.05 lagsf agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9 4
0.05 uagsf agsf/uagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9 18
0.05 bpf pf/bpf/numprt_100_ess_0.5 24
0.05 apf pf/apf/numprt_1000_ess_0.5 11
0.05 upf pf/upf/numprt_1000_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0 2
""".strip().splitlines()]

print(f"{'a':5s} {'filter':6s} {'trial':>5s} {'med|dr|':>8s} {'med|dθ| (rad)':>14s} {'end range GT/est':>18s} {'bearing share of SE':>20s}")
for a, inf, folder, trial in PANELS:
    d = f"data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_2.5/{folder}/{trial}"
    res = util.io.load(d + "/results")
    data = [res[0]] if isinstance(res, tuple) else res
    states = util.io.load(d + "/gt")[0]
    means, _ = nm.get_means_covs(data, inf, with_covs=False)
    w = nm.get_weights(data, inf)
    est = jnp.einsum('kt,ktd->td', w, means)
    p_gt, p_est = states[:, [0, 2]], est[:, [0, 2]]
    r_gt, r_est = jnp.linalg.norm(p_gt, axis=1), jnp.linalg.norm(p_est, axis=1)
    th = jnp.arctan2(p_est[:, 1], p_est[:, 0]) - jnp.arctan2(p_gt[:, 1], p_gt[:, 0])
    th = (th + jnp.pi) % (2 * jnp.pi) - jnp.pi
    # Split the position error into the radial part and the part along the bearing (tangential) direction
    err = p_est - p_gt
    u_r = p_gt / r_gt[:, None]
    e_r = jnp.sum(err * u_r, axis=1)
    e_t2 = jnp.sum(err ** 2, axis=1) - e_r ** 2
    share = float(jnp.sum(e_t2) / jnp.sum(err ** 2))
    print(f"{a:5s} {inf:6s} {trial:>5s} {float(jnp.median(jnp.abs(r_est - r_gt))):8.2f} {float(jnp.median(jnp.abs(th))):14.2f} "
          f"{float(r_gt[-1]):8.1f} / {float(r_est[-1]):6.1f} {share:20.2f}")
