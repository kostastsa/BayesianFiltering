"""Quick check of the NEES as an 'empty space' metric on stored runs (wrapped bearing). Debug only (rule 5).
NEES_t = (x_t - m_t)^T P_t^{-1} (x_t - m_t), with m_t, P_t the mean and covariance of the whole filter posterior
(mixture moments for the GSF and AGSF, weighted particle moments for the PFs). A step 'misses' if NEES_t > chi2_4(0.999).
Prints, per row: the median over trials of the fraction of missed steps, and the trials with more than 20 % missed steps.
Only runs with a saved results.pkl (the PF runs with 10^4 and 10^5 particles have none)."""
import os, sys, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import numpy as np
import util.io, util.numerics as nm

CHI2 = 18.467  # chi2 quantile 0.999, 4 degrees of freedom
E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}_wrap/{f}"
ROWS = [("L-GSF 1000", "lgsf", "gsf/lgsf/ncomps_1000_iters_1"),
        ("U-GSF 100", "ugsf", "gsf/ugsf/ncomps_100_iters_1_alpha_1.0_beta_0.0_kappa_0.0"),
        ("L-AGSF 100", "lagsf", "agsf/lagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"),
        ("U-AGSF 100", "uagsf", "agsf/uagsf/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"),
        ("BPF 1000", "bpf", "pf/bpf/numprt_1000_ess_0.5"), ("APF 1000", "apf", "pf/apf/numprt_1000_ess_0.5"),
        ("RPF 1000", "rpf", "pf/rpf/numprt_1000_ess_0.5"), ("UPF 1000", "upf", "pf/upf/numprt_1000_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0")]


def moments(res, inf):
    data = [res[0]] if isinstance(res, tuple) else res
    w = np.asarray(nm.get_weights(data, inf))                     # (K, T)
    if inf in nm.PF_NAMES:
        x = np.asarray(data[0]["particles"])                       # (K, T, d)
        m = np.einsum("kt,ktd->td", w, x); dx = x - m
        P = np.einsum("kt,ktd,kte->tde", w, dx, dx)
    else:
        mu, S = np.asarray(res["means"]), np.asarray(res["covariances"])
        m = np.einsum("kt,ktd->td", w, mu); dx = mu - m
        P = np.einsum("kt,ktde->tde", w, S) + np.einsum("kt,ktd,kte->tde", w, dx, dx)
    return m, P


for a, s in [("0.5", "2.5"), ("0.05", "2.5"), ("0.5", "2.5e-05"), ("0.05", "2.5e-05")]:
    print(f"\n### a = {a}, sigma2 = {s}\n\n| Row | trials | median % of steps outside the 99.9 % region | trials with > 20 % outside | lost (MSE > 10) |\n|---|---|---|---|---|")
    for label, inf, f in ROWS:
        fr, lost = [], 0
        for d in sorted(glob.glob(E.format(a=a, s=s, f=f) + "/*/")):
            if not os.path.exists(d + "results.pkl"):
                continue
            res = util.io.load(d + "results"); x = np.asarray(util.io.load(d + "gt")[0])
            m, P = moments(res, inf)
            e = x - m
            P = P + 1e-12 * np.eye(4)
            nees = np.einsum("td,td->t", e, np.linalg.solve(P, e[..., None])[..., 0])
            fr.append(np.mean(nees > CHI2))
            lost += float(np.mean(np.sum(e ** 2, axis=1))) > 10
        fr = np.array(fr)
        print(f"| {label} | {len(fr)} | {100 * np.median(fr):.1f} % | {int(np.sum(fr > 0.2))} | {lost} |")
