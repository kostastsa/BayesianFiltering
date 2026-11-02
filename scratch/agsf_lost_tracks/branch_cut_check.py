"""Check of the branch-cut mechanism in lost AGSF runs of Exp. A. Debug only (rule 5).
The bearing of mtt is arctan2(x2, x1), with a jump of 2 pi on the negative x1-axis. For each run: the first step with
squared error above 1, the smallest true range in the 20 steps before it, whether the true bearing crosses the cut
there, the fraction of the steps after it with the estimate within 0.3 rad of the cut, and the unscented variance of
the predicted bearing of the components (from the filtered mean and covariance, alpha = 1, kappa = 0)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import jax.numpy as jnp
from jax import vmap
import util.io

E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}/agsf/{f}/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9/{t}"
RUNS = [l.split() for l in sys.argv[1:]] or [l.split(",") for l in """
0.05,2.5e-05,uagsf,2,9
0.05,2.5e-05,lagsf,2,18
0.5,2.5e-05,lagsf,2,7
0.5,0.025,lagsf,2,5
0.05,0.025,lagsf,2,5
0.5,2.5,lagsf,100,11
0.5,2.5,uagsf,2,1
0.05,2.5,lagsf,2,5
""".split()]


def bearing_var(m, P):
    """Unscented variance of arctan2(x2, x1) for N(m, P), 2d sigma points with equal weights (lambda = 0)."""
    d = m.shape[0]
    lam, V = jnp.linalg.eigh(0.5 * (P + P.T))
    L = V * jnp.sqrt(jnp.maximum(lam, 0.0))
    pts = jnp.concatenate([m + jnp.sqrt(d) * L.T, m - jnp.sqrt(d) * L.T])
    b = jnp.arctan2(pts[:, 2], pts[:, 0])
    return jnp.var(b)


print(f"{'run':32s} {'t(SE>1)':>7s} {'min range':>9s} {'cut crossed':>11s} {'est near cut':>12s} {'bearing var: comps after / R':>30s}")
for a, s, f, M, t in RUNS:
    d = E.format(a=a, s=s, f=f, M=M, t=t)
    res = util.io.load(d + "/results"); states = util.io.load(d + "/gt")[0]
    m, P, w = res["means"], res["covariances"], res["weights"]
    est = jnp.einsum("kt,ktd->td", w / w.sum(0), m)
    se = jnp.sum((est - states) ** 2, axis=1)
    t1 = int(jnp.argmax(se > 1))
    lo = max(t1 - 20, 0)
    bt = jnp.arctan2(states[:, 2], states[:, 0])
    rng = jnp.hypot(states[:, 0], states[:, 2])
    crossed = bool(jnp.any(jnp.abs(jnp.diff(bt[lo:t1 + 1])) > jnp.pi))
    be = jnp.arctan2(est[t1:, 2], est[t1:, 0])
    near = float(jnp.mean(jnp.pi - jnp.abs(be) < 0.3))
    bv = vmap(vmap(bearing_var))(m[:, t1:], P[:, t1:])
    print(f"{f} M={M} a={a} s2={s} t{t:3s} {t1:7d} {float(rng[lo:t1 + 1].min()):9.3f} {str(crossed):>11s} {near:12.2f} "
          f"{float(jnp.median(bv)):12.3g} / {float(s):.2g}")
