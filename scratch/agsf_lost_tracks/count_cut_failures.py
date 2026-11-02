"""Count the lost runs (MSE > 10) of Exp. A whose divergence starts at a branch-cut crossing. Debug only (rule 5).
For each trial with a saved results.pkl: t1 = first step with squared error > 1. The divergence is 'at the cut' if the
true bearing arctan2(x2, x1) jumps by more than pi within the steps t1 - 3 .. t1. Also counts the trials whose true
bearing crosses the cut at least once (lost or not)."""
import os, sys, glob, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import jax.numpy as jnp
import util.io, util.numerics as nm

INF = {"lgsf": "lgsf", "ugsf": "ugsf", "lagsf": "lagsf", "uagsf": "uagsf", "bpf": "bpf", "apf": "apf", "rpf": "rpf", "upf": "upf"}
print(f"{'setting':22s} {'filter folder':42s} {'n':>3s} {'cross':>5s} {'lost':>4s} {'lost at cut':>11s} {'lost, no results':>16s}")
for exp in sorted(glob.glob("data.nosync/experiments/mtt/*/*/*/*/")):
    if "data_pf" in exp or "ess_2.0" in exp:
        continue
    parts = exp.rstrip("/").split("/")
    setting = parts[3].split("timesteps_500_")[1]
    inf = parts[5]
    n = cross = lost = at_cut = lost_nores = 0
    for f in sorted(glob.glob(exp + "*/mse.pkl")):
        d = os.path.dirname(f)
        if not os.path.exists(d + "/gt.pkl"):
            continue
        n += 1
        states = util.io.load(d + "/gt")[0]
        bt = jnp.arctan2(states[:, 2], states[:, 0])
        jumps = jnp.abs(jnp.diff(bt)) > jnp.pi
        cross += int(jnp.any(jumps))
        if float(pickle.load(open(f, "rb"))[0]) <= 10:
            continue
        lost += 1
        if not os.path.exists(d + "/results.pkl"):
            lost_nores += 1
            continue
        res = util.io.load(d + "/results")
        data = [res[0]] if isinstance(res, tuple) else res
        means, _ = nm.get_means_covs(data, INF[inf], with_covs=False)
        w = nm.get_weights(data, INF[inf])
        se = jnp.sum((jnp.einsum("kt,ktd->td", w, means) - states) ** 2, axis=1)
        t1 = int(jnp.argmax(se > 1))
        at_cut += int(jnp.any(jumps[max(t1 - 4, 0):t1]))
    if n:
        print(f"{setting:22s} {'/'.join(parts[4:]):42.42s} {n:3d} {cross:5d} {lost:4d} {at_cut:11d} {lost_nores:16d}")
