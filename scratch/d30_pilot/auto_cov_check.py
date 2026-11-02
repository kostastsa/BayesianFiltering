"""Quick check: does the automatic covariance rule remove the lost tracks of the AGSF with the wrapped bearing at
sigma2 = 2.5? Debug only (rule 5), with the FilterRunner of standalone_filter_runner.py. No results go to data.nosync.
Runs: the four AGSF runs of figures/paired_high_noise.png (M = 100, N = L = 5), on their stored data (wrapped model).
Settings: prop/prop (rho1 = rho2 = 0.9, the table setting), prop/auto (rho1 = 0.9, rho2 from the rule, the adaptive
variant of the paper) and auto/auto. The rule (filters/agsf.py, auto_cov_selection) is
rho = min(0.95, r Tr(P J^T J) / sum_i Tr(H_i P)^2) with r = 2/L = 2/N = 0.4. Filter seeds 0, 1, 2, the same for all
settings. Prints the MSE of each run and the median rho; plots seed 0 next to the stored pilot run.
Usage (repository root, jaxenv):  python scratch/d30_pilot/auto_cov_check.py
"""
import os, sys, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import jax.numpy as jnp, jax.random as jr
import standalone_filter_runner as sfr
from simulators import mtt
import util.io, util.numerics as nm

E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_2.5_wrap/agsf/{f}/M_100_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9/{t}"
CASES = [("uagsf", "0.05", 2), ("lagsf", "0.05", 18), ("uagsf", "0.5", 5), ("lagsf", "0.5", 4)]
SETTINGS = {"prop/prop": ("prop", 0.9, "prop", 0.9), "prop/auto": ("prop", 0.9, "auto", 0.4), "auto/auto": ("auto", 0.4, "auto", 0.4)}
SEEDS = [0, 1, 2]
NAME = {"lagsf": "L-AGSF", "uagsf": "U-AGSF"}


class WrappedMTT(sfr.SimulatorConfig):
    def __init__(self, a):
        super().__init__(state_dim=4, emission_dim=2, input_dim=1); self.a = a
    def setup(self):
        return mtt.setup(4, 2, 1, self.a, 2.5, True)
    def get_inputs(self, key, num_timesteps):
        return mtt.get_inputs(key, num_timesteps)


def run(runner, f, setting, seed):
    ac1, r1, ac2, r2 = SETTINGS[setting]
    runner.key = jr.PRNGKey(seed)
    kw = dict(M=100, N=5, L=5, autocov1=ac1, r1=r1, autocov2=ac2, r2=r2, num_iterations=1)
    res = runner.run_lagsf(**kw) if f == "lagsf" else runner.run_uagsf(alpha=1.0, beta=0.0, kappa=0.0, **kw)
    return res


rows, states, titles, lines = [], [], [], []
for f, a, t in CASES:
    d = E.format(a=a, f=f, t=t)
    st, obs, tp = util.io.load(d + "/gt")
    runner = sfr.FilterRunner(WrappedMTT(float(a)), num_timesteps=500, seed=0)
    runner.states, runner.observations, runner.true_params = st, obs, tp
    stored_mse = float(pickle.load(open(d + "/mse.pkl", "rb"))[0])
    row = {"stored prop/prop": util.io.load(d + "/results")}
    title = {"stored prop/prop": f"{NAME[f]} M=100, a={a}, $\\sigma^2$=2.5, trial {t}\npilot run, prop/prop: MSE {stored_mse:.3g}"}
    line = f"{NAME[f]} a={a} trial {t:2d} | stored prop/prop {stored_mse:6.2f} |"
    for setting in SETTINGS:
        mses = []
        for seed in SEEDS:
            res = run(runner, f, setting, seed)
            m = float(nm.mse(res, "lagsf", st)[0])
            mses.append(m)
            if seed == 0:
                aux = res["aux_outputs"]
                rho1 = float(jnp.median(aux["rho1s"])); rho2 = float(jnp.median(aux["rho2s"]))
                row[setting] = res
                title[setting] = f"{setting}, seed 0: MSE {m:.3g}\nmedian rho1 {rho1:.2f}, rho2 {rho2:.2f}"
        lost = sum(m > 10 for m in mses)
        line += f" {setting}: " + " ".join(f"{m:6.2f}" for m in mses) + f" (lost {lost}/3) |"
    rows.append(row); states.append(st); titles.append(title); lines.append(line)
    print(line, flush=True)

order = ["stored prop/prop"] + list(SETTINGS)
runner.plot_three_panel_scatter(rows, dims=(0, 2), states_list=states, panel_titles=titles,
                                algorithm_order=order, rasterize_points=True, show=True)
fig = plt.gcf()
for ax in fig.axes:
    if ax.has_data():
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        if x0 < 0:
            ax.plot([x0, 0], [0, 0], "b--", lw=1.0, zorder=11)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.title.set_fontsize(9)
fig.suptitle("Wrapped bearing, $\\sigma^2$ = 2.5: fixed (prop) against automatic (auto) split covariances", fontsize=13, y=1.01)
out = os.path.join(os.path.dirname(__file__), "figures", "auto_cov_high_noise")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
open(os.path.join(os.path.dirname(__file__), "auto_cov_check.txt"), "w").write("\n".join(lines) + "\n")
print("saved", out + ".{png,pdf}")
