"""Comparison scatter of paired runs of the D-30 pilot: raw bearing (T1, top row) and wrapped bearing (bottom row),
same trial, thus the same states and filter keys. Debug view only (rule 5). No filter runs again.
Figure 1 (low noise): the lost runs that the wrap corrects. Figure 2 (sigma2 = 2.5): for six rows with M >= 10, the
run with the median MSE among the trials that only the wrapped version loses. Blue dashed line: the branch cut.
Usage (repository root, jaxenv):  python scratch/d30_pilot/paired_scatter.py
"""
import os, sys, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import standalone_filter_runner as sfr
import util.io

E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}{w}"
F = {"L-GSF": "gsf/lgsf/ncomps_{M}_iters_1", "U-GSF": "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0",
     "L-AGSF": "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9",
     "U-AGSF": "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"}
OUT = os.path.join(os.path.dirname(__file__), "figures")


def d(name, M, a, s, w):
    return f"{E.format(a=a, s=s, w='_wrap' if w else '')}/{F[name].format(M=M)}"


def mse(path, t):
    return float(pickle.load(open(f"{path}/{t}/mse.pkl", "rb"))[0])


def median_new_lost(name, M, a, s):
    """Trial with the median new MSE among the trials lost (MSE > 10) only with the wrapped bearing."""
    cand = sorted((mse(d(name, M, a, s, True), t), t) for t in range(1, 21)
                  if mse(d(name, M, a, s, True), t) > 10 and mse(d(name, M, a, s, False), t) <= 10)
    return cand[(len(cand) - 1) // 2][1]


def draw(cases, path, suptitle):
    keys = [f"c{j}" for j in range(len(cases))]
    rows, states, titles = [{}, {}], [{}, {}], [{}, {}]
    for k, (name, M, a, s, t) in zip(keys, cases):
        for i, w in enumerate([False, True]):
            p = f"{d(name, M, a, s, w)}/{t}"
            rows[i][k] = util.io.load(p + "/results")
            states[i][k] = util.io.load(p + "/gt")[0]
            titles[i][k] = (f"{name} M={M}, a={a}, $\\sigma^2$={s}, trial {t}\n" if i == 0 else "") + \
                           f"{'raw' if not w else 'wrapped'}: MSE {mse(d(name, M, a, s, w), t):.3g}"
    runner.plot_three_panel_scatter(rows, dims=(0, 2), states_list=states, panel_titles=titles,
                                    row_titles=["raw bearing (old)", "wrapped bearing (new)"],
                                    algorithm_order=keys, rasterize_points=True, show=True)
    fig = plt.gcf()
    for ax in fig.axes:
        if ax.has_data():
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            if x0 < 0:
                ax.plot([x0, 0], [0, 0], "b--", lw=1.0, zorder=11)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.title.set_fontsize(9)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".{png,pdf}")


runner = sfr.FilterRunner(sfr.MTTConfig(R_scale=2.5, acc=0.5), num_timesteps=10, seed=0)
os.makedirs(OUT, exist_ok=True)

low = [("L-GSF", 1000, "0.05", "0.025", 12), ("L-AGSF", 2, "0.05", "0.025", 5), ("L-AGSF", 2, "0.5", "0.025", 5),
       ("L-GSF", 100, "0.05", "2.5e-05", 4), ("L-AGSF", 2, "0.05", "2.5e-05", 18), ("U-AGSF", 2, "0.05", "2.5e-05", 9)]
draw(low, os.path.join(OUT, "paired_low_noise"), "Low noise: lost runs that the wrapped bearing corrects")

high = [(n, M, a, "2.5", median_new_lost(n, M, a, "2.5")) for n, M, a in
        [("U-AGSF", 100, "0.05"), ("L-AGSF", 100, "0.05"), ("U-AGSF", 100, "0.5"),
         ("L-AGSF", 100, "0.5"), ("U-GSF", 1000, "0.5"), ("U-GSF", 1000, "0.05")]]
print("high-noise cases:", high)
draw(high, os.path.join(OUT, "paired_high_noise"), "$\\sigma^2$ = 2.5: runs that only the wrapped bearing loses (median MSE of these runs)")
