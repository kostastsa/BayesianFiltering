"""Comparison scatter of stored lost runs (MSE > 10) of the AGSF with M = 2 in Exp. A. Debug only (rule 5).
No filter runs again. The blue dashed line is the branch cut of the bearing arctan2(x2, x1) (negative x1-axis).
Figure 1 (sigma2 = 2.5): rows L-AGSF and U-AGSF, columns: lost trials at the 25th, 50th and 75th percentile of the MSE,
for a = 0.5 and a = 0.05. Figure 2 (sigma2 = 0.025 and 2.5e-05): all lost runs of the AGSF with M = 2.
Usage (repository root, jaxenv):  python scratch/agsf_lost_tracks/m2_lost_runs.py
"""
import glob, os, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import standalone_filter_runner as sfr
import util.io

E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}/agsf/{f}/M_2_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"
NAME = {"lagsf": "L-AGSF", "uagsf": "U-AGSF"}
OUT = os.path.join(os.path.dirname(__file__), "figures")


def lost_trials(exp):
    out = []
    for f in glob.glob(exp + "/*/mse.pkl"):
        mse = float(pickle.load(open(f, "rb"))[0])
        if mse > 10:
            out.append((mse, int(os.path.basename(os.path.dirname(f)))))
    return sorted(out)


def panel(a, s, f, trial, mse):
    d = f"{E.format(a=a, s=s, f=f)}/{trial}"
    return util.io.load(d + "/results"), util.io.load(d + "/gt")[0], f"{NAME[f]} M=2, a={a}, $\\sigma^2$={s}\ntrial {trial}, MSE {mse:.1f}"


def draw(rows, path, row_titles):
    """rows: list of lists of (results, states, title). One panel per entry, key = column index."""
    ncol = max(len(r) for r in rows)
    keys = [f"c{j}" for j in range(ncol)]
    res = [{keys[j]: p[0] for j, p in enumerate(r)} for r in rows]
    sts = [{keys[j]: p[1] for j, p in enumerate(r)} for r in rows]
    tit = [{keys[j]: p[2] for j, p in enumerate(r)} for r in rows]
    runner.plot_three_panel_scatter(res, dims=(0, 2), states_list=sts, panel_titles=tit, row_titles=row_titles,
                                    algorithm_order=keys, rasterize_points=True, show=True)
    fig = plt.gcf()
    for ax in fig.axes:
        if ax.has_data():
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            if x0 < 0:
                ax.plot([x0, 0], [0, 0], "b--", lw=1.0, zorder=11)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        for t in ax.texts:  # smaller titles: the default 14 pt bold is too large for two lines
            pass
        ax.title.set_fontsize(10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".{png,pdf}")


runner = sfr.FilterRunner(sfr.MTTConfig(R_scale=2.5, acc=0.5), num_timesteps=10, seed=0)
os.makedirs(OUT, exist_ok=True)

# Figure 1: sigma2 = 2.5, three lost trials per (filter, a) at the 25th, 50th and 75th percentile of the MSE
rows = []
for f in ["lagsf", "uagsf"]:
    row = []
    for a in ["0.5", "0.05"]:
        lost = lost_trials(E.format(a=a, s="2.5", f=f))
        for q in (0.25, 0.5, 0.75):
            mse, trial = lost[int(q * (len(lost) - 1))]
            row.append(panel(a, "2.5", f, trial, mse))
            print(f"fig1 {f} a={a} q={q}: trial {trial}, MSE {mse:.1f}")
    rows.append(row)
draw(rows, os.path.join(OUT, "m2_lost_runs_sigma2_2.5"), ["L-AGSF, M = 2", "U-AGSF, M = 2"])

# Figure 2: all lost runs with M = 2 at sigma2 = 0.025 and 2.5e-05
row = []
for s in ["0.025", "2.5e-05"]:
    for a in ["0.5", "0.05"]:
        for f in ["lagsf", "uagsf"]:
            for mse, trial in lost_trials(E.format(a=a, s=s, f=f)):
                row.append(panel(a, s, f, trial, mse))
                print(f"fig2 {f} a={a} s2={s}: trial {trial}, MSE {mse:.1f}")
draw([row], os.path.join(OUT, "m2_lost_runs_low_noise"), ["AGSF, M = 2"])
