"""Comparison scatter of stored lost runs (MSE > 10) of the GSF, AGSF and particle filters in Exp. A.

Debug view only, not an experiment result (CLAUDE.md, rule 5). No filter runs again: each panel shows the saved
results.pkl and gt.pkl of one trial. Each panel has its own data, because each experiment has its own data.
Rows: sigma2 = 2.5 with a = 0.5 (Table I) and a = 0.05 (Table III). Columns: one filter each, with the largest size
that has a lost run with a saved results.pkl (the PF runs with 10^4 and 10^5 particles have no results.pkl).
In each panel the trial is the lost trial with the median MSE (a typical failure, not the worst one).

Usage (repository root, jaxenv):  python scratch/agsf_lost_tracks/lost_runs_scatter.py
"""
import glob, os, pickle, sys
import matplotlib
matplotlib.use("Agg")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import standalone_filter_runner as sfr
import util.io

EXP = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_2.5"
SPLIT = "N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9"
UKF = "alpha_1.0_beta_0.0_kappa_0.0"
# (column name, size label, candidate folders in order of preference)
COLUMNS = [
    ("L-GSF", ["gsf/lgsf/ncomps_1000_iters_1", "gsf/lgsf/ncomps_100_iters_1"]),
    ("U-GSF", [f"gsf/ugsf/ncomps_1000_iters_1_{UKF}", f"gsf/ugsf/ncomps_100_iters_1_{UKF}"]),
    ("L-AGSF", [f"agsf/lagsf/M_100_{SPLIT}", f"agsf/lagsf/M_10_{SPLIT}"]),
    ("U-AGSF", [f"agsf/uagsf/M_100_{SPLIT}", f"agsf/uagsf/M_10_{SPLIT}"]),
    ("BPF", ["pf/bpf/numprt_1000_ess_0.5", "pf/bpf/numprt_100_ess_0.5"]),
    ("APF", ["pf/apf/numprt_1000_ess_0.5", "pf/apf/numprt_100_ess_0.5"]),
    ("RPF", ["pf/rpf/numprt_1000_ess_0.5", "pf/rpf/numprt_100_ess_0.5"]),
    ("UPF", [f"pf/upf/numprt_1000_ess_0.5_{UKF}", f"pf/upf/numprt_100_ess_0.5_{UKF}"]),
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def size_label(folder):
    tail = folder.split("/")[-1]
    return "M=" + tail.split("_")[1]


def median_lost_trial(exp_dir):
    """(trial, mse) of the lost trial with the median MSE among the trials with a saved results.pkl, or None."""
    lost = []
    for f in glob.glob(f"{exp_dir}/*/mse.pkl"):
        d = os.path.dirname(f)
        mse = float(pickle.load(open(f, "rb"))[0])
        if mse > 10 and os.path.exists(f"{d}/results.pkl"):
            lost.append((mse, int(os.path.basename(d))))
    if not lost:
        return None
    lost.sort()
    mse, trial = lost[(len(lost) - 1) // 2]
    return trial, mse


rows, states_rows, titles, info = [], [], [], []
for a in ["0.5", "0.05"]:
    row, states, title = {}, {}, {}
    for name, folders in COLUMNS:
        for folder in folders:
            exp_dir = f"{EXP.format(a=a)}/{folder}"
            pick = median_lost_trial(exp_dir)
            if pick is None:
                continue
            trial, mse = pick
            res = util.io.load(f"{exp_dir}/{trial}/results")
            row[name] = res[0] if isinstance(res, tuple) else res
            states[name] = util.io.load(f"{exp_dir}/{trial}/gt")[0]
            title[name] = f"{name} {size_label(folder)}\ntrial {trial}, MSE {mse:.1f}"
            info.append(f"a={a}  {name:7s} {size_label(folder):7s} trial {trial:2d}  MSE {mse:7.1f}  {exp_dir}/{trial}")
            break
        else:
            title[name] = f"{name}\nno lost run with results"
            info.append(f"a={a}  {name:7s} no lost run with a saved results.pkl")
    rows.append(row); states_rows.append(states); titles.append(title)

print("\n".join(info))
# The FilterRunner is needed only for the plot method. Its own simulation is not used.
runner = sfr.FilterRunner(sfr.MTTConfig(R_scale=2.5, acc=0.5), num_timesteps=10, seed=0)
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ("png", "pdf"):
    runner.plot_three_panel_scatter(
        rows, dims=(0, 2), states_list=states_rows, panel_titles=titles,
        row_titles=["a = 0.5, sigma2 = 2.5", "a = 0.05, sigma2 = 2.5"],
        algorithm_order=[c[0] for c in COLUMNS], rasterize_points=True,
        save_path=os.path.join(OUT_DIR, f"lost_runs_sigma2_2.5.{ext}"), show=False)
print("saved", os.path.join(OUT_DIR, "lost_runs_sigma2_2.5.{png,pdf}"))
