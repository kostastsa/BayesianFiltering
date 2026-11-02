"""Size of the effect of the branch cut on the GSF and AGSF rows of Tables I and III. Debug only (rule 5).
Uses the saved per-step metrics (lpe.pkl, mse.pkl: (mean, array over time)) and gt.pkl. No filter runs.
(1) Row mean of MSE and LPE over all trials, and without the lost trials (MSE > 10) whose divergence starts within
    10 steps after a crossing of the cut by the true bearing.
(2) In the trials that are not lost: share of the summed per-step LPE in the windows [crossing, crossing + 30]
    against the share of the steps in these windows. A ratio above 1 means extra error after crossings."""
import os, sys, glob, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import numpy as np
import util.io

W = 30
print(f"{'setting':20s} {'row':18s} {'MSE all':>9s} {'MSE -cut':>9s} {'LPE all':>10s} {'LPE -cut':>10s} {'cut-lost':>8s} {'LPE share/step share (not lost)':>32s}")
for exp in sorted(glob.glob("data.nosync/experiments/mtt/*/*sf/*/*/")):
    parts = exp.rstrip("/").split("/")
    setting = parts[3].split("timesteps_500_")[1]
    row = parts[5] + " " + parts[6].split("_iters")[0].replace("ncomps_", "M=").replace("M_", "M=").split("_N")[0]
    mse_all, lpe_all, keep_mse, keep_lpe, ncut = [], [], [], [], 0
    lpe_in = lpe_tot = steps_in = steps_tot = 0.0
    for d in sorted(glob.glob(exp + "*/")):
        if not os.path.exists(d + "mse.pkl"):
            continue
        mse, mse_t = pickle.load(open(d + "mse.pkl", "rb")); lpe, lpe_t = pickle.load(open(d + "lpe.pkl", "rb"))
        mse_t, lpe_t = np.asarray(mse_t), np.asarray(lpe_t)
        states = np.asarray(util.io.load(d + "gt")[0])
        bt = np.arctan2(states[:, 2], states[:, 0])
        cross = np.where(np.abs(np.diff(bt)) > np.pi)[0] + 1
        mse_all.append(float(mse)); lpe_all.append(float(lpe))
        lost = float(mse) > 10
        t1 = int(np.argmax(mse_t > 1)) if lost else None
        at_cut = lost and any(0 <= t1 - c <= 10 for c in cross)
        ncut += at_cut
        if not at_cut:
            keep_mse.append(float(mse)); keep_lpe.append(float(lpe))
        if not lost:
            mask = np.zeros(len(bt), bool)
            for c in cross:
                mask[c:c + W] = True
            lpe_in += lpe_t[mask].sum(); lpe_tot += lpe_t.sum(); steps_in += mask.sum(); steps_tot += len(bt)
    ratio = (lpe_in / lpe_tot) / (steps_in / steps_tot) if steps_in and lpe_tot > 0 else float("nan")
    share = f"{lpe_in / lpe_tot:5.2f} / {steps_in / steps_tot:4.2f} = {ratio:5.2f}" if steps_in and lpe_tot > 0 else "n/a (LPE sum <= 0)"
    print(f"{setting:20s} {row:18s} {np.mean(mse_all):9.3g} {np.mean(keep_mse):9.3g} {np.mean(lpe_all):10.3g} {np.mean(keep_lpe):10.3g} {ncut:8d} {share:>32s}")
