"""D-30: Tables I and III with the raw bearing (old) and the wrapped bearing (new). Debug only (rule 5).
GSF and AGSF: T1 and the D-30 pilot (SEED_BASE 2000, 20 trials). BPF and APF: A2 and T4 (SEED_BASE 1000, 30 trials).
RPF and UPF: A1 and T4 (SEED_BASE 3000, 30 trials). The old and new runs of a trial have the same states and filter
keys (checked: scratch/d30_pilot/README.md). Reads the Error summary.txt files (bootstrap mean +- sd) and the
per-trial mse.pkl.
Usage: python scratch/d30_pilot/compare.py [--all] > scratch/d30_pilot/compare.md   (--all: also the PF rows)"""
import os, sys, glob, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax")); os.chdir(ROOT)
import numpy as np

E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}{w}"
ROWS = [("L-GSF", "gsf/lgsf/ncomps_{M}_iters_1", [1, 100, 1000]),
        ("U-GSF", "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0", [1, 100, 1000]),
        ("L-AGSF", "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [2, 10, 100]),
        ("U-AGSF", "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [2, 10, 100])]
if "--all" in sys.argv:
    ROWS += [("BPF", "pf/bpf/numprt_{M}_ess_0.5", [100, 1000, 10000, 100000]),
             ("APF", "pf/apf/numprt_{M}_ess_0.5", [100, 1000, 10000, 100000]),
             ("RPF", "pf/rpf/numprt_{M}_ess_0.5", [100, 1000, 10000]),
             ("UPF", "pf/upf/numprt_{M}_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0", [100, 1000, 10000])]


def summary(path):
    out = {}
    for line in open(path):
        if line.startswith("trials:"):
            out["n"] = int(line.split(":")[1])
        if line.startswith("trials with mse > 10:"):
            out["lost"] = int(line.split(":")[1])
        for k in ("mse", "lpe"):
            if line.startswith(k + ":"):
                v = line.split(":")[1].split(",")[0].split("+-")
                out[k] = (float(v[0]), float(v[1]))
    return out


def per_trial(d, name, n):
    return np.array([float(pickle.load(open(f"{d}/{t}/{name}.pkl", "rb"))[0]) for t in range(1, n + 1)])


fmt = lambda x: f"{x:.2e}" if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2) else f"{x:.2f}"
for a, table in [("0.5", "I"), ("0.05", "III")]:
    for s in ["2.5", "0.025", "2.5e-05"]:
        print(f"\n### Table {table}, a = {a}, sigma2 = {s}\n")
        print("| Filter | M | MSE old → new | LPE old → new | Lost old → new (trials) |")
        print("|---|---|---|---|---|")
        for name, folder, Ms in ROWS:
            for M in Ms:
                d0 = f"{E.format(a=a, s=s, w='')}/{folder.format(M=M)}"
                d1 = f"{E.format(a=a, s=s, w='_wrap')}/{folder.format(M=M)}"
                try:
                    o, n = summary(d0 + "/Error summary.txt"), summary(d1 + "/Error summary.txt")
                    assert o["n"] == n["n"], (d0, o["n"], n["n"])
                except (FileNotFoundError, OSError):
                    print(f"| {name} | {M} | missing | | |"); continue
                print(f"| {name} | {M} | {fmt(o['mse'][0])} → {fmt(n['mse'][0])} | {fmt(o['lpe'][0])} → {fmt(n['lpe'][0])} "
                      f"| {o['lost']} → {n['lost']} ({n['n']}) |")
