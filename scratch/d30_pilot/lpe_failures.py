"""Failure counts based on the LPE, over the six settings of Tables I and III (wrapped bearing). Debug only (rule 5).
Per-trial LPE from the lpe.pkl files of 'main.py errors'. A trial fails if
  (abs) its LPE is above T_ABS, or
  (rel) its LPE is more than T_REL above the smallest median LPE of all rows in the same setting.
Usage: python scratch/d30_pilot/lpe_failures.py [raw]   (raw: the old runs without the wrap)"""
import os, sys, pickle, glob
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
W = "" if "raw" in sys.argv else "_wrap"
E = "data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}" + W + "/{f}"
ROWS = [("L-GSF", "gsf/lgsf/ncomps_{M}_iters_1", [1, 100, 1000]),
        ("U-GSF", "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0", [1, 100, 1000]),
        ("L-AGSF", "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [2, 10, 100]),
        ("U-AGSF", "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [2, 10, 100]),
        ("BPF", "pf/bpf/numprt_{M}_ess_0.5", [100, 1000, 10000, 100000]),
        ("APF", "pf/apf/numprt_{M}_ess_0.5", [100, 1000, 10000, 100000]),
        ("RPF", "pf/rpf/numprt_{M}_ess_0.5", [100, 1000, 10000]),
        ("UPF", "pf/upf/numprt_{M}_ess_0.5_alpha_1.0_beta_0.0_kappa_0.0", [100, 1000, 10000])]
T_ABS, T_REL = 10.0, 10.0
SETTINGS = [(a, s) for a in ["0.5", "0.05"] for s in ["2.5", "0.025", "2.5e-05"]]

lpe = {}
for a, s in SETTINGS:
    for name, folder, Ms in ROWS:
        for M in Ms:
            d = E.format(a=a, s=s, f=folder.format(M=M))
            files = sorted(glob.glob(d + "/*/lpe.pkl"))
            lpe[(a, s, name, M)] = np.array([float(pickle.load(open(f, "rb"))[0]) for f in files])
best = {(a, s): min(np.median(v) for k, v in lpe.items() if k[:2] == (a, s)) for a, s in SETTINGS}

def counts(keys):
    n = fa = fr = 0
    for k in keys:
        v = lpe[k]; n += len(v); fa += int(np.sum(v > T_ABS)); fr += int(np.sum(v > best[k[:2]] + T_REL))
    return n, fa, fr

print(f"Bearing: {'raw' if W == '' else 'wrapped'}. Best median LPE per setting: " + ", ".join(f"a={a} s2={s}: {best[(a, s)]:.1f}" for a, s in SETTINGS))
print("\n| Family | trials | failed, LPE > 10 | failed, > best + 10 |\n|---|---|---|---|")
fam_rows = []
for name, _, Ms in ROWS:
    n, fa, fr = counts([k for k in lpe if k[2] == name])
    fam_rows.append((fr / n, name, n, fa, fr))
for _, name, n, fa, fr in sorted(fam_rows, reverse=True):
    print(f"| {name} | {n} | {fa} ({100 * fa / n:.0f} %) | {fr} ({100 * fr / n:.0f} %) |")
print("\n| Row | trials | LPE > 10 | > best + 10 | by setting (a=0.5: 2.5 / 0.025 / 2.5e-05 ; a=0.05: same), rel. |\n|---|---|---|---|---|")
for name, _, Ms in ROWS:
    for M in Ms:
        ks = [(a, s, name, M) for a, s in SETTINGS]
        n, fa, fr = counts(ks)
        per = [int(np.sum(lpe[k] > best[k[:2]] + T_REL)) for k in ks]
        print(f"| {name} M={M} | {n} | {100 * fa / n:.0f} % | {100 * fr / n:.0f} % | {per[0]}/{per[1]}/{per[2]} ; {per[3]}/{per[4]}/{per[5]} |")
