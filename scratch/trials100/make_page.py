"""Preview page of Tables I, II and III with 100 trials (task T3, D-34), for claude.ai.
Reads the 'Error summary.txt' files (run 'main.py errors 1 100' first) and, for the comparison, the 20/30-trial values that are
embedded in the earlier preview pages scratch/d30_pilot/page/{revised_tables,table_II}.html.
Writes scratch/trials100/page/tables_100.html.  Usage (repository root): python scratch/trials100/make_page.py"""
import json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gaussfiltax"))
from main import read_error_summary  # noqa: E402

EXP = ROOT + "/data.nosync/experiments/"
A_DIR = "mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_{a}_sigma2_{s}_wrap/"
B_DIR = "lmsv/state-dim_4_emission-dim_4_num-timesteps_200/"
NEW = {"RPF", "UPF"}
PF = lambda f: (f, "pf/" + f.lower() + "/numprt_{M}_ess_0.5" + ("_alpha_1.0_beta_0.0_kappa_0.0" if f == "UPF" else ""))
A_FAM = [("L-GSF", "gsf/lgsf/ncomps_{M}_iters_1", [(1, "M=1 (EKF)"), (100, "M=100"), (1000, "M=1000")]),
         ("U-GSF", "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0", [(1, "M=1 (UKF)"), (100, "M=100"), (1000, "M=1000")]),
         (*PF("BPF"), [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000"), (100000, "M=100000")]),
         (*PF("APF"), [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000"), (100000, "M=100000")]),
         (*PF("RPF"), [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000")]),
         (*PF("UPF"), [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000")]),
         ("L-AGSF", "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [(2, "M=2, N=5, L=5"), (10, "M=10, N=5, L=5"), (100, "M=100, N=5, L=5")]),
         ("U-AGSF", "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_prop_r2_0.9", [(2, "M=2, N=5, L=5"), (10, "M=10, N=5, L=5"), (100, "M=100, N=5, L=5")])]
B_PF = [(100, "M=100"), (1000, "M=1000"), (10000, "M=10000")]
B_MIX = [(10, "M=10"), (100, "M=100"), (1000, "M=1000")]
B_AG = [(10, "M=10, N=5, L=5"), (100, "M=100, N=5, L=5"), (1000, "M=1000, N=5, L=5")]
B_FAM = [("L-GSF", "gsf/lgsf/ncomps_{M}_iters_1", B_MIX),
         ("U-GSF", "gsf/ugsf/ncomps_{M}_iters_1_alpha_1.0_beta_0.0_kappa_0.0", B_MIX),
         (*PF("BPF"), B_PF), (*PF("APF"), B_PF), (*PF("RPF"), B_PF), (*PF("UPF"), B_PF),
         ("L-AGSF", "agsf/lagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4", B_AG),
         ("U-AGSF", "agsf/uagsf/M_{M}_N_5_L_5_iters_1_ac1_prop_r1_0.9_ac2_auto_r2_0.4", B_AG)]
SIGMA2 = ["2.5", "0.025", "2.5e-05"]


def cell(path):
    v = read_error_summary(path + "Error summary.txt")
    return {"n": v["trials"], "lost": v["lost"], "mse": list(v["mse"][:2]), "lpe": list(v["lpe"][:2]),
            "outside": list(v["outside"][:2]), "empty": int(v["empty"].split()[0])}


def old_data(name):
    s = open(os.path.join(ROOT, "scratch/d30_pilot/page", name)).read()
    return json.loads(re.search(r"const DATA = (\[.*?\]);\n", s).group(1))


old_A = {t["table"]: {f["name"]: {r["label"]: r["cells"] for r in f["rows"]} for f in t["families"]} for t in old_data("revised_tables.html")}
old_B = {f["name"]: {r["label"]: r for r in f["rows"]} for f in old_data("table_II.html")}
keep = lambda c: {k: c[k] for k in ("n", "lost", "mse", "lpe", "outside", "empty") if k in c}

tables = []
for tab, a in (("I", "0.5"), ("III", "0.05")):
    fams = []
    for name, folder, sizes in A_FAM:
        rows = []
        for M, lab in sizes:
            cells = [cell(EXP + A_DIR.format(a=a, s=s) + folder.format(M=M) + "/") for s in SIGMA2]
            prev = old_A.get(tab, {}).get(name, {}).get(lab)
            for c, p in zip(cells, prev or [None] * 3):
                c["prev"] = keep(p) if p else None
            rows.append({"label": lab, "cells": cells})
        fams.append({"name": name, "new": name in NEW, "rows": rows})
    tables.append({"table": tab, "exp": "A", "a": a, "families": fams})

fams = []
for name, folder, sizes in B_FAM:
    rows = []
    for M, lab in sizes:
        c = cell(EXP + B_DIR + folder.format(M=M) + "/")
        p = old_B.get(name, {}).get(lab)
        c["prev"] = keep(p) if p else None
        rows.append({"label": lab, "cells": [c]})
    fams.append({"name": name, "new": name in NEW, "rows": rows})
tables.append({"table": "II", "exp": "B", "families": fams})

ns = sorted({c["n"] for t in tables for f in t["families"] for r in f["rows"] for c in r["cells"]})
print("trials in the summaries:", ns)
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, "page_template.html")).read()
os.makedirs(os.path.join(here, "page"), exist_ok=True)
out = os.path.join(here, "page", "tables_100.html")
open(out, "w").write(tpl.replace("/*DATA*/[]", json.dumps(tables)))
print("written", out)
