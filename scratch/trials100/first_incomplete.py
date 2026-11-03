"""First incomplete trial of a run_tables_I_III.zsh job (D-34). Prints one number.
A trial is complete when every run of the job's descriptor files has mse.pkl and nees.pkl.
Usage (repository root): python scratch/trials100/first_incomplete.py SIM A NAMES START
  SIM: mtt or lmsv. A: 0.5 or 0.05 (mtt; ignored for lmsv). NAMES: as in run_tables_I_III.zsh, e.g. bpf_wrap,apf_wrap.
  START: the first trial of the job (21 for GSF/AGSF, 31 for PFs)."""
import os, sys
sys.path.insert(0, "gaussfiltax")
import experiment_descriptor as ed, util.io, misc

sim, a, names, start = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
if sim == "lmsv":
    files = [f"exps/lmsv/state_dim_4/lmsv_{n}_scan_num_prt.txt" for n in names]
else:
    files = [f"exps/mtt/state_dim_4/a_{a}_sigma2_{s}/mtt_{n}_scan_num_prt.txt" for s in ("2.5", "0.025", "2.5e-05") for n in names]
dirs = [os.path.join(misc.get_root(), "experiments", e.get_dir()) for f in files for e in ed.parse(util.io.load_txt(f))]
k = start
while all(os.path.exists(f"{d}/{k}/mse.pkl") and os.path.exists(f"{d}/{k}/nees.pkl") for d in dirs):
    k += 1
print(k)
