#!/bin/zsh
# Exp. A (mtt), particle filter rows (BPF, APF, RPF, UPF) of Tables I and III of the paper. The settings are a = $3 and sigma2 in {2.5, 0.025, 2.5e-5}.
# Descriptor files: exps/mtt/state_dim_4/a_<a>_sigma2_<sigma2>/mtt_{bpf,apf}_scan_num_prt.txt (written by gaussfiltax/create_exps.py).
#
# Usage: ./run_tables_I_III.zsh FIRST LAST A      e.g.  ./run_tables_I_III.zsh 1 30 0.5   (Table I)   ./run_tables_I_III.zsh 1 30 0.05   (Table III)
#
# One trial index at a time: 'main.py trials' (seed 1000 + trial, so every trial has its own data), 'main.py errors' (no plots),
# then the raw results of the runs with 10^4 and 10^5 particles of these descriptor files are deleted (1.2 GB for one run with 10^5 particles). The metrics
# stay in the trial folders. The final 'main.py errors FIRST LAST <files>' then reads the saved metrics (see Results in CLAUDE.md).
# Do not run 'main.py trials' again for a trial after the raw results are deleted: it would delete the folder and run the trial again.
# Optional: $4 = comma-separated names of the descriptor files (default bpf,apf; apf_ess2 is the APF that resamples at every step)
# and the environment variable SEED_BASE (default 1000).
FIRST=$1; LAST=$2; A=$3
names=( ${(s:,:)4:-bpf,apf} )
SEED_BASE=${SEED_BASE:-1000}
setopt nullglob
files=( exps/mtt/state_dim_4/a_${A}_sigma2_*/mtt_${^names}_scan_num_prt.txt )
echo "$(date +%T) a=$A: ${#files} descriptor files, trials $FIRST to $LAST"

# Folders of the particle filter runs with 10^4 and 10^5 particles in the descriptor files of THIS call only.
# (A glob over all pf folders would also delete the raw results of another job that runs at the same time.)
big_dirs=( ${(f)"$(python -c '
import os, sys
sys.path.insert(0, "gaussfiltax")
import experiment_descriptor as ed, util.io, misc
for f in sys.argv[1:]:
    for e in ed.parse(util.io.load_txt(f)):
        if getattr(e.inf, "num_prt", 0) in (10000, 100000):
            print(os.path.join(misc.get_root(), "experiments", e.get_dir()))
' $files 2>/dev/null)"} )
echo "$(date +%T) raw results deleted after errors in ${#big_dirs} experiment folders"

for k in {$FIRST..$LAST}; do

    for attempt in 1 2 3; do
        python gaussfiltax/main.py trials $((SEED_BASE + k)) $k $k $files && break
        echo "$(date +%T) trials $k crashed (attempt $attempt)"
        [[ $attempt == 3 ]] && { echo "STOP: trials $k failed"; exit 1 }
    done

    python gaussfiltax/main.py errors --no-plots $k $k $files || { echo "STOP: errors $k failed"; exit 1 }

    for d in $big_dirs; do
        rm -f $d/$k/results.pkl
    done
    echo "$(date +%T) DONE trial $k"
done
echo "$(date +%T) ALL DONE a=$A"
