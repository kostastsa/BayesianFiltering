#!/bin/zsh
# Task A3: run time of each filter without the JIT compile time. Run in the repository root, in jaxenv, with no other job.
# Descriptor files: the '*_timing.txt' files ('python gaussfiltax/create_exps.py timing <files>'). Each experiment runs on the
# stored data of trials 1 to 5 of its table row (data block). GAUSSFILTAX_WARM_TIME=1: the runner calls the filter two times
# through one jit object and saves the time of the second call (experiment_runner._timed).
#
# Usage: ./run_timing.zsh FIRST LAST FILE [FILE ...]     e.g.  ./run_timing.zsh 1 5 exps/lmsv/state_dim_4/*_timing.txt
# One file at a time. After each file, results.pkl and aux_outputs.pkl of its runs are deleted (runtime.txt stays).
FIRST=$1; LAST=$2; shift 2
SEED_BASE=${SEED_BASE:-8000}
setopt nullglob   # a run without aux_outputs.pkl must not stop the rm line
i=0
for f in "$@"; do
    i=$((i + 1))
    echo "$(date +%T) $f: trials $FIRST to $LAST"
    GAUSSFILTAX_WARM_TIME=1 python gaussfiltax/main.py trials $((SEED_BASE + i)) $FIRST $LAST "$f" > /dev/null 2>&1 \
        || { echo "STOP: trials failed for $f"; exit 1 }
    dirs=( ${(f)"$(python -c '
import os, sys
sys.path.insert(0, "gaussfiltax")
import experiment_descriptor as ed, util.io, misc
for e in ed.parse(util.io.load_txt(sys.argv[1])):
    print(os.path.join(misc.get_root(), "experiments", e.get_dir()))
' "$f" 2>/dev/null)"} )
    for d in $dirs; do
        rm -f $d/*/results.pkl $d/*/aux_outputs.pkl
        echo "   $(basename $d): $(cat $d/$FIRST/runtime.txt | head -1)"
    done
done
echo "$(date +%T) TIMING DONE"
