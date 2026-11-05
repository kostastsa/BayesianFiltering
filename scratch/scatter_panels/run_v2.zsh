#!/bin/zsh
# Version 2 (see common.py): all 8 filters on the same stored data sets (the user chose trials 1 to 40), for each setting (make_v2_descs.py).
# One process for each setting, at the same time. For each trial k: 'main.py trials' (seed 9500 + k), 'main.py errors',
# then delete results.pkl and aux_outputs.pkl (about 60 MB for each trial). At the end, 'main.py errors 1 100'.
# Usage (repository root, jaxenv): ./scratch/scatter_panels/run_v2.zsh FIRST LAST
FIRST=$1; LAST=$2
D=scratch/scatter_panels/descs; L=scratch/scatter_panels/out; mkdir -p $L
setopt nullglob
for f in $D/paired_*.txt; do
    t=${f:t:r}
    ( dirs=( ${(f)"$(python -c '
import os, sys
sys.path.insert(0, "gaussfiltax")
import experiment_descriptor as ed, util.io, misc
for e in ed.parse(util.io.load_txt(sys.argv[1])):
    print(os.path.join(misc.get_root(), "experiments", e.get_dir()))
' $f 2>/dev/null)"} )
      for k in $(seq $FIRST $LAST); do
          python gaussfiltax/main.py trials $((9500 + k)) $k $k $f >> $L/v2_trials_$t.log 2>&1 || { echo "$(date +%T) $t trials $k FAILED"; exit 1 }
          python gaussfiltax/main.py errors --no-plots $k $k $f >> $L/v2_errors_$t.log 2>&1 || { echo "$(date +%T) $t errors $k FAILED"; exit 1 }
          for d in $dirs; do rm -f "${d:?}/$k/results.pkl" "${d:?}/$k/aux_outputs.pkl"; done
          (( k % 10 == 0 )) && echo "$(date +%T) $t trial $k done"
      done
      python gaussfiltax/main.py errors --no-plots 1 $LAST $f >> $L/v2_errors_$t.log 2>&1 && echo "$(date +%T) $t DONE" ) &
done
wait
echo "$(date +%T) V2 ALL DONE"
