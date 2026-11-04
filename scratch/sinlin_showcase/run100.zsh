#!/bin/zsh
# A10 (D-42): the showcase with 100 trials and the U-AGSF. Run in the repository root, in jaxenv.
# Usage: ./scratch/sinlin_showcase/run100.zsh FIRST LAST [TAGS]     (TAGS default: d5,d10,d20,d50,d10_rot,d20_rot)
# Each tag runs in its own process, at the same time. One trial at a time: the EKF data source, the other filters on its
# stored data, 'main.py errors --no-plots' for the trial, then shrink.py (eig2_mm.pkl, and for trials above 20 it deletes
# results.pkl, aux_outputs.pkl and observations.png). At the end, 'main.py errors' over all trials reads the saved metrics.
# Trials 1 to 20 exist (A9, seeds 9100 + i and 9200 + i in one call). Only the new U-AGSF runs of these trials run.
# Trial k > 20 has its own seed: (9100 + i) * 1000 + k and (9200 + i) * 1000 + k. A restart skips the finished trials.
FIRST=$1; LAST=$2
TAGS=(${(s:,:)${3:-d5,d10,d20,d50,d10_rot,d20_rot}})
D=scratch/sinlin_showcase/descs
L=scratch/sinlin_showcase/out; mkdir -p $L
i=0
for t in $TAGS; do
    ( for k in $(seq $FIRST $LAST); do
          s1=$((9100 + i)); s2=$((9200 + i))
          (( k > 20 )) && { s1=$(( (9100 + i) * 1000 + k )); s2=$(( (9200 + i) * 1000 + k )) }
          python gaussfiltax/main.py trials $s1 $k $k "$D/src_$t.txt" >> $L/src100_$t.log 2>&1 || { echo "$(date +%T) $t src $k FAILED"; exit 1 }
          python gaussfiltax/main.py trials $s2 $k $k "$D/filters_$t.txt" >> $L/trials100_$t.log 2>&1 || { echo "$(date +%T) $t trials $k FAILED"; exit 1 }
          python gaussfiltax/main.py errors --no-plots $k $k "$D/src_$t.txt" "$D/filters_$t.txt" >> $L/errors100_$t.log 2>&1 || { echo "$(date +%T) $t errors $k FAILED"; exit 1 }
          python scratch/sinlin_showcase/shrink.py $k "$D/src_$t.txt" "$D/filters_$t.txt" || { echo "$(date +%T) $t shrink $k FAILED"; exit 1 }
          echo "$(date +%T) $t trial $k done"
      done
      python gaussfiltax/main.py errors --no-plots 1 $LAST "$D/src_$t.txt" "$D/filters_$t.txt" >> $L/errors100_$t.log 2>&1 \
          && echo "$(date +%T) $t DONE" || echo "$(date +%T) $t FAILED (final errors)" ) &
    i=$((i + 1))
done
wait
echo "$(date +%T) SHOWCASE100 ALL DONE"
