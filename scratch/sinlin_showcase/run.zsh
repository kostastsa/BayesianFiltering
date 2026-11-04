#!/bin/zsh
# Showcase of plan step 6 (see make_descs.py). Run in the repository root, in jaxenv.
# Usage: ./scratch/sinlin_showcase/run.zsh FIRST LAST [TAGS]     (TAGS default: d5,d10,d20,d50,d10_rot,d20_rot)
# Each tag runs in its own process, at the same time: the EKF data source (seed 9100 + index), then the other filters on
# its stored data (seed 9200 + index, only the filter keys), then 'main.py errors --no-plots' for both files.
FIRST=$1; LAST=$2
TAGS=(${(s:,:)${3:-d5,d10,d20,d50,d10_rot,d20_rot}})
D=scratch/sinlin_showcase/descs
L=scratch/sinlin_showcase/out; mkdir -p $L
i=0
for t in $TAGS; do
    ( python gaussfiltax/main.py trials $((9100 + i)) $FIRST $LAST "$D/src_$t.txt" > $L/src_$t.log 2>&1 \
      && python gaussfiltax/main.py trials $((9200 + i)) $FIRST $LAST "$D/filters_$t.txt" > $L/trials_$t.log 2>&1 \
      && python gaussfiltax/main.py errors --no-plots 1 $LAST "$D/src_$t.txt" "$D/filters_$t.txt" > $L/errors_$t.log 2>&1 \
      && echo "$(date +%T) $t DONE" || echo "$(date +%T) $t FAILED" ) &
    i=$((i + 1))
done
wait
echo "$(date +%T) SHOWCASE ALL DONE"
