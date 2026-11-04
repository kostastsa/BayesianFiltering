#!/bin/zsh
# Pilot A8 (D-39, see make_descs.py). Run in the repository root, in jaxenv.
# Usage: ./scratch/agsf_sdp_pilot/run_pilot.zsh [N]     (trials 1 to N, default 5)
# Exp. A and Exp. B run at the same time, each with its own seed (9000, 9001). Then 'main.py errors' for each file.
# Each trial t uses the stored data of trial t of the reference row (data block), so the seed changes only the filter keys.
N=${1:-5}
D=scratch/agsf_sdp_pilot/descs
L=scratch/agsf_sdp_pilot/out; mkdir -p $L
i=0
for e in expA expB; do
    ( python gaussfiltax/main.py trials $((9000 + i)) 1 $N "$D/pilot_$e.txt" > $L/trials_$e.log 2>&1 \
      && python gaussfiltax/main.py errors --no-plots 1 $N "$D/pilot_$e.txt" > $L/errors_$e.log 2>&1 \
      && echo "$(date +%T) $e DONE" || echo "$(date +%T) $e FAILED" ) &
    i=$((i + 1))
done
wait
echo "$(date +%T) PILOT ALL DONE"
