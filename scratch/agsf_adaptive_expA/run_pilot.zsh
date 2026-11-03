#!/bin/zsh
# Pilot of the adaptive AGSF in Exp. A (see make_descs.py). Run in the repository root, in jaxenv.
# Usage: ./scratch/agsf_adaptive_expA/run_pilot.zsh [N]     (trials 1 to N, default 12)
# The three sigma2 files run at the same time, each with its own seed (8000, 8001, 8002). Then 'main.py errors' for each file.
# Each trial t uses the stored data of trial t of the Table I AGSF row (data block), so the seed changes only the filter keys.
N=${1:-12}
D=scratch/agsf_adaptive_expA/descs
L=scratch/agsf_adaptive_expA/out; mkdir -p $L
i=0
for s in 2.5 0.025 2.5e-05; do
    ( python gaussfiltax/main.py trials $((8000 + i)) 1 $N "$D/pilot_sigma2_$s.txt" > $L/trials_$s.log 2>&1 \
      && python gaussfiltax/main.py errors --no-plots 1 $N "$D/pilot_sigma2_$s.txt" > $L/errors_$s.log 2>&1 \
      && echo "$(date +%T) sigma2=$s DONE" || echo "$(date +%T) sigma2=$s FAILED" ) &
    i=$((i + 1))
done
wait
echo "$(date +%T) PILOT ALL DONE"
