#!/bin/zsh
# Version 1 (see common.py): repeat the selected table trials exactly into data.nosync/repro (v1_jobs.py), then compute
# their metrics there with 'main.py errors'. The table folders do not change. Run in the repository root, in jaxenv.
# Usage: ./scratch/scatter_panels/run_v1.zsh
set -e
export GAUSSFILTAX_ROOT=$PWD/data.nosync/repro
python scratch/scatter_panels/v1_jobs.py > scratch/scatter_panels/out/v1_jobs.txt
echo "$(date +%T) $(wc -l < scratch/scatter_panels/out/v1_jobs.txt) calls"
while read -r seed trial files only; do
    args=(); for o in ${(s:,:)only}; do args+=(--only "$o"); done
    python gaussfiltax/main.py trials $seed $trial $trial ${(s:,:)files} $args > /dev/null 2>&1 || { echo "STOP: trials $seed $trial"; exit 1 }
done < scratch/scatter_panels/out/v1_jobs.txt
echo "$(date +%T) V1 RUNS DONE"
