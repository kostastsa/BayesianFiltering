#!/bin/zsh
# Version 2: run the selected trials (v2_select.py) again, with the seed of run_v2.zsh (9500 + k). The same key gives the
# same results, and results.pkl stays for the plot. Then 'main.py errors' over all trials restores the summaries.
# Usage (repository root, jaxenv): ./scratch/scatter_panels/run_v2_select.zsh
set -e
python scratch/scatter_panels/v2_select.py > scratch/scatter_panels/out/v2_jobs.txt
cp scratch/scatter_panels/out/v2_selection.json scratch/scatter_panels/out/v2_selection_before.json
while read -r f t1 t2 t3; do
    for k in $t1 $t2 $t3; do
        python gaussfiltax/main.py trials $((9500 + k)) $k $k $f > /dev/null 2>&1
        python gaussfiltax/main.py errors --no-plots $k $k $f > /dev/null 2>&1
    done
    python gaussfiltax/main.py errors --no-plots 1 40 $f > /dev/null 2>&1
    echo "$(date +%T) ${f:t}: trials $t1 $t2 $t3 again"
done < scratch/scatter_panels/out/v2_jobs.txt
python scratch/scatter_panels/v2_select.py > scratch/scatter_panels/out/v2_jobs_after.txt
diff scratch/scatter_panels/out/v2_selection_before.json scratch/scatter_panels/out/v2_selection.json && echo "same trials and MSE values after the repeat (the repeat is exact)"
