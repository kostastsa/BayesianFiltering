#!/bin/zsh
# Check of the BPF/APF against the particles library (scratch/particles_check): R runs of the filter on each of the 30 data sets
# of the main.py runs with 1000 particles. Descriptor files: exps/mtt/state_dim_4/a_<a>_sigma2_<sigma2>/mtt_<name>_samedata_scan_num_prt.txt
# (written by gaussfiltax/create_exps.py, with a data block). Trials 1..30 R: trial t uses data set ((t - 1) mod 30) + 1.
#
# Usage: ./run_samedata.zsh R A SIGMA2[,SIGMA2...] [NAME]     e.g.  ./run_samedata.zsh 20 0.5 2.5,0.025,2.5e-05 apf
#
# One 'main.py trials' and one 'main.py errors --no-plots' for each setting. The seed is SEED_BASE (default 5000) + the number of the
# setting in the list. Give each call its own SEED_BASE. Write sigma2 as in the folder names (2.5e-05, not 2.5e-5).
R=$1; A=$2; sigmas=( ${(s:,:)3} ); NAME=${4:-apf}
SEED_BASE=${SEED_BASE:-5000}
LAST=$((30 * R))
i=0
for s in $sigmas; do
    i=$((i + 1))
    f=exps/mtt/state_dim_4/a_${A}_sigma2_${s}/mtt_${NAME}_samedata_scan_num_prt.txt
    [[ -f $f ]] || { echo "STOP: no file $f"; exit 1 }
    echo "$(date +%T) a=$A sigma2=$s: trials 1 to $LAST"
    for attempt in 1 2 3; do
        python gaussfiltax/main.py trials $((SEED_BASE + i)) 1 $LAST $f && break
        echo "$(date +%T) trials crashed (attempt $attempt)"
        [[ $attempt == 3 ]] && { echo "STOP: trials failed"; exit 1 }
    done
    python gaussfiltax/main.py errors --no-plots 1 $LAST $f || { echo "STOP: errors failed"; exit 1 }
    echo "$(date +%T) DONE a=$A sigma2=$s"
done
echo "$(date +%T) ALL DONE"
