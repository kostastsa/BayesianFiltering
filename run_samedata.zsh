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
    if [[ -n $DELETE_RESULTS ]]; then
        # DELETE_RESULTS=1: chunks of 60 trials. After the 'errors' of each chunk, delete its results.pkl (Table R1 needs only
        # mse.pkl). The peak disk use is then about 0.7 GB instead of 7 GB. 'errors 1 LAST' at the end reads the saved metrics.
        d=$(python -c 'import os, sys; sys.path.insert(0, "gaussfiltax"); import experiment_descriptor as ed, util.io, misc; print(os.path.join(misc.get_root(), "experiments", ed.parse(util.io.load_txt(sys.argv[1]))[0].get_dir()))' $f 2>/dev/null)
        for c in $(seq 1 60 $LAST); do
            ce=$((c + 59)); (( ce > LAST )) && ce=$LAST
            # skip a chunk whose trials all have mse.pkl (a restart)
            n_done=0; for t in $(seq $c $ce); do [[ -f $d/$t/mse.pkl ]] && n_done=$((n_done + 1)); done
            (( n_done == ce - c + 1 )) && continue
            # 'main.py trials' starts the key sequence at the first trial of the call. Thus each chunk needs its own seed.
            # With one seed, trial c + k gets the key of trial 1 + k and (60 = 2 x 30) the same data set: an exact copy.
            # Chunk 1 keeps the seed SEED_BASE + i. Chunk n > 1 uses (SEED_BASE + i) * 1000 + n.
            n_chunk=$(( (c - 1) / 60 + 1 ))
            seed=$((SEED_BASE + i)); (( n_chunk > 1 )) && seed=$(( (SEED_BASE + i) * 1000 + n_chunk ))
            for attempt in 1 2 3; do
                python gaussfiltax/main.py trials $seed $c $ce $f && break
                echo "$(date +%T) trials $c-$ce crashed (attempt $attempt)"
                [[ $attempt == 3 ]] && { echo "STOP: trials failed"; exit 1 }
            done
            python gaussfiltax/main.py errors --no-plots $c $ce $f || { echo "STOP: errors failed"; exit 1 }
            rm -f $d/*/results.pkl
            echo "$(date +%T) trials $c-$ce done, results.pkl deleted"
        done
        python gaussfiltax/main.py errors --no-plots 1 $LAST $f || { echo "STOP: errors failed"; exit 1 }
    else
        for attempt in 1 2 3; do
            python gaussfiltax/main.py trials $((SEED_BASE + i)) 1 $LAST $f && break
            echo "$(date +%T) trials crashed (attempt $attempt)"
            [[ $attempt == 3 ]] && { echo "STOP: trials failed"; exit 1 }
        done
        python gaussfiltax/main.py errors --no-plots 1 $LAST $f || { echo "STOP: errors failed"; exit 1 }
    fi
    echo "$(date +%T) DONE a=$A sigma2=$s"
done
echo "$(date +%T) ALL DONE"
