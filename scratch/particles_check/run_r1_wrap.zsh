#!/bin/zsh
# Table R1 with the wrapped bearing (D-33), all steps of README.md in order. Run in the repository root, in jaxenv.
# Ours: 20 runs on each of the 30 data sets (run_samedata.zsh, files mtt_{bpf,apf}_wrap_samedata_*), one setting at a time.
# A setting whose Error summary.txt has 600 trials is skipped, so a restart does not repeat it (its results.pkl are deleted).
# Then export_gt.py, run_particles.py (~/.venvs/particles) and make_table.py, all with the argument 'wrap'.
# Output: scratch/particles_check/out/table_wrap.{md,tex}.
set -e
S=(2.5 0.025 2.5e-05)
E=data.nosync/experiments/mtt/state-dim_4_emission-dim_2_num-timesteps_500_a_0.5_sigma2_
for f in bpf apf; do
    seed=6000; [[ $f == apf ]] && seed=6100
    i=0
    for s in $S; do
        i=$((i + 1))
        summ="${E}${s}_wrap/data_pf_${f}_numprt_1000_ess_0.5/pf/$f/numprt_1000_ess_0.5/Error summary.txt"
        if [[ -f $summ ]] && grep -q "^trials: 600" "$summ"; then echo "$(date +%T) $f sigma2=$s complete, skipped"; continue; fi
        DELETE_RESULTS=1 SEED_BASE=$((seed + i - 1)) ./run_samedata.zsh 20 0.5 $s ${f}_wrap
    done
done
for s in $S; do
    python scratch/particles_check/export_gt.py 0.5 $s wrap
    [[ -f scratch/particles_check/out/particles_a0.5_sigma2_${s}_wrap.npz ]] || ~/.venvs/particles/bin/python scratch/particles_check/run_particles.py 0.5 $s wrap
done
python scratch/particles_check/make_table.py 0.5 ${(j:,:)S} wrap
echo "$(date +%T) R1 WRAP DONE"
