#!/bin/zsh
# BPF and APF in the settings of Tables I and III. Small M first, so that partial results are usable at any time.
# M = 100 and 1000: 100 data sets. M = 10^4: 30. M = 10^5: 10, lowest noise only (BPF, APF and APF with the old rule).
cd "${0:A:h}/out" && mkdir -p logs || exit 1
export PYTHONWARNINGS=ignore
run() {  # a r M n [filters]
  [[ -f rerun_a$1_r$2_M$3.npz ]] && return
  FILTERS=${5:-bpf,apf,apf_every,bpf_every,bpf_pre} python3 ../expA_rerun.py $1 $2 $3 $4 > logs/rerun_a$1_r$2_M$3.log 2>&1
}
for M in 100 1000; do for a in 0.5 0.05; do for r in 2.5 0.025 2.5e-5; do run $a $r $M 100; done; done; done
for a in 0.5 0.05; do for r in 2.5e-5 0.025 2.5; do run $a $r 10000 30 bpf,apf,apf_every; done; done
for a in 0.5 0.05; do run $a 2.5e-5 100000 10 bpf,apf,apf_every; done   # the lowest noise only, where the look-ahead matters
echo finished > logs/RERUN_DONE
