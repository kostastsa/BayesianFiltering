#!/bin/zsh
# Metrics of the complete trial 66 (a = 0.05, BPF/APF), then delete its large results (k > 30). One-time fix of 03-Nov-2026.
cd '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax'
nice -n 10 python gaussfiltax/main.py errors --no-plots 66 66 exps/mtt/state_dim_4/a_0.05_sigma2_*/mtt_{bpf,apf}_wrap_scan_num_prt.txt >> data.nosync/logs/T100_fix_errors_a0.05_t66.log 2>&1
n=$(ls data.nosync/experiments/mtt/*_a_0.05_sigma2_*_wrap/pf/{bpf,apf}/*/66/nees.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "$(date +%T) trial 66 a=0.05: nees.pkl in $n of 24 runs"
[[ $n == 24 ]] && rm -f data.nosync/experiments/mtt/*_a_0.05_sigma2_*_wrap/pf/{bpf,apf}/*/66/results.pkl
